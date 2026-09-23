import random

import pytest

from comfy_controller.core.policy import (
    BreakerState,
    CircuitBreaker,
    DEFAULT_CLASSIFIER,
    ErrorClassifier,
    GlobalBreaker,
    QC_MAX_RETRIES,
    RetryPolicy,
    _Rule,
)
from comfy_controller.models import AssetKind, AssetState, ErrorClass, ExecutionError


def _err(exception_type: str, exception_message: str) -> ExecutionError:
    return ExecutionError(
        prompt_id="p1",
        node_id="1",
        node_type="KSampler",
        exception_type=exception_type,
        exception_message=exception_message,
    )


# --------------------------------------------------------------------------- ErrorClassifier

CLASSIFIER_CASES = [
    # --- TEMPORARY ---
    ("torch.cuda.OutOfMemoryError", "CUDA out of memory", ErrorClass.TEMPORARY),
    ("RuntimeError", "CUDA out of memory. Tried to allocate 2GiB", ErrorClass.TEMPORARY),
    ("asyncio.TimeoutError", "", ErrorClass.TEMPORARY),
    ("RuntimeError", "request timed out", ErrorClass.TEMPORARY),
    ("ClientConnectorError", "Cannot connect to host", ErrorClass.TEMPORARY),
    ("RuntimeError", "connection refused", ErrorClass.TEMPORARY),
    ("RuntimeError", "server busy, please retry", ErrorClass.TEMPORARY),
    # --- PERMANENT ---
    ("FileNotFoundError", "No such file or directory: 'input.png'", ErrorClass.PERMANENT),
    ("RuntimeError", "checkpoint not found: sdxl.safetensors", ErrorClass.PERMANENT),
    ("RuntimeError", "missing model: some_model.ckpt", ErrorClass.PERMANENT),
    ("ValueError", "invalid value for enum sampler_name", ErrorClass.PERMANENT),
    ("ValueError", "'foo' not in list of valid choices", ErrorClass.PERMANENT),
    ("prompt_outputs_failed_validation", "Required input is missing: image", ErrorClass.PERMANENT),
    # --- UNKNOWN (unmatched -> must not crash, must fall back) ---
    ("SomeWeirdInternalError", "the gremlins did it", ErrorClass.UNKNOWN),
    ("", "", ErrorClass.UNKNOWN),
]


@pytest.mark.parametrize("exception_type,exception_message,expected", CLASSIFIER_CASES)
def test_classifier_table(exception_type, exception_message, expected):
    assert DEFAULT_CLASSIFIER.classify(_err(exception_type, exception_message)) is expected


def test_classifier_oom_is_temporary_and_never_a_restart_signal():
    """The load-bearing case: ComfyUI catches CUDA OOM per-node. The prompt
    fails but the worker thread survives, so OOM must classify as an ordinary
    retryable TEMPORARY failure — never as something that would make the
    controller think the worker died and needs a restart."""
    error = _err("torch.cuda.OutOfMemoryError", "CUDA out of memory")
    assert DEFAULT_CLASSIFIER.classify(error) is ErrorClass.TEMPORARY

    # And the resulting decision is "retry the job", not "terminal/restart".
    decision = RetryPolicy(rng=random.Random(0)).next_action(
        ErrorClass.TEMPORARY, attempt=0, kind=AssetKind.SINGLE
    )
    assert decision.should_retry is True
    assert decision.terminal_state is None


def test_classifier_never_crashes_on_garbage_input():
    error = ExecutionError(prompt_id="p", exception_type="\x00weird", exception_message="")
    assert DEFAULT_CLASSIFIER.classify(error) is ErrorClass.UNKNOWN


def test_classifier_is_easily_extended():
    custom = ErrorClassifier(
        extra_rules=[_Rule(ErrorClass.PERMANENT, message_substrings=("banana",))]
    )
    assert custom.classify(_err("RuntimeError", "banana peel on the GPU")) is ErrorClass.PERMANENT
    # Extension doesn't remove the defaults.
    assert custom.classify(_err("torch.cuda.OutOfMemoryError", "oom")) is ErrorClass.TEMPORARY


# --------------------------------------------------------------------------- RetryPolicy


def test_temporary_backoff_ladder_single():
    policy = RetryPolicy(rng=random.Random(0))
    delays = []
    for attempt in range(3):
        d = policy.next_action(ErrorClass.TEMPORARY, attempt, AssetKind.SINGLE)
        assert d.should_retry
        assert d.terminal_state is None
        delays.append(d.delay_s)
    # jittered around 10/30/60
    assert 8 <= delays[0] <= 12
    assert 24 <= delays[1] <= 36
    assert 48 <= delays[2] <= 72

    exhausted = policy.next_action(ErrorClass.TEMPORARY, attempt=3, kind=AssetKind.SINGLE)
    assert not exhausted.should_retry
    # It needed retries to fail this many times -- we're less sure it's truly
    # unfixable than an immediate PERMANENT classification, so route to human
    # review rather than a flat PERMANENT_ERROR.
    assert exhausted.terminal_state is AssetState.SKIPPED_REVIEW


def test_temporary_exhausted_on_sequence_parks_not_skips():
    policy = RetryPolicy(rng=random.Random(0))
    d = policy.next_action(ErrorClass.TEMPORARY, attempt=3, kind=AssetKind.SEQUENCE)
    assert not d.should_retry
    assert d.terminal_state is AssetState.PARKED_APPROVAL


def test_permanent_never_retries():
    policy = RetryPolicy(rng=random.Random(0))
    for attempt in range(5):
        d = policy.next_action(ErrorClass.PERMANENT, attempt, AssetKind.SINGLE)
        assert not d.should_retry
        assert d.delay_s == 0.0
        assert d.terminal_state is AssetState.PERMANENT_ERROR


def test_permanent_on_sequence_parks():
    policy = RetryPolicy(rng=random.Random(0))
    d = policy.next_action(ErrorClass.PERMANENT, attempt=0, kind=AssetKind.SEQUENCE)
    assert d.terminal_state is AssetState.PARKED_APPROVAL


def test_unknown_gets_exactly_one_retry_then_skips():
    policy = RetryPolicy(rng=random.Random(0))
    first = policy.next_action(ErrorClass.UNKNOWN, attempt=0, kind=AssetKind.SINGLE)
    assert first.should_retry
    assert first.terminal_state is None

    second = policy.next_action(ErrorClass.UNKNOWN, attempt=1, kind=AssetKind.SINGLE)
    assert not second.should_retry
    # Unknown means "we're not sure it's unfixable" -> route to human review,
    # not a flat PERMANENT_ERROR.
    assert second.terminal_state is AssetState.SKIPPED_REVIEW


def test_unknown_on_sequence_parks():
    policy = RetryPolicy(rng=random.Random(0))
    d = policy.next_action(ErrorClass.UNKNOWN, attempt=1, kind=AssetKind.SEQUENCE)
    assert d.terminal_state is AssetState.PARKED_APPROVAL


def test_retry_policy_is_pure_same_inputs_same_shape():
    """Not byte-for-byte identical (jitter), but always the same should_retry
    and terminal_state for the same inputs."""
    policy = RetryPolicy(rng=random.Random(1))
    for _ in range(20):
        d = policy.next_action(ErrorClass.PERMANENT, attempt=0, kind=AssetKind.SINGLE)
        assert d.should_retry is False
        assert d.terminal_state is AssetState.PERMANENT_ERROR


def test_jitter_actually_varies_delay():
    policy = RetryPolicy(rng=random.Random(42))
    delays = {
        policy.next_action(ErrorClass.TEMPORARY, 0, AssetKind.SINGLE).delay_s for _ in range(10)
    }
    assert len(delays) > 1  # not constant


@pytest.mark.parametrize(
    "kind,max_retries", [(AssetKind.SINGLE, 3), (AssetKind.SEQUENCE, 2)]
)
def test_qc_retry_counts_match_design_doc(kind, max_retries):
    assert QC_MAX_RETRIES[kind] == max_retries
    policy = RetryPolicy()
    for attempt in range(max_retries):
        d = policy.qc_retry_decision(attempt, kind)
        assert d.should_retry
        assert d.terminal_state is None
    exhausted = policy.qc_retry_decision(max_retries, kind)
    assert not exhausted.should_retry
    expected_terminal = (
        AssetState.PARKED_APPROVAL if kind is AssetKind.SEQUENCE else AssetState.SKIPPED_REVIEW
    )
    assert exhausted.terminal_state is expected_terminal


# --------------------------------------------------------------------------- CircuitBreaker


class _Clock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_breaker_closed_by_default():
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=5)
    assert b.state("wf-a") is BreakerState.CLOSED
    assert b.allow("wf-a")


def test_breaker_opens_after_n_consecutive_same_class_failures():
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=5, cooldown_s=300)
    for _ in range(4):
        b.record_failure("wf-a", ErrorClass.TEMPORARY)
        assert b.state("wf-a") is BreakerState.CLOSED
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    assert b.state("wf-a") is BreakerState.OPEN
    assert not b.allow("wf-a")


def test_breaker_different_error_classes_do_not_accumulate_together():
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=3)
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    b.record_failure("wf-a", ErrorClass.PERMANENT)  # resets the streak
    b.record_failure("wf-a", ErrorClass.PERMANENT)
    assert b.state("wf-a") is BreakerState.CLOSED  # only 2 consecutive PERMANENT


def test_breaker_success_resets_consecutive_failures():
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=3)
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    b.record_success("wf-a")
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    assert b.state("wf-a") is BreakerState.CLOSED  # streak was reset in between


def test_breaker_half_open_after_cooldown_allows_one_canary():
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=1, cooldown_s=300)
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    assert b.state("wf-a") is BreakerState.OPEN
    assert not b.allow("wf-a")

    clock.advance(299)
    assert not b.allow("wf-a")  # cooldown not elapsed yet

    clock.advance(2)
    assert b.allow("wf-a")  # canary granted
    assert b.state("wf-a") is BreakerState.HALF_OPEN
    assert not b.allow("wf-a")  # a second canary is refused while one is in flight


def test_breaker_half_open_success_closes():
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=1, cooldown_s=300)
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    clock.advance(300)
    assert b.allow("wf-a")  # canary
    b.record_success("wf-a")
    assert b.state("wf-a") is BreakerState.CLOSED
    assert b.allow("wf-a")


def test_breaker_half_open_failure_reopens_with_longer_cooldown():
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=1, cooldown_s=300)
    b.record_failure("wf-a", ErrorClass.TEMPORARY)
    clock.advance(300)
    assert b.allow("wf-a")  # canary granted
    b.record_failure("wf-a", ErrorClass.TEMPORARY)  # canary failed
    assert b.state("wf-a") is BreakerState.OPEN

    clock.advance(300)  # old cooldown would have elapsed, new (600s) has not
    assert not b.allow("wf-a")
    clock.advance(300)
    assert b.allow("wf-a")


def test_breaker_partition_isolation():
    """One bad workflow (partition A) opening its breaker must not affect
    partition B at all."""
    clock = _Clock()
    b = CircuitBreaker(now=clock, failure_threshold=3)
    for _ in range(3):
        b.record_failure("workflow-A", ErrorClass.PERMANENT)
    assert b.state("workflow-A") is BreakerState.OPEN
    assert not b.allow("workflow-A")

    assert b.state("workflow-B") is BreakerState.CLOSED
    assert b.allow("workflow-B")
    b.record_failure("workflow-B", ErrorClass.PERMANENT)
    assert b.state("workflow-B") is BreakerState.CLOSED  # only 1 failure so far


# --------------------------------------------------------------------------- GlobalBreaker


def test_global_breaker_trips_on_repeated_crashes_regardless_of_job():
    clock = _Clock()
    g = GlobalBreaker(now=clock, crash_threshold=3, crash_window_s=600)
    g.record_crash()
    g.record_crash()
    assert not g.is_tripped()
    g.record_crash()
    assert g.is_tripped()
    assert "repeated_crashes" in g.reason


def test_global_breaker_crash_window_expires_old_crashes():
    clock = _Clock()
    g = GlobalBreaker(now=clock, crash_threshold=3, crash_window_s=60)
    g.record_crash()
    clock.advance(100)  # outside the window, ages out
    g.record_crash()
    g.record_crash()
    assert not g.is_tripped()  # only 2 within the window


def test_global_breaker_one_shot_conditions():
    clock = _Clock()
    g = GlobalBreaker(now=clock)
    assert not g.is_tripped()
    g.trip("disk_full")
    assert g.is_tripped()
    assert g.reason == "disk_full"


def test_global_breaker_stays_tripped_until_explicit_reset():
    clock = _Clock()
    g = GlobalBreaker(now=clock)
    g.trip("budget_ceiling")
    clock.advance(10_000)
    assert g.is_tripped()  # no automatic recovery
    g.reset()
    assert not g.is_tripped()
    assert g.reason is None
