"""Pure decision logic: classify errors, decide retries, and trip circuit breakers.

No I/O, no async, no network, no `time.sleep`. Every time-dependent piece takes an
injected `now()` callable so tests are deterministic. The controller (elsewhere)
is responsible for actually sleeping and for persisting the resulting state in
SQLite — nothing here holds state that would be lost to `kill -9`, except the
CircuitBreaker/GlobalBreaker objects themselves, which the controller must
reconstruct from persisted counters at startup if it wants breaker state to
survive a restart.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from ..models import AssetKind, AssetState, ErrorClass, ExecutionError

# --------------------------------------------------------------------------- classifier


@dataclass(frozen=True)
class _Rule:
    """One row of the classification table.

    A rule matches when the exception *type* contains any of `type_substrings`
    (case-sensitive — exception type names are, e.g. `torch.cuda.OutOfMemoryError`)
    or the exception *message* contains any of `message_substrings`
    (case-insensitive — messages are free text).
    """

    error_class: ErrorClass
    type_substrings: tuple[str, ...] = ()
    message_substrings: tuple[str, ...] = ()

    def matches(self, exception_type: str, exception_message: str) -> bool:
        t = exception_type or ""
        m = (exception_message or "").lower()
        if self.type_substrings and any(s in t for s in self.type_substrings):
            return True
        if self.message_substrings and any(s in m for s in self.message_substrings):
            return True
        return False


# Order matters: first match wins. Keep TEMPORARY/PERMANENT signals specific
# enough that they don't accidentally shadow each other.
_DEFAULT_RULES: tuple[_Rule, ...] = (
    # --- TEMPORARY --------------------------------------------------------
    # CUDA OOM is caught per-node by ComfyUI itself: the prompt fails and the
    # models involved get unloaded, but the worker *thread stays alive* and
    # keeps pulling jobs off the queue. This is NOT a worker crash and must
    # NEVER be treated as a signal to restart ComfyUI — it is an ordinary,
    # retryable job failure like any other TEMPORARY error.
    _Rule(
        ErrorClass.TEMPORARY,
        type_substrings=("OutOfMemoryError",),
        message_substrings=("out of memory", "cuda out of memory"),
    ),
    _Rule(
        ErrorClass.TEMPORARY,
        type_substrings=("TimeoutError", "asyncio.TimeoutError"),
        message_substrings=("timed out", "timeout"),
    ),
    _Rule(
        ErrorClass.TEMPORARY,
        type_substrings=("ConnectionError", "ClientConnectorError", "ClientOSError"),
        message_substrings=("connection reset", "connection refused", "connection error"),
    ),
    _Rule(
        ErrorClass.TEMPORARY,
        message_substrings=("server busy", "queue is full", "try again later"),
    ),
    # --- PERMANENT ----------------------------------------------------------
    # Missing files / models / checkpoints: retrying changes nothing.
    _Rule(
        ErrorClass.PERMANENT,
        type_substrings=("FileNotFoundError",),
        message_substrings=("no such file", "not found"),
    ),
    _Rule(
        ErrorClass.PERMANENT,
        message_substrings=(
            "checkpoint",
            "model not found",
            "missing model",
            "unable to find",
        ),
    ),
    # Node validation / bad input: ComfyUI's prompt-validation errors, invalid
    # enum choices ("value not in list"), bad literal input values.
    _Rule(
        ErrorClass.PERMANENT,
        type_substrings=("ValueError", "TypeError", "KeyError"),
        message_substrings=(
            "invalid",
            "not in list",
            "value not in",
            "required input is missing",
            "validation",
        ),
    ),
)


class ErrorClassifier:
    """Maps an `ExecutionError` to an `ErrorClass`.

    Table-driven and easily extended: pass additional `_Rule`s (or just append
    to a copy of `_DEFAULT_RULES`) rather than growing an if/elif chain.
    Anything that matches no rule falls back to UNKNOWN — it never raises.
    """

    def __init__(self, extra_rules: Sequence[_Rule] = ()) -> None:
        self._rules: tuple[_Rule, ...] = (*_DEFAULT_RULES, *extra_rules)

    def classify(self, error: ExecutionError) -> ErrorClass:
        for rule in self._rules:
            if rule.matches(error.exception_type, error.exception_message):
                return rule.error_class
        return ErrorClass.UNKNOWN


DEFAULT_CLASSIFIER = ErrorClassifier()

# --------------------------------------------------------------------------- retry policy

# QC repair retries (design doc): a single asset gets more chances than a whole
# sequence, because re-running a sequence is far more expensive.
QC_MAX_RETRIES: dict[AssetKind, int] = {
    AssetKind.SINGLE: 3,
    AssetKind.SEQUENCE: 2,
}

# TEMPORARY backoff ladder: 10s, 30s, 60s, then give up (max 3 retry attempts).
_TEMPORARY_BACKOFF_S: tuple[float, ...] = (10.0, 30.0, 60.0)

# UNKNOWN gets exactly one retry, using the gentlest backoff step — we don't
# know what it is, so we don't hammer the server, but we also don't want to
# burn three attempts on something that might be PERMANENT in disguise.
_UNKNOWN_BACKOFF_S: float = 10.0

_JITTER_FRACTION = 0.2  # +/-20% jitter so many concurrently-failing jobs don't retry in lockstep


@dataclass(frozen=True)
class RetryDecision:
    should_retry: bool
    delay_s: float
    terminal_state: AssetState | None = None  # set only when should_retry is False


def _jittered(base_delay_s: float, rng: random.Random) -> float:
    spread = base_delay_s * _JITTER_FRACTION
    return max(0.0, base_delay_s + rng.uniform(-spread, spread))


def _terminal_state(kind: AssetKind, *, permanent: bool) -> AssetState:
    """Sequences can't be silently 'skipped' one frame at a time — the asset
    row *is* the whole sequence, and frames depend on each other. So once a
    sequence exhausts its retries, the whole thing goes to PARKED_APPROVAL for
    human review rather than being quietly dropped. A single asset can be
    skipped: PERMANENT_ERROR when we're sure it's unfixable, SKIPPED_REVIEW
    when we're not (UNKNOWN) and a human should sanity-check the skip.
    """
    if kind is AssetKind.SEQUENCE:
        return AssetState.PARKED_APPROVAL
    return AssetState.PERMANENT_ERROR if permanent else AssetState.SKIPPED_REVIEW


class RetryPolicy:
    """Pure function of (error_class, attempt, kind) -> RetryDecision.

    Holds no state itself — the controller persists `attempt` in SQLite so a
    `kill -9` mid-batch loses nothing; replaying the same `attempt` through
    this policy after a restart yields the same decision.

    `attempt` is the number of attempts already made (0 == this is the first
    failure, no retry consumed yet).
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()

    def next_action(
        self, error_class: ErrorClass, attempt: int, kind: AssetKind
    ) -> RetryDecision:
        if error_class is ErrorClass.PERMANENT:
            return RetryDecision(False, 0.0, _terminal_state(kind, permanent=True))

        if error_class is ErrorClass.TEMPORARY:
            if attempt < len(_TEMPORARY_BACKOFF_S):
                delay = _jittered(_TEMPORARY_BACKOFF_S[attempt], self._rng)
                return RetryDecision(True, delay, None)
            return RetryDecision(False, 0.0, _terminal_state(kind, permanent=False))

        # ErrorClass.UNKNOWN
        if attempt < 1:
            return RetryDecision(True, _jittered(_UNKNOWN_BACKOFF_S, self._rng), None)
        return RetryDecision(False, 0.0, _terminal_state(kind, permanent=False))

    def qc_retry_decision(self, attempt: int, kind: AssetKind) -> RetryDecision:
        """A QC (rubric) failure, as opposed to an execution error. Regeneration
        is cheap compared to a system-level retry, so no backoff delay — just a
        cap on how many times we'll re-roll before giving up.
        """
        max_retries = QC_MAX_RETRIES[kind]
        if attempt < max_retries:
            return RetryDecision(True, 0.0, None)
        terminal = AssetState.PARKED_APPROVAL if kind is AssetKind.SEQUENCE else AssetState.SKIPPED_REVIEW
        return RetryDecision(False, 0.0, terminal)


# --------------------------------------------------------------------------- circuit breaker


class BreakerState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _PartitionState:
    state: BreakerState = BreakerState.CLOSED
    consecutive_failures: int = 0
    last_failure_class: ErrorClass | None = None
    opened_at: float | None = None
    cooldown_s: float = 0.0
    canary_in_flight: bool = False


class CircuitBreaker:
    """Partitioned by job type (e.g. workflow name) so one bad workflow can't
    halt jobs of every other kind. Opens after `failure_threshold` *consecutive
    same-class* failures within a partition; a canary (`allow()` returning True
    on a HALF_OPEN partition) decides whether it re-closes or re-opens with a
    longer cooldown.

    Clock is injected (`now`) so tests never sleep.
    """

    def __init__(
        self,
        now: Callable[[], float],
        failure_threshold: int = 5,
        cooldown_s: float = 300.0,
    ) -> None:
        self._now = now
        self._threshold = failure_threshold
        self._base_cooldown_s = cooldown_s
        self._partitions: dict[str, _PartitionState] = {}

    def _get(self, partition: str) -> _PartitionState:
        return self._partitions.setdefault(partition, _PartitionState())

    def state(self, partition: str) -> BreakerState:
        return self._get(partition).state

    def allow(self, partition: str) -> bool:
        """May the caller submit a job in this partition right now?

        Calling this on an OPEN partition whose cooldown has elapsed flips it
        to HALF_OPEN and grants exactly one canary; further calls while that
        canary is outstanding are refused until `record_success`/`record_failure`
        resolves it.
        """
        p = self._get(partition)
        if p.state is BreakerState.CLOSED:
            return True
        if p.state is BreakerState.OPEN:
            assert p.opened_at is not None
            if self._now() - p.opened_at >= p.cooldown_s:
                p.state = BreakerState.HALF_OPEN
                p.canary_in_flight = True
                return True
            return False
        # HALF_OPEN: only ever one canary in flight at a time.
        if p.canary_in_flight:
            return False
        p.canary_in_flight = True
        return True

    def record_success(self, partition: str) -> None:
        p = self._get(partition)
        if p.state is BreakerState.HALF_OPEN:
            # Canary succeeded: fully close and forget the failure streak.
            p.state = BreakerState.CLOSED
            p.cooldown_s = 0.0
        p.consecutive_failures = 0
        p.last_failure_class = None
        p.canary_in_flight = False

    def record_failure(self, partition: str, error_class: ErrorClass) -> None:
        p = self._get(partition)
        if p.state is BreakerState.HALF_OPEN:
            # Canary failed: re-open with a longer cooldown so we don't
            # hammer a still-broken partition every 300s forever.
            p.state = BreakerState.OPEN
            p.opened_at = self._now()
            p.cooldown_s = (p.cooldown_s or self._base_cooldown_s) * 2
            p.canary_in_flight = False
            return

        if p.last_failure_class == error_class:
            p.consecutive_failures += 1
        else:
            p.consecutive_failures = 1
            p.last_failure_class = error_class

        if p.consecutive_failures >= self._threshold:
            p.state = BreakerState.OPEN
            p.opened_at = self._now()
            p.cooldown_s = self._base_cooldown_s


class GlobalBreaker:
    """Trips on conditions no per-partition retry or isolation can fix:
    repeated ComfyUI crashes regardless of which job was running, disk full,
    or a budget ceiling. Independent of `CircuitBreaker` partitioning — this
    is a single latch for the whole batch.

    Once tripped it stays tripped: these conditions need an operator, not a
    timer, so there is no automatic half-open/cooldown recovery. Call
    `reset()` explicitly once the operator has intervened.
    """

    def __init__(
        self,
        now: Callable[[], float],
        crash_threshold: int = 3,
        crash_window_s: float = 600.0,
    ) -> None:
        self._now = now
        self._crash_threshold = crash_threshold
        self._crash_window_s = crash_window_s
        self._crash_times: list[float] = []
        self._tripped = False
        self._reason: str | None = None

    def record_crash(self) -> None:
        """Call once per ComfyUI worker crash/restart, regardless of which job
        was running when it happened. Crashes outside the rolling window age
        out, so a slow trickle of unrelated one-off crashes doesn't trip this."""
        if self._tripped:
            return
        now = self._now()
        self._crash_times = [t for t in self._crash_times if now - t <= self._crash_window_s]
        self._crash_times.append(now)
        if len(self._crash_times) >= self._crash_threshold:
            self._trip(f"repeated_crashes:{len(self._crash_times)}_in_{self._crash_window_s}s")

    def trip(self, reason: str) -> None:
        """For one-shot conditions the caller detects itself (disk full, budget ceiling)."""
        self._trip(reason)

    def _trip(self, reason: str) -> None:
        self._tripped = True
        self._reason = reason

    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> str | None:
        return self._reason

    def reset(self) -> None:
        self._tripped = False
        self._reason = None
        self._crash_times.clear()
