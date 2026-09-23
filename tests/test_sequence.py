import pytest
from hypothesis import given, settings, strategies as st

from comfy_controller.core.sequence import (
    QCCompare,
    RegenerateFrame,
    SequenceRepairPlanner,
)


def test_no_failed_frames_is_immediately_fixed():
    planner = SequenceRepairPlanner()
    plan = planner.plan(frame_count=10, failed_frames=set(), attempt_counts={})
    assert plan.outcome == "fixed"
    assert plan.steps == ()


def test_single_failed_frame_regenerates_with_predecessor_reference():
    planner = SequenceRepairPlanner()
    plan = planner.plan(frame_count=10, failed_frames={5}, attempt_counts={})
    assert plan.outcome == "in_progress"
    assert plan.steps == (
        RegenerateFrame(frame_index=5, reference_frame_index=4),
        QCCompare(frame_index=5, against=(4, 6)),
    )


def test_first_frame_has_no_predecessor_reference():
    planner = SequenceRepairPlanner()
    plan = planner.plan(frame_count=5, failed_frames={0}, attempt_counts={})
    assert plan.steps[0] == RegenerateFrame(frame_index=0, reference_frame_index=None)
    # only a successor to compare against
    assert plan.steps[1] == QCCompare(frame_index=0, against=(1,))


def test_last_frame_has_no_successor_to_compare_against():
    planner = SequenceRepairPlanner()
    plan = planner.plan(frame_count=5, failed_frames={4}, attempt_counts={})
    assert plan.steps[0] == RegenerateFrame(frame_index=4, reference_frame_index=3)
    assert plan.steps[1] == QCCompare(frame_index=4, against=(3,))


def test_cascading_contiguous_failures_reference_each_other_in_order():
    """Step 5: a run of bad frames regenerates from N onward, each referencing
    the previous. Frame 6 is also in `failed_frames`, so 5's compare against 6
    is deferred (6 isn't stable yet). 6 always checks against its predecessor
    5 (the reference it was just generated from, fresh or not) and, since 7 is
    stable, against 7 too."""
    planner = SequenceRepairPlanner()
    plan = planner.plan(frame_count=10, failed_frames={5, 6}, attempt_counts={})
    assert plan.steps == (
        RegenerateFrame(frame_index=5, reference_frame_index=4),
        QCCompare(frame_index=5, against=(4,)),
        RegenerateFrame(frame_index=6, reference_frame_index=5),
        QCCompare(frame_index=6, against=(5, 7)),
    )


def test_exhausted_retries_parks_whole_sequence():
    planner = SequenceRepairPlanner(max_retries=2)
    plan = planner.plan(frame_count=10, failed_frames={5}, attempt_counts={5: 2})
    assert plan.outcome == "parked"
    assert plan.steps == ()


def test_any_frame_exhausted_parks_even_if_others_have_budget():
    planner = SequenceRepairPlanner(max_retries=2)
    plan = planner.plan(frame_count=10, failed_frames={5, 6}, attempt_counts={5: 0, 6: 2})
    assert plan.outcome == "parked"


def test_out_of_range_index_rejected():
    planner = SequenceRepairPlanner()
    with pytest.raises(ValueError):
        planner.plan(frame_count=5, failed_frames={10}, attempt_counts={})


# --------------------------------------------------------------------------- crash mid-sequence


def test_crash_reruns_the_whole_sequence_from_scratch():
    planner = SequenceRepairPlanner()
    plan = planner.plan_after_crash(frame_count=4)
    assert plan.outcome == "in_progress"
    assert plan.steps == (
        RegenerateFrame(frame_index=0, reference_frame_index=None),
        RegenerateFrame(frame_index=1, reference_frame_index=0),
        RegenerateFrame(frame_index=2, reference_frame_index=1),
        RegenerateFrame(frame_index=3, reference_frame_index=2),
    )


def test_crash_on_empty_sequence_is_a_no_op():
    planner = SequenceRepairPlanner()
    plan = planner.plan_after_crash(frame_count=0)
    assert plan.outcome == "fixed"
    assert plan.steps == ()


# --------------------------------------------------------------------------- property tests
#
# These simulate the *iterative* repair loop the controller actually runs:
# call plan(), "execute" the regenerate steps (incrementing attempt counts),
# resolve the QCCompare steps against a hypothesis-generated pass/fail oracle,
# derive the next round's failed_frames per step 4/5, and feed that back into
# plan() again — round after round — until a terminal outcome is reached.


def _run_to_terminal(frame_count, initially_failed, max_retries, qc_outcomes, max_rounds=50):
    """qc_outcomes: callable(frame_index, attempt_number) -> bool (True = passes
    against a given neighbor). Returns (final_outcome, regen_counts, rounds_used).
    """
    planner = SequenceRepairPlanner(max_retries=max_retries)
    attempt_counts: dict[int, int] = {}
    regen_counts: dict[int, int] = {}
    failed = set(initially_failed)

    for round_no in range(max_rounds):
        plan = planner.plan(frame_count, failed, attempt_counts)
        if plan.outcome != "in_progress":
            return plan.outcome, regen_counts, round_no

        regenerated_this_round: set[int] = set()
        for step in plan.steps:
            if isinstance(step, RegenerateFrame):
                regenerated_this_round.add(step.frame_index)
                attempt_counts[step.frame_index] = attempt_counts.get(step.frame_index, 0) + 1
                regen_counts[step.frame_index] = regen_counts.get(step.frame_index, 0) + 1

        next_failed: set[int] = set()
        for step in plan.steps:
            if isinstance(step, QCCompare):
                ok = all(
                    qc_outcomes(step.frame_index, neighbor, attempt_counts[step.frame_index])
                    for neighbor in step.against
                )
                if not ok:
                    next_failed.add(step.frame_index)
                    # step 5: mismatch cascades forward to the next frame too.
                    if step.frame_index + 1 < frame_count:
                        next_failed.add(step.frame_index + 1)

        failed = next_failed
        if not failed:
            return "fixed", regen_counts, round_no + 1

    raise AssertionError("did not terminate within max_rounds")


@given(
    frame_count=st.integers(min_value=1, max_value=12),
    initially_failed=st.data(),
    max_retries=st.integers(min_value=1, max_value=4),
    always_fails=st.booleans(),
)
@settings(max_examples=200)
def test_plan_always_terminates_and_respects_retry_cap(
    frame_count, initially_failed, max_retries, always_fails
):
    failed = initially_failed.draw(
        st.sets(st.integers(min_value=0, max_value=frame_count - 1), min_size=1, max_size=frame_count)
    )

    # Oracle: either every QC check fails forever (forces the worst case: max
    # retries burned, then park -- unless there's genuinely nothing to compare
    # against, e.g. a lone frame_count=1 "sequence", which trivially converges)
    # or every QC check passes on the first try (forces immediate convergence).
    def oracle(frame_index, neighbor, attempt_number):
        return not always_fails

    outcome, regen_counts, rounds = _run_to_terminal(
        frame_count, failed, max_retries, oracle
    )

    # The plan always terminates (no exception from _run_to_terminal's
    # max_rounds guard) and ends in one of exactly the two allowed outcomes.
    assert outcome in ("fixed", "parked")
    # No frame is ever regenerated more than max_retries times.
    for count in regen_counts.values():
        assert count <= max_retries
    if outcome == "parked":
        # Parking only happens because some frame actually hit the cap.
        assert any(c >= max_retries for c in regen_counts.values())
    if not always_fails:
        # Every check the oracle is asked passes, so nothing should ever be
        # judged still-failing.
        assert outcome == "fixed"


@given(
    frame_count=st.integers(min_value=2, max_value=10),
    max_retries=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=100)
def test_plan_terminates_with_mixed_random_qc_outcomes(frame_count, max_retries, seed):
    import random as _random

    rng = _random.Random(seed)
    failed = {rng.randrange(frame_count)}

    def oracle(frame_index, neighbor, attempt_number):
        return rng.random() < 0.5

    outcome, regen_counts, rounds = _run_to_terminal(frame_count, failed, max_retries, oracle)

    assert outcome in ("fixed", "parked")
    for count in regen_counts.values():
        assert count <= max_retries
