"""Pure planner for the sequence frame-repair procedure (design doc):

    1. Frame N fails QC.
    2. Regenerate N using N-1 as reference.
    3. QC the new N against both N-1 and N+1.
    4. Matches both -> fixed.
    5. Mismatch with N+1 -> regenerate from N onward, each frame referencing
       the previous.
    6. Still failing after max retries (2 for sequences) -> park the WHOLE
       sequence for review.

No I/O: the planner only ever *proposes* steps. It never learns the outcome of
a QC check itself — the caller executes the steps, observes real QC verdicts,
updates `attempt_counts`/`failed_frames`, and asks the planner again for the
next round. This keeps the planner a pure function of its inputs and testable
without a fake ComfyUI at all.

A crash mid-sequence is handled separately (`plan_after_crash`): frames in a
sequence depend on their predecessor, so a crash can't be resumed frame-by-frame
— the whole sequence reruns from frame 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AbstractSet, Literal, Mapping

from .policy import QC_MAX_RETRIES
from ..models import AssetKind

DEFAULT_MAX_RETRIES = QC_MAX_RETRIES[AssetKind.SEQUENCE]  # 2, single source of truth

Outcome = Literal["in_progress", "fixed", "parked"]


@dataclass(frozen=True)
class RegenerateFrame:
    """Regenerate `frame_index`, referencing `reference_frame_index` (or no
    reference at all when the failing frame is frame 0)."""

    frame_index: int
    reference_frame_index: int | None


@dataclass(frozen=True)
class QCCompare:
    """QC `frame_index` against each index in `against` (its stable/just-fixed
    neighbors). Order within `against` is not significant."""

    frame_index: int
    against: tuple[int, ...]


RepairStep = RegenerateFrame | QCCompare


@dataclass(frozen=True)
class RepairPlan:
    steps: tuple[RepairStep, ...]
    outcome: Outcome


class SequenceRepairPlanner:
    """Stateless: every call is a fresh function of its arguments. `max_retries`
    defaults to the design doc's sequence value (2); pass a different value only
    to match a different `AssetKind` policy (single assets aren't sequences and
    don't go through this planner at all).
    """

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES) -> None:
        self._max_retries = max_retries

    def plan(
        self,
        frame_count: int,
        failed_frames: AbstractSet[int],
        attempt_counts: Mapping[int, int],
    ) -> RepairPlan:
        """Plan the next round of repair steps.

        `failed_frames` is the set of frame indices that still need fixing
        going into this round (initially: whatever failed QC; on later rounds,
        expanded by the caller per step 5 when a regenerated frame mismatches
        its successor). `attempt_counts` is how many times each frame has
        already been regenerated (by this planner's prior rounds).
        """
        if not failed_frames:
            return RepairPlan((), "fixed")

        if any(attempt_counts.get(i, 0) >= self._max_retries for i in failed_frames):
            # Step 6: at least one frame has exhausted its retries. Frames in a
            # sequence are not independently skippable, so the whole sequence
            # is parked rather than dropping just that frame.
            return RepairPlan((), "parked")

        if any(i < 0 or i >= frame_count for i in failed_frames):
            raise ValueError("failed_frames contains an index outside [0, frame_count)")

        steps: list[RepairStep] = []
        ordered = sorted(failed_frames)
        for i in ordered:
            reference = i - 1 if i > 0 else None
            steps.append(RegenerateFrame(frame_index=i, reference_frame_index=reference))

            neighbors: list[int] = []
            if i > 0:
                neighbors.append(i - 1)
            # Only compare against i+1 if it is stable right now (not itself
            # being regenerated this same round) — otherwise that comparison
            # is meaningless until i+1 has been (re)generated too, which the
            # caller will schedule in a later round if it turns out to still
            # be needed (step 5's cascade).
            if i + 1 < frame_count and (i + 1) not in failed_frames:
                neighbors.append(i + 1)
            steps.append(QCCompare(frame_index=i, against=tuple(neighbors)))

        return RepairPlan(tuple(steps), "in_progress")

    def plan_after_crash(self, frame_count: int) -> RepairPlan:
        """A crash mid-sequence: frames depend on their predecessor, so partial
        progress can't be trusted. Regenerate everything from frame 0, each
        frame referencing the previous newly-generated one. No QC steps here —
        each frame goes through the normal QC pipeline once regenerated, and
        any post-hoc QC failures re-enter `plan()` as usual.
        """
        if frame_count <= 0:
            return RepairPlan((), "fixed")
        steps: tuple[RepairStep, ...] = tuple(
            RegenerateFrame(frame_index=i, reference_frame_index=(i - 1 if i > 0 else None))
            for i in range(frame_count)
        )
        return RepairPlan(steps, "in_progress")
