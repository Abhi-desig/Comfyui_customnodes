"""Pure report-formatting logic: AssetRecords -> report structures.

Deliberately has no Slack/HTTP/console dependency at all, so it can be unit
tested on its own and reused by the CLI's `report` command as well as by any
NotifierPort implementation. A notifier's job is only to *render* what this
module computes (as Slack blocks, plain text, ...) -- never to recompute it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .models import AssetRecord, AssetState

# --------------------------------------------------------------------- report


@dataclass(frozen=True)
class FailureRateRow:
    """One row of the failure-rate table: attempts vs. successes for a group
    (by workflow -- the axis a reviewer actually wants to act on)."""

    label: str
    attempts: int
    successes: int

    @property
    def failures(self) -> int:
        return self.attempts - self.successes

    @property
    def failure_rate(self) -> float:
        """0.0 on an empty group rather than raising -- callers shouldn't have
        to guard a ZeroDivisionError just to render a table."""
        return 0.0 if self.attempts == 0 else self.failures / self.attempts


@dataclass
class MorningReport:
    """Grouped-by-terminal-state content of the morning report."""

    completed_count: int = 0
    output_folder: str | None = None

    # Detail lists -- only for states a human needs to look at individually.
    # SAVED is deliberately not one of these: a clean run doesn't need every
    # success enumerated, just the count above.
    skipped_for_review: list[AssetRecord] = field(default_factory=list)
    awaiting_approval: list[AssetRecord] = field(default_factory=list)
    judge_unavailable: list[AssetRecord] = field(default_factory=list)
    permanent_errors: list[AssetRecord] = field(default_factory=list)

    still_in_progress: int = 0
    failure_rate: list[FailureRateRow] = field(default_factory=list)
    todo: list[str] = field(default_factory=list)


# JUDGE_UNAVAILABLE is intentionally its own bucket, never merged into
# skipped_for_review: it means the judge infrastructure broke, not that the
# asset failed the rubric. Conflating the two is exactly the mistake a morning
# reviewer must not make (see design doc).
_STATE_TO_GROUP = {
    AssetState.SKIPPED_REVIEW: "skipped_for_review",
    AssetState.PARKED_APPROVAL: "awaiting_approval",
    AssetState.JUDGE_UNAVAILABLE: "judge_unavailable",
    AssetState.PERMANENT_ERROR: "permanent_errors",
}


def build_morning_report(
    summary: dict[AssetState, int] | None,
    records: list[AssetRecord],
    *,
    output_folder: str | None = None,
) -> MorningReport:
    """Turn (summary counts, detail records) into a `MorningReport`.

    `summary` is treated as the store's authoritative counts (it may cover
    assets the caller didn't bother attaching detail records for, e.g. plain
    successes); we fall back to counting `records` so this still works if the
    caller only ever passes records and no summary.
    """
    report = MorningReport(output_folder=output_folder)
    for r in records:
        group = _STATE_TO_GROUP.get(r.state)
        if group is not None:
            getattr(report, group).append(r)
        elif r.state is not AssetState.SAVED and not r.state.is_terminal:
            report.still_in_progress += 1

    if summary:
        report.completed_count = summary.get(AssetState.SAVED, 0)
    else:
        report.completed_count = sum(1 for r in records if r.state == AssetState.SAVED)

    report.failure_rate = build_failure_rate_table(records)
    report.todo = build_todo(report)
    return report


def build_failure_rate_table(
    records: list[AssetRecord],
    *,
    group_by: Callable[[AssetRecord], str] = lambda r: r.asset.workflow,
) -> list[FailureRateRow]:
    """Attempts vs. successes, grouped (by workflow by default) -- the design
    doc calls this out as *the* number worth reporting."""
    groups: dict[str, list[AssetRecord]] = {}
    for r in records:
        groups.setdefault(group_by(r), []).append(r)

    rows = [
        FailureRateRow(
            label=label,
            # max(attempt, 1): a terminal record always represents at least one
            # attempt, even if the attempt counter wasn't bumped for some reason;
            # avoids a spurious 0-attempt row skewing the total.
            attempts=sum(max(rec.attempt, 1) for rec in recs),
            successes=sum(1 for rec in recs if rec.state == AssetState.SAVED),
        )
        for label, recs in sorted(groups.items())
    ]
    if len(rows) > 1:
        rows.append(
            FailureRateRow(
                label="TOTAL",
                attempts=sum(row.attempts for row in rows),
                successes=sum(row.successes for row in rows),
            )
        )
    return rows


def build_todo(report: MorningReport) -> list[str]:
    """Short actionable to-do list a human reviewer skims first."""
    todo: list[str] = []
    if report.awaiting_approval:
        todo.append(f"Approve or reject {len(report.awaiting_approval)} asset(s) awaiting review.")
    if report.judge_unavailable:
        todo.append(
            f"Re-run the judge on {len(report.judge_unavailable)} asset(s) it could not evaluate "
            "(infra issue, not a quality failure)."
        )
    if report.permanent_errors:
        todo.append(f"Investigate {len(report.permanent_errors)} permanent error(s) before the next batch.")
    if report.skipped_for_review:
        todo.append(f"Review {len(report.skipped_for_review)} asset(s) the judge flagged for manual review.")
    if not todo:
        todo.append("Nothing needs attention -- all assets completed cleanly.")
    return todo


def skip_reason(record: AssetRecord) -> str:
    """Human-readable reason a SKIPPED_REVIEW asset needs a look: the judge's
    own verdict, falling back to the raw error if there's no verdict at all."""
    if record.last_verdict is not None:
        failed = ", ".join(record.last_verdict.failed_checks) or "unspecified check"
        critique = f" -- {record.last_verdict.critique}" if record.last_verdict.critique else ""
        return f"failed: {failed}{critique}"
    return error_reason(record)


def error_reason(record: AssetRecord) -> str:
    """Human-readable reason for a JUDGE_UNAVAILABLE / PERMANENT_ERROR record."""
    if record.last_error is not None:
        return record.last_error.exception_message or record.last_error.exception_type or "unknown error"
    return "no reason recorded"


def flagged_frame_paths(report: MorningReport) -> list[Path]:
    """Image paths worth eyeballing in a contact sheet: everything that isn't a
    clean success. JUDGE_UNAVAILABLE assets are included too -- the frame
    itself might be fine, it's the judge that broke, but a reviewer still wants
    to glance at it."""
    flagged = report.skipped_for_review + report.judge_unavailable + report.permanent_errors
    return [Path(r.output_paths[0]) for r in flagged if r.output_paths]


# ------------------------------------------------------------------ coalescing


@dataclass
class _PendingAlert:
    title: str
    details: list[str] = field(default_factory=list)


class AlertCoalescer:
    """Buffers `submit(title, detail)` calls sharing a title for `window_s`
    seconds and flushes them as a single call to `sink` -- turns "50 image
    failures" into one message rather than fifty.

    Kept here rather than duplicated in each notifier because the *decision*
    of when N buffered items become 1 outgoing item is transport-agnostic; a
    notifier only supplies `sink`, the thing that actually sends.
    """

    def __init__(
        self,
        sink: Callable[[str, list[str]], Awaitable[None]],
        *,
        window_s: float = 20.0,
    ) -> None:
        self._sink = sink
        self._window_s = window_s
        self._pending: dict[str, _PendingAlert] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def submit(self, title: str, detail: str) -> None:
        pending = self._pending.get(title)
        if pending is None:
            self._pending[title] = _PendingAlert(title=title, details=[detail])
            self._tasks[title] = asyncio.create_task(self._flush_after(title))
        else:
            pending.details.append(detail)

    async def _flush_after(self, title: str) -> None:
        try:
            await asyncio.sleep(self._window_s)
        finally:
            self._tasks.pop(title, None)
        await self._flush_one(title)

    async def _flush_one(self, title: str) -> None:
        pending = self._pending.pop(title, None)
        if pending is not None:
            await self._sink(pending.title, pending.details)

    async def aclose(self) -> None:
        """Flush everything immediately, cancelling timers. Called at batch
        shutdown, and by tests -- it's what makes coalescing assertable
        without sleeping for the full window."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        for title in list(self._pending.keys()):
            await self._flush_one(title)
