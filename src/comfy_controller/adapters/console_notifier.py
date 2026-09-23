"""A stdout-only NotifierPort -- the safety net for local dev/dry-runs, and for
anyone who just doesn't want Slack. Same coalescing behaviour as Slack's
notifier: a burst of alert() calls for the same title still becomes one
printed block, not fifty lines.
"""

from __future__ import annotations

import structlog

from ..models import AssetRecord, AssetState
from ..reporting import AlertCoalescer, build_morning_report, error_reason, skip_reason

log = structlog.get_logger(__name__)

_MAX_LISTED = 20


class ConsoleNotifier:
    """Implements NotifierPort by printing to stdout. Every public method
    swallows and logs its own errors -- printing must never take down the
    batch, same rule as the Slack notifier."""

    def __init__(self, *, output_folder: str | None = None, coalesce_window_s: float = 20.0) -> None:
        self._output_folder = output_folder
        self._coalescer = AlertCoalescer(self._print_coalesced_alert, window_s=coalesce_window_s)

    async def batch_started(self, total: int) -> None:
        try:
            print(f"[batch] started -- {total} asset(s) queued")
        except Exception:
            log.exception("console_batch_started_failed", total=total)

    async def milestone(self, done: int, total: int) -> None:
        try:
            pct = 0 if total == 0 else round(done / total * 100)
            print(f"[milestone] {pct}% complete -- {done}/{total}")
        except Exception:
            log.exception("console_milestone_failed", done=done, total=total)

    async def alert(self, title: str, detail: str) -> None:
        try:
            await self._coalescer.submit(title, detail)
        except Exception:
            log.exception("console_alert_failed", title=title)

    async def morning_report(self, summary: dict[AssetState, int], records: list[AssetRecord]) -> None:
        try:
            report = build_morning_report(summary, records, output_folder=self._output_folder)
            print(_render_morning_report_text(report))
        except Exception:
            log.exception("console_morning_report_failed")

    async def aclose(self) -> None:
        """Flush any still-buffered alerts immediately. Not part of
        NotifierPort; call at batch shutdown so nothing buffered is lost."""
        await self._coalescer.aclose()

    async def _print_coalesced_alert(self, title: str, details: list[str]) -> None:
        n = len(details)
        if n == 1:
            print(f"[alert] {title}: {details[0]}")
            return
        shown = details[:_MAX_LISTED]
        print(f"[alert] {title} ({n}x)")
        for d in shown:
            print(f"    - {d}")
        if n > len(shown):
            print(f"    ... and {n - len(shown)} more")


def _render_morning_report_text(report) -> str:  # noqa: ANN001 - MorningReport, kept untyped to avoid import cycle noise
    lines = ["=== Morning report ==="]
    folder = f" (-> {report.output_folder})" if report.output_folder else ""
    lines.append(f"Completed: {report.completed_count}{folder}")

    def _section(title: str, records, reason_fn) -> None:
        if not records:
            return
        lines.append(f"\n{title} ({len(records)}):")
        for r in records[:_MAX_LISTED]:
            lines.append(f"  - {r.asset.id}: {reason_fn(r)}")
        if len(records) > _MAX_LISTED:
            lines.append(f"  ... and {len(records) - _MAX_LISTED} more")

    _section("Awaiting approval", report.awaiting_approval, lambda r: "pending human review")
    _section("Skipped for review", report.skipped_for_review, skip_reason)
    # Its own section, distinct from "skipped for review": infra broke here,
    # the asset itself was never judged as bad.
    _section("Judge unavailable (infra issue, not a quality failure)", report.judge_unavailable, error_reason)
    _section("Permanent errors", report.permanent_errors, error_reason)

    if report.failure_rate:
        lines.append("\nFailure rate (attempts vs. successes):")
        header = f"  {'workflow':<24} {'attempts':>8} {'successes':>10} {'fail %':>7}"
        lines.append(header)
        for row in report.failure_rate:
            lines.append(
                f"  {row.label[:24]:<24} {row.attempts:>8} {row.successes:>10} {row.failure_rate * 100:>6.1f}%"
            )

    lines.append("\nTo-do:")
    for t in report.todo:
        lines.append(f"  - {t}")

    return "\n".join(lines)
