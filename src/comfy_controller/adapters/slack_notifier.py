"""Slack notifications for the overnight batch: live alerts + the morning report.

Implements `NotifierPort` (see ports.py). Uses a bot token (`WebClient`) when
configured so the morning report can attach a contact sheet of flagged frames
via `files_upload_v2`; falls back to an incoming webhook, and then to a plain
log line, so a missing/partial Slack config never blocks a run.

Every public method swallows and logs its own errors -- per the design doc, a
notifier is not allowed to take down the batch.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import structlog
from slack_sdk import WebClient, WebhookClient
from slack_sdk.models.blocks import (
    Block,
    DividerBlock,
    HeaderBlock,
    MarkdownTextObject,
    SectionBlock,
)

from ..models import AssetRecord, AssetState
from ..reporting import (
    AlertCoalescer,
    FailureRateRow,
    MorningReport,
    build_morning_report,
    error_reason,
    flagged_frame_paths,
    skip_reason,
)

log = structlog.get_logger(__name__)

_MAX_LISTED = 20  # cap per-section detail lines; the full counts still show


def mrkdwn_escape(text: Any) -> str:
    """Escape the three characters Slack's mrkdwn parser treats as markup.

    Everything interpolated into a block here is untrusted: asset ids come
    from an operator's manifest, and `critique` is model output. Slack's link
    syntax is `<url|label>`, so an unescaped critique reading
    `<https://evil.example|Approved>` renders in the morning report as a real,
    clickable link labelled "Approved" -- a phishing primitive delivered by
    our own alerting. Per Slack's own guidance these three, and only these
    three, are escaped; `*`/`_`/`` ` `` are harmless formatting.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


class SlackNotifier:
    """NotifierPort backed by Slack. Live alerts (`alert`) are coalesced per
    title via `AlertCoalescer` -- a burst of per-image failures collapses into
    one message instead of fifty, per the design doc's "no per-image messages"
    rule.
    """

    def __init__(
        self,
        *,
        bot_token: str | None = None,
        webhook_url: str | None = None,
        channel: str | None = None,
        output_folder: str | None = None,
        coalesce_window_s: float = 20.0,
        contact_sheet_builder: Any = None,
        client: WebClient | None = None,
        webhook_client: WebhookClient | None = None,
    ) -> None:
        self._channel = channel
        self._output_folder = output_folder
        # Lazily built (needs PIL); injectable so tests never need real images.
        self._contact_sheet_builder = contact_sheet_builder or _default_contact_sheet

        # An injected client (tests / callers with their own session) always
        # wins over building a real one from config.
        self._client = client if client is not None else (WebClient(token=bot_token) if bot_token else None)
        self._webhook = (
            webhook_client if webhook_client is not None else (WebhookClient(webhook_url) if webhook_url else None)
        )
        if self._client is not None and not self._channel:
            log.warning(
                "slack_bot_token_without_channel",
                detail="bot token configured but no channel given; falling back to webhook/text",
            )
            self._client = None

        self._coalescer = AlertCoalescer(self._send_coalesced_alert, window_s=coalesce_window_s)

    # ------------------------------------------------------------- NotifierPort

    async def batch_started(self, total: int) -> None:
        try:
            text = f"Batch started -- *{total}* asset(s) queued."
            await self._send_blocks(
                [HeaderBlock(text="Batch started"), SectionBlock(text=MarkdownTextObject(text=text))],
                fallback_text=f"Batch started: {total} assets queued",
            )
        except Exception:
            log.exception("slack_batch_started_failed", total=total)

    async def milestone(self, done: int, total: int) -> None:
        try:
            pct = 0 if total == 0 else round(done / total * 100)
            text = f"*{pct}%* complete -- {done}/{total} assets done."
            await self._send_blocks([SectionBlock(text=MarkdownTextObject(text=text))], fallback_text=text)
        except Exception:
            log.exception("slack_milestone_failed", done=done, total=total)

    async def alert(self, title: str, detail: str) -> None:
        # Never call Slack synchronously here -- just buffer. The coalescer
        # owns the timing; this method can't fail in an interesting way.
        try:
            await self._coalescer.submit(title, detail)
        except Exception:
            log.exception("slack_alert_failed", title=title)

    async def morning_report(self, summary: dict[AssetState, int], records: list[AssetRecord]) -> None:
        try:
            report = build_morning_report(summary, records, output_folder=self._output_folder)
            blocks = _render_morning_report_blocks(report)
            await self._send_blocks(
                blocks, fallback_text=f"Morning report: {report.completed_count} completed"
            )
            await self._maybe_attach_contact_sheet(report)
        except Exception:
            log.exception("slack_morning_report_failed")

    async def aclose(self) -> None:
        """Flush any still-buffered alerts immediately. Not part of
        NotifierPort -- call it at the end of a batch so nothing buffered is
        lost when the process exits."""
        await self._coalescer.aclose()

    # ------------------------------------------------------------------ alerts

    async def _send_coalesced_alert(self, title: str, details: list[str]) -> None:
        n = len(details)
        header = title if n == 1 else f"{title} ({n}x)"
        if n == 1:
            body = mrkdwn_escape(details[0])
        else:
            shown = details[:_MAX_LISTED]
            body = "\n".join(f"- {mrkdwn_escape(d)}" for d in shown)
            if n > len(shown):
                body += f"\n... and {n - len(shown)} more"
        # HeaderBlock is plain_text, not mrkdwn, so the title needs no escaping.
        blocks = [HeaderBlock(text=header[:150]), SectionBlock(text=MarkdownTextObject(text=body))]
        await self._send_blocks(blocks, fallback_text=header)

    # ------------------------------------------------------------ contact sheet

    async def _maybe_attach_contact_sheet(self, report: MorningReport) -> None:
        if self._client is None:
            return  # files_upload_v2 needs a bot token; webhook/text can't attach files
        paths = flagged_frame_paths(report)
        if not paths:
            return
        try:
            image_bytes = self._contact_sheet_builder(paths)
        except Exception:
            log.exception("slack_contact_sheet_build_failed")
            return
        if not image_bytes:
            return
        try:
            await asyncio.to_thread(
                self._client.files_upload_v2,
                channel=self._channel,
                content=image_bytes,
                filename="contact_sheet.png",
                title="Flagged frames",
                initial_comment="Contact sheet of flagged frames for review.",
            )
        except Exception:
            log.exception("slack_contact_sheet_upload_failed")

    # ------------------------------------------------------------------ sending

    async def _send_blocks(self, blocks: list[Block], *, fallback_text: str) -> None:
        """The one place that talks to Slack. Never raises -- callers only log."""
        try:
            payload = [b.to_dict() for b in blocks]
            if self._client is not None:
                await asyncio.to_thread(
                    self._client.chat_postMessage, channel=self._channel, text=fallback_text, blocks=payload
                )
            elif self._webhook is not None:
                await asyncio.to_thread(self._webhook.send, text=fallback_text, blocks=payload)
            else:
                # Neither a bot token nor a webhook is configured -- degrade to
                # a log line rather than silently dropping the notification.
                log.info("slack_not_configured", text=fallback_text)
        except Exception:
            log.exception("slack_send_failed", text=fallback_text)


# --------------------------------------------------------------------- render


def _render_morning_report_blocks(report: MorningReport) -> list[Block]:
    blocks: list[Block] = [HeaderBlock(text="Morning report")]

    folder_line = f" -> <{report.output_folder}|output folder>" if report.output_folder else ""
    blocks.append(SectionBlock(text=MarkdownTextObject(text=f"*Completed:* {report.completed_count}{folder_line}")))

    if report.awaiting_approval:
        lines = "\n".join(f"- `{mrkdwn_escape(r.asset.id)}`" for r in report.awaiting_approval[:_MAX_LISTED])
        blocks += [
            DividerBlock(),
            SectionBlock(
                text=MarkdownTextObject(text=f"*Awaiting approval ({len(report.awaiting_approval)}):*\n{lines}")
            ),
        ]

    if report.skipped_for_review:
        lines = "\n".join(
            f"- `{mrkdwn_escape(r.asset.id)}`: {mrkdwn_escape(skip_reason(r))}"
            for r in report.skipped_for_review[:_MAX_LISTED]
        )
        blocks += [
            DividerBlock(),
            SectionBlock(
                text=MarkdownTextObject(text=f"*Skipped for review ({len(report.skipped_for_review)}):*\n{lines}")
            ),
        ]

    if report.judge_unavailable:
        # Its own section, never merged into "skipped for review": this means
        # the judge infrastructure broke, not that the asset failed the rubric.
        lines = "\n".join(
            f"- `{mrkdwn_escape(r.asset.id)}`: {mrkdwn_escape(error_reason(r))}"
            for r in report.judge_unavailable[:_MAX_LISTED]
        )
        blocks += [
            DividerBlock(),
            SectionBlock(
                text=MarkdownTextObject(
                    text=(
                        f":warning: *Judge unavailable ({len(report.judge_unavailable)})* "
                        f"-- infra issue, not a quality failure:\n{lines}"
                    )
                )
            ),
        ]

    if report.permanent_errors:
        lines = "\n".join(
            f"- `{mrkdwn_escape(r.asset.id)}`: {mrkdwn_escape(error_reason(r))}"
            for r in report.permanent_errors[:_MAX_LISTED]
        )
        blocks += [
            DividerBlock(),
            SectionBlock(
                text=MarkdownTextObject(text=f"*Permanent errors ({len(report.permanent_errors)}):*\n{lines}")
            ),
        ]

    if report.failure_rate:
        blocks += [
            DividerBlock(),
            SectionBlock(
                text=MarkdownTextObject(
                    text="*Failure rate (attempts vs. successes):*\n```\n"
                    + _render_failure_rate_table(report.failure_rate)
                    + "\n```"
                )
            ),
        ]

    blocks += [
        DividerBlock(),
        SectionBlock(text=MarkdownTextObject(text="*To-do:*\n" + "\n".join(f"- {t}" for t in report.todo))),
    ]
    return blocks


def _render_failure_rate_table(rows: list[FailureRateRow]) -> str:
    # A plain monospace table in a text block, not slack_sdk's TableBlock:
    # table blocks aren't reliably supported across all workspaces/scopes yet,
    # and this renders identically everywhere chat.postMessage works.
    header = f"{'workflow':<24} {'attempts':>8} {'successes':>10} {'fail %':>7}"
    lines = [header, "-" * len(header)]
    for row in rows:
        # Escaped even inside the code fence: Slack still resolves `<...|...>`
        # there on some clients, and `label` is a manifest-supplied path.
        label = mrkdwn_escape(row.label[:24])
        lines.append(
            f"{label:<24} {row.attempts:>8} {row.successes:>10} {row.failure_rate * 100:>6.1f}%"
        )
    return "\n".join(lines)


def _default_contact_sheet(paths: list[Path], *, tile: int = 256, columns: int = 4) -> bytes | None:
    """Simple grid montage of flagged frames so a reviewer can eyeball what
    needs attention without opening every file. Best-effort: any missing or
    corrupt image is skipped rather than failing the whole report. PIL is
    imported lazily so importing this module never requires it up front."""
    from io import BytesIO

    from PIL import Image

    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            img.thumbnail((tile, tile))
            images.append(img)
        except Exception:
            continue
    if not images:
        return None

    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * tile), color=(32, 32, 32))
    for i, img in enumerate(images):
        x, y = (i % columns) * tile, (i // columns) * tile
        sheet.paste(img, (x, y))

    buf = BytesIO()
    sheet.save(buf, format="PNG")
    return buf.getvalue()
