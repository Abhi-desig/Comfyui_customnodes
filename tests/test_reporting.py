"""Tests for reporting.py (pure logic) and the two NotifierPort adapters.

No network anywhere: Slack is always a fake client/webhook object we inject
and inspect, never the real slack_sdk transport.
"""

from __future__ import annotations

import pytest

from comfy_controller.adapters.console_notifier import ConsoleNotifier
from comfy_controller.adapters.slack_notifier import SlackNotifier
from comfy_controller.models import (
    Asset,
    AssetRecord,
    AssetState,
    CheckResult,
    ExecutionError,
    QCVerdict,
)
from comfy_controller.reporting import (
    AlertCoalescer,
    build_failure_rate_table,
    build_morning_report,
    error_reason,
    flagged_frame_paths,
    skip_reason,
)

# --------------------------------------------------------------------- helpers


def make_record(
    asset_id: str,
    *,
    workflow: str = "wf.json",
    state: AssetState = AssetState.SAVED,
    attempt: int = 1,
    output_paths: list[str] | None = None,
    last_error: ExecutionError | None = None,
    last_verdict: QCVerdict | None = None,
) -> AssetRecord:
    return AssetRecord(
        asset=Asset(id=asset_id, workflow=workflow),
        state=state,
        attempt=attempt,
        output_paths=output_paths or [],
        last_error=last_error,
        last_verdict=last_verdict,
    )


def make_error(msg: str = "boom") -> ExecutionError:
    return ExecutionError(prompt_id="p1", exception_type="RuntimeError", exception_message=msg)


def make_verdict(*, failed: str = "colour") -> QCVerdict:
    return QCVerdict(
        checks=[
            CheckResult(name="logo", evidence="present", passed=True),
            CheckResult(name=failed, evidence="off-brand", passed=False),
        ],
        critique="colour palette doesn't match brief",
    )


class FakeSlackClient:
    """Stands in for slack_sdk.WebClient. Records every call; can be told to
    raise, to prove a Slack outage never propagates."""

    def __init__(self, *, raise_error: bool = False) -> None:
        self.raise_error = raise_error
        self.messages: list[dict] = []
        self.files: list[dict] = []

    def chat_postMessage(self, *, channel, text, blocks=None, **kwargs):
        if self.raise_error:
            raise RuntimeError("slack is down")
        self.messages.append({"channel": channel, "text": text, "blocks": blocks})
        return {"ok": True}

    def files_upload_v2(self, *, channel, content, filename, title=None, initial_comment=None, **kwargs):
        if self.raise_error:
            raise RuntimeError("slack is down")
        self.files.append({"channel": channel, "filename": filename, "content": content})
        return {"ok": True}


class FakeWebhookClient:
    """Stands in for slack_sdk.WebhookClient (no file uploads -- webhooks can't)."""

    def __init__(self, *, raise_error: bool = False) -> None:
        self.raise_error = raise_error
        self.sent: list[dict] = []

    def send(self, *, text, blocks=None, **kwargs):
        if self.raise_error:
            raise RuntimeError("webhook is down")
        self.sent.append({"text": text, "blocks": blocks})
        return None


# -------------------------------------------------------------- reporting.py


def test_morning_report_groups_by_terminal_state():
    records = [
        make_record("skip-1", state=AssetState.SKIPPED_REVIEW, last_verdict=make_verdict()),
        make_record("wait-1", state=AssetState.PARKED_APPROVAL),
        make_record("judge-1", state=AssetState.JUDGE_UNAVAILABLE, last_error=make_error("429 exhausted")),
        make_record("err-1", state=AssetState.PERMANENT_ERROR, last_error=make_error("missing input.png")),
        make_record("ok-1", state=AssetState.SAVED),
        make_record("inflight-1", state=AssetState.SUBMITTED),
    ]
    report = build_morning_report(None, records, output_folder="/out/batch42")

    assert report.completed_count == 1
    assert report.output_folder == "/out/batch42"
    assert [r.asset.id for r in report.skipped_for_review] == ["skip-1"]
    assert [r.asset.id for r in report.awaiting_approval] == ["wait-1"]
    assert [r.asset.id for r in report.judge_unavailable] == ["judge-1"]
    assert [r.asset.id for r in report.permanent_errors] == ["err-1"]
    assert report.still_in_progress == 1


def test_judge_unavailable_is_never_merged_into_skipped_for_review():
    records = [
        make_record("skip-1", state=AssetState.SKIPPED_REVIEW, last_verdict=make_verdict()),
        make_record("judge-1", state=AssetState.JUDGE_UNAVAILABLE, last_error=make_error("529 overloaded")),
    ]
    report = build_morning_report(None, records)

    skipped_ids = {r.asset.id for r in report.skipped_for_review}
    judge_ids = {r.asset.id for r in report.judge_unavailable}
    assert skipped_ids == {"skip-1"}
    assert judge_ids == {"judge-1"}
    assert skipped_ids.isdisjoint(judge_ids)

    # And the reasons read distinctly: a rubric failure vs. an infra error.
    assert "failed:" in skip_reason(report.skipped_for_review[0])
    assert error_reason(report.judge_unavailable[0]) == "529 overloaded"


def test_completed_count_prefers_summary_over_records():
    records = [make_record("a", state=AssetState.SAVED)]
    summary = {AssetState.SAVED: 100}
    report = build_morning_report(summary, records)
    # summary is the store's authoritative count and may cover far more
    # successes than the caller bothered attaching detail records for.
    assert report.completed_count == 100


def test_todo_lists_something_for_every_non_empty_bucket():
    records = [
        make_record("wait-1", state=AssetState.PARKED_APPROVAL),
        make_record("judge-1", state=AssetState.JUDGE_UNAVAILABLE, last_error=make_error()),
    ]
    report = build_morning_report(None, records)
    joined = " ".join(report.todo)
    assert "awaiting review" in joined.lower() or "approve" in joined.lower()
    assert "judge" in joined.lower()


def test_todo_says_nothing_needed_on_a_clean_run():
    report = build_morning_report(None, [make_record("ok-1", state=AssetState.SAVED)])
    assert report.todo == ["Nothing needs attention -- all assets completed cleanly."]


def test_failure_rate_table_arithmetic():
    records = [
        make_record("a1", workflow="a.json", state=AssetState.SAVED, attempt=1),
        make_record("a2", workflow="a.json", state=AssetState.PERMANENT_ERROR, attempt=2),
        make_record("b1", workflow="b.json", state=AssetState.SAVED, attempt=1),
        make_record("b2", workflow="b.json", state=AssetState.SAVED, attempt=1),
    ]
    rows = build_failure_rate_table(records)
    by_label = {r.label: r for r in rows}

    assert by_label["a.json"].attempts == 3   # 1 + 2
    assert by_label["a.json"].successes == 1
    assert by_label["a.json"].failure_rate == pytest.approx(2 / 3)

    assert by_label["b.json"].attempts == 2
    assert by_label["b.json"].successes == 2
    assert by_label["b.json"].failure_rate == 0.0

    assert by_label["TOTAL"].attempts == 5
    assert by_label["TOTAL"].successes == 3
    assert by_label["TOTAL"].failure_rate == pytest.approx(2 / 5)


def test_failure_rate_row_handles_zero_attempts():
    rows = build_failure_rate_table([])
    assert rows == []


def test_flagged_frame_paths_covers_all_three_flagged_buckets_but_not_completed():
    records = [
        make_record("ok-1", state=AssetState.SAVED, output_paths=["/out/ok.png"]),
        make_record("skip-1", state=AssetState.SKIPPED_REVIEW, output_paths=["/out/skip.png"], last_verdict=make_verdict()),
        make_record("judge-1", state=AssetState.JUDGE_UNAVAILABLE, output_paths=["/out/judge.png"], last_error=make_error()),
        make_record("err-1", state=AssetState.PERMANENT_ERROR, output_paths=[], last_error=make_error()),
    ]
    report = build_morning_report(None, records)
    paths = {str(p) for p in flagged_frame_paths(report)}
    assert paths == {"/out/skip.png", "/out/judge.png"}


# -------------------------------------------------------------- AlertCoalescer


async def test_alert_coalescer_batches_same_title_into_one_sink_call():
    sent: list[tuple[str, list[str]]] = []

    async def sink(title: str, details: list[str]) -> None:
        sent.append((title, details))

    coalescer = AlertCoalescer(sink, window_s=999)  # never fires on its own in this test
    for i in range(50):
        await coalescer.submit("asset failed", f"asset-{i} crashed")
    await coalescer.aclose()

    assert len(sent) == 1
    title, details = sent[0]
    assert title == "asset failed"
    assert len(details) == 50


async def test_alert_coalescer_keeps_distinct_titles_separate():
    sent: list[tuple[str, list[str]]] = []

    async def sink(title: str, details: list[str]) -> None:
        sent.append((title, details))

    coalescer = AlertCoalescer(sink, window_s=999)
    await coalescer.submit("comfyui restarted", "restart 1")
    await coalescer.submit("3 crashes in a row", "asset-9")
    await coalescer.aclose()

    assert {title for title, _ in sent} == {"comfyui restarted", "3 crashes in a row"}
    assert len(sent) == 2


# ------------------------------------------------------------------- SlackNotifier


async def test_slack_batch_started_sends_one_message():
    client = FakeSlackClient()
    notifier = SlackNotifier(channel="C1", client=client)
    await notifier.batch_started(120)
    assert len(client.messages) == 1
    assert "120" in client.messages[0]["text"]


async def test_slack_alert_burst_coalesces_into_one_message_not_fifty():
    client = FakeSlackClient()
    notifier = SlackNotifier(channel="C1", client=client, coalesce_window_s=999)
    for i in range(50):
        await notifier.alert("image failed QC", f"asset-{i}: colour check failed")
    await notifier.aclose()

    assert len(client.messages) == 1
    header = client.messages[0]["blocks"][0]["text"]["text"]
    assert header == "image failed QC (50x)"


async def test_slack_morning_report_sections_and_judge_unavailable_is_distinct():
    client = FakeSlackClient()
    notifier = SlackNotifier(channel="C1", client=client, output_folder="/out/batch42")
    records = [
        make_record("skip-1", state=AssetState.SKIPPED_REVIEW, last_verdict=make_verdict()),
        make_record("wait-1", state=AssetState.PARKED_APPROVAL),
        make_record("judge-1", state=AssetState.JUDGE_UNAVAILABLE, last_error=make_error("429 exhausted")),
        make_record("err-1", state=AssetState.PERMANENT_ERROR, last_error=make_error("missing input.png")),
        make_record("ok-1", state=AssetState.SAVED),
    ]
    summary = {AssetState.SAVED: 1}
    await notifier.morning_report(summary, records)

    assert len(client.messages) == 1
    text_blocks = [b["text"]["text"] for b in client.messages[0]["blocks"] if b.get("type") == "section"]
    joined = "\n".join(text_blocks)

    assert "/out/batch42" in joined
    assert "Completed:* 1" in joined
    assert "Awaiting approval (1)" in joined
    assert "Skipped for review (1)" in joined
    assert "Judge unavailable (1)" in joined
    assert "infra issue, not a quality failure" in joined
    assert "Permanent errors (1)" in joined
    # The judge-unavailable reason and the rubric-failure reason must not
    # bleed into each other's section.
    skipped_block = next(t for t in text_blocks if "Skipped for review" in t)
    judge_block = next(t for t in text_blocks if "Judge unavailable" in t)
    assert "429 exhausted" not in skipped_block
    assert "429 exhausted" in judge_block
    assert "colour palette" not in judge_block


async def test_slack_morning_report_attaches_contact_sheet_when_flagged_frames_exist():
    client = FakeSlackClient()
    calls: list[list] = []

    def fake_builder(paths):
        calls.append(paths)
        return b"fake-png-bytes"

    notifier = SlackNotifier(channel="C1", client=client, contact_sheet_builder=fake_builder)
    records = [
        make_record("err-1", state=AssetState.PERMANENT_ERROR, output_paths=["/out/err.png"], last_error=make_error()),
    ]
    await notifier.morning_report({}, records)

    assert len(client.files) == 1
    assert client.files[0]["content"] == b"fake-png-bytes"
    assert len(calls) == 1 and str(calls[0][0]) == "/out/err.png"


async def test_slack_no_contact_sheet_attempt_when_nothing_flagged():
    client = FakeSlackClient()
    called = False

    def fake_builder(paths):
        nonlocal called
        called = True
        return b"x"

    notifier = SlackNotifier(channel="C1", client=client, contact_sheet_builder=fake_builder)
    await notifier.morning_report({AssetState.SAVED: 5}, [make_record("ok-1", state=AssetState.SAVED)])
    assert called is False
    assert client.files == []


async def test_slack_falls_back_to_webhook_when_no_bot_token():
    webhook = FakeWebhookClient()
    notifier = SlackNotifier(webhook_client=webhook)
    await notifier.batch_started(10)
    assert len(webhook.sent) == 1


async def test_slack_webhook_never_attempts_file_upload():
    # A webhook-only config has no files_upload_v2 at all; morning_report must
    # not even try, or this would blow up with AttributeError.
    webhook = FakeWebhookClient()
    notifier = SlackNotifier(webhook_client=webhook)
    records = [make_record("err-1", state=AssetState.PERMANENT_ERROR, output_paths=["/out/err.png"], last_error=make_error())]
    await notifier.morning_report({}, records)  # must not raise
    assert len(webhook.sent) == 1


async def test_slack_graceful_degradation_with_no_token_and_no_webhook():
    notifier = SlackNotifier()  # nothing configured at all
    # None of these may raise -- they should quietly degrade to a log line.
    await notifier.batch_started(5)
    await notifier.milestone(2, 5)
    await notifier.alert("stalled", "asset-1 stalled > 10 min")
    await notifier.aclose()
    await notifier.morning_report({}, [make_record("ok-1")])


async def test_slack_raising_client_never_propagates():
    client = FakeSlackClient(raise_error=True)
    notifier = SlackNotifier(channel="C1", client=client)
    # Every one of these would raise inside the fake client; none may escape.
    await notifier.batch_started(1)
    await notifier.milestone(1, 2)
    await notifier.alert("crash", "boom")
    await notifier.aclose()
    await notifier.morning_report({}, [make_record("err-1", state=AssetState.PERMANENT_ERROR, last_error=make_error())])


async def test_slack_bot_token_without_channel_falls_back_safely():
    # A bot client with no channel configured can't call chat.postMessage
    # (which requires one), so the notifier must demote itself rather than
    # blow up on the first real send.
    client = FakeSlackClient()
    notifier = SlackNotifier(client=client)  # no channel=
    await notifier.batch_started(1)
    assert client.messages == []  # never touched the bot client


# ----------------------------------------------------------------- ConsoleNotifier


async def test_console_notifier_prints_morning_report_sections(capsys):
    notifier = ConsoleNotifier(output_folder="/out/batch42")
    records = [
        make_record("skip-1", state=AssetState.SKIPPED_REVIEW, last_verdict=make_verdict()),
        make_record("judge-1", state=AssetState.JUDGE_UNAVAILABLE, last_error=make_error("529 overloaded")),
        make_record("ok-1", state=AssetState.SAVED),
    ]
    await notifier.morning_report({AssetState.SAVED: 1}, records)
    out = capsys.readouterr().out

    assert "/out/batch42" in out
    assert "Skipped for review (1)" in out
    assert "Judge unavailable" in out
    assert "not a quality failure" in out


async def test_console_notifier_coalesces_alert_burst(capsys):
    notifier = ConsoleNotifier(coalesce_window_s=999)
    for i in range(10):
        await notifier.alert("image failed", f"asset-{i}")
    await notifier.aclose()
    out = capsys.readouterr().out

    assert out.count("[alert]") == 1
    assert "(10x)" in out


async def test_console_notifier_never_raises_on_a_broken_record(capsys):
    # A record with no error and no verdict at all (edge case) must still
    # render something sane rather than raising KeyError/AttributeError.
    notifier = ConsoleNotifier()
    await notifier.morning_report({}, [make_record("mystery-1", state=AssetState.PERMANENT_ERROR)])
    out = capsys.readouterr().out
    assert "mystery-1" in out
    assert "no reason recorded" in out
