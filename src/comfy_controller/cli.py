"""argparse CLI: run / resume / status / approve / report.

`resume` is deliberately not a separate code path: startup reconciliation in
`BatchRunner.run()` is what makes "run again against an existing DB" safe, so
`resume` calls the exact same entry point `run` does, just with the manifest
argument optional (idempotent `load_batch` means passing it again is also
safe -- `resume` differs only in not requiring it).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Any

import structlog

from .adapters.comfy_http import ComfyHTTP
from .adapters.console_notifier import ConsoleNotifier
from .adapters.slack_notifier import SlackNotifier
from .adapters.sqlite_store import SQLiteStore
from .config import AppConfig, ConfigError
from .core.policy import CircuitBreaker, GlobalBreaker, RetryPolicy
from .core.sequence import SequenceRepairPlanner
from .manifest import ManifestError, load_manifest
from .models import AssetState
from .ports import NotifierPort
from .runner import BatchRunner, BudgetLimits, RunnerConfig, read_all_records
from .supervisor import HealthSupervisor
from .testing.fake_judge import FakeJudge

log = structlog.get_logger(__name__)


# ------------------------------------------------------------------ logging


def configure_logging(*, level: int = logging.INFO, json_logs: bool | None = None) -> None:
    """Configure structlog once, at process startup.

    Every module in `src/` does `structlog.get_logger(__name__)` and nothing
    ever called `structlog.configure()`, so the whole controller ran on
    structlog's implicit defaults: a lazy proxy that only binds on first use,
    no level filtering, and a console renderer regardless of whether a human
    or a log collector is reading. For an unattended overnight batch the
    *logs are the post-mortem*, so this pins the processor chain explicitly
    and routes it through stdlib `logging` so third-party libraries
    (aiohttp, anthropic, slack_sdk) land in the same stream.

    `json_logs` defaults to "JSON when stderr is not a TTY" -- readable when
    an operator is watching, machine-parseable under systemd/cron.
    """
    if json_logs is None:
        json_logs = not sys.stderr.isatty()

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    shared: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    renderer = (
        structlog.processors.JSONRenderer()
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=level, force=True)
    handler = logging.getLogger().handlers[0]
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ], foreign_pre_chain=shared)
    )

    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# ------------------------------------------------------------------- wiring


def _build_notifier(cfg: AppConfig) -> NotifierPort:
    if cfg.notifier == "slack":
        return SlackNotifier(
            bot_token=cfg.slack_bot_token,
            webhook_url=cfg.slack_webhook_url,
            channel=cfg.slack_channel,
        )
    return ConsoleNotifier()


def _build_judge(cfg: AppConfig):
    if cfg.judge_backend == "fake":
        # For dry runs/dev only -- `AppConfig.load` already refuses
        # judge_backend=claude without ANTHROPIC_API_KEY, so this branch is
        # the escape hatch for testing the wiring without a real API key.
        return FakeJudge()
    import anthropic

    from .adapters.claude_judge import ClaudeJudge, RubricConfig

    rubric = RubricConfig.from_yaml(cfg.rubric_path)
    client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
    return ClaudeJudge(client, rubric)


def build_runner(cfg: AppConfig) -> tuple[BatchRunner, SQLiteStore, ComfyHTTP]:
    store = SQLiteStore(cfg.db_path)
    comfy = ComfyHTTP(cfg.comfy_base_url)
    judge = _build_judge(cfg)
    notifier = _build_notifier(cfg)
    supervisor = HealthSupervisor(
        comfy,
        stall_timeout=cfg.supervisor.stall_timeout,
        canary_timeout=cfg.supervisor.canary_timeout,
        hard_ceiling=cfg.supervisor.hard_ceiling,
        hysteresis_n=cfg.supervisor.hysteresis_n,
        backoff_base=cfg.supervisor.backoff_base,
        backoff_cap=cfg.supervisor.backoff_cap,
        intensity_max_restarts=cfg.supervisor.intensity_max_restarts,
        intensity_window=cfg.supervisor.intensity_window,
        recycle_every_n_jobs=cfg.supervisor.recycle_every_n_jobs,
        launch_argv=cfg.supervisor.launch_argv,
    )
    runner_config = RunnerConfig(
        model_version=cfg.model_version,
        await_outcome_timeout_s=cfg.await_outcome_timeout_s,
        concurrency=cfg.concurrency,
        tick_interval_s=cfg.tick_interval_s,
        stall_alert_s=cfg.stall_alert_s,
        milestone_fractions=cfg.milestone_fractions,
        comfy_output_dir=Path(cfg.comfy_output_dir) if cfg.comfy_output_dir else None,
        budget=BudgetLimits(
            max_wall_clock_s=cfg.budget.max_wall_clock_s,
            max_gpu_seconds=cfg.budget.max_gpu_seconds,
            max_judge_usd=cfg.budget.max_judge_usd,
        ),
    )
    breaker = CircuitBreaker(
        now=time.monotonic, failure_threshold=cfg.breaker.failure_threshold, cooldown_s=cfg.breaker.cooldown_s
    )
    global_breaker = GlobalBreaker(
        now=time.monotonic,
        crash_threshold=cfg.global_breaker.crash_threshold,
        crash_window_s=cfg.global_breaker.crash_window_s,
    )
    def _judge_cost_usd() -> float:
        # ClaudeJudge exposes a `.cost` CostTracker; FakeJudge doesn't track
        # spend at all, so default to 0.0 (no judge-cost budget to enforce).
        return getattr(getattr(judge, "cost", None), "total_usd", 0.0)

    runner = BatchRunner(
        comfy=comfy,
        store=store,
        judge=judge,
        supervisor=supervisor,
        notifier=notifier,
        config=runner_config,
        breaker=breaker,
        global_breaker=global_breaker,
        retry_policy=RetryPolicy(),
        seq_planner=SequenceRepairPlanner(),
        frame_consistency=_build_frame_consistency(cfg),
        judge_cost_usd_fn=_judge_cost_usd,
    )
    return runner, store, comfy


def _build_frame_consistency(cfg: AppConfig):
    """Construct the cross-frame consistency checker for SEQUENCE assets.

    `BatchRunner` takes this as an optional dependency and nothing ever
    supplied one, so `config/rubric.yaml`'s whole `frame_consistency:` section
    configured a code path that could not run and "the product changed between
    frame 4 and 5" went undetected. Disabled explicitly via
    `frame_consistency: false` in the config; otherwise built from the rubric
    file, falling back to the checker's own defaults if that file has no such
    section. Construction probes for optional torch/CLIP/LPIPS backends, so a
    failure here degrades to "no cross-frame checking" rather than refusing to
    start the batch.
    """
    if not cfg.frame_consistency:
        return None
    from .adapters.frame_consistency import ConsistencyConfig, FrameConsistencyChecker

    try:
        consistency_cfg = (
            ConsistencyConfig.from_yaml(cfg.rubric_path)
            if Path(cfg.rubric_path).exists()
            else ConsistencyConfig()
        )
        return FrameConsistencyChecker(consistency_cfg)
    except Exception:
        log.exception("frame_consistency_unavailable", rubric_path=cfg.rubric_path)
        return None


# --------------------------------------------------------------------- commands


async def _aclose_quietly(obj: object, method: str) -> None:
    """Best-effort shutdown of one component. A teardown failure must never
    mask the run's own outcome, nor stop the remaining components closing."""
    fn = getattr(obj, method, None)
    if fn is None:
        return
    try:
        await fn()
    except Exception:
        log.exception("shutdown_step_failed", component=type(obj).__name__, method=method)


async def _run_batch(config_path: str, manifest_path: str | None) -> None:
    cfg = AppConfig.load(config_path)
    runner, store, comfy = build_runner(cfg)
    assets = (
        load_manifest(manifest_path, workflow_dir=cfg.workflow_dir, max_frame_count=cfg.max_frame_count)
        if manifest_path
        else None
    )
    try:
        await runner.run(assets)
    finally:
        # Order matters. The notifier flushes LAST-but-one because its
        # coalescer may still be holding the final alerts of the run, and
        # those are worth more than anything else here. The supervisor closes
        # its event pump AND terminates the ComfyUI child it spawned --
        # neither was ever called, so every controller exit left an orphaned
        # ComfyUI process holding the GPU against the next run.
        await _aclose_quietly(runner.notifier, "aclose")
        await _aclose_quietly(runner.supervisor, "aclose")
        await comfy.close()
        await store.close()


def cmd_run(args: argparse.Namespace) -> int:
    asyncio.run(_run_batch(args.config, args.manifest))
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    asyncio.run(_run_batch(args.config, args.manifest))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(args.config)
    store = SQLiteStore(cfg.db_path)
    try:
        summary = asyncio.run(store.summary())
    finally:
        asyncio.run(store.close())

    if not summary:
        print("No assets loaded yet.")
        return 0
    for state in AssetState:
        count = summary.get(state, 0)
        if count:
            print(f"{state.value:<20} {count}")
    print(f"{'TOTAL':<20} {sum(summary.values())}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(args.config)
    store = SQLiteStore(cfg.db_path)
    try:
        summary = asyncio.run(store.summary())
        records = read_all_records(store)
    finally:
        asyncio.run(store.close())

    from .adapters.console_notifier import _render_morning_report_text
    from .reporting import build_morning_report

    report = build_morning_report(summary, records)
    print(_render_morning_report_text(report))
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(args.config)
    store = SQLiteStore(cfg.db_path)
    try:
        records = {r.asset.id: r for r in read_all_records(store)}
        rec = records.get(args.asset_id)
        if rec is None:
            print(f"Unknown asset id: {args.asset_id!r}", file=sys.stderr)
            return 1
        asyncio.run(
            store.transition(
                args.asset_id,
                AssetState.SAVED,
                output_paths=rec.output_paths or None,
                verdict=rec.last_verdict,
            )
        )
    finally:
        asyncio.run(store.close())
    print(f"Approved {args.asset_id} -> SAVED")
    return 0


# ------------------------------------------------------------------------ argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="comfyctl")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Run a new batch from a manifest.")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("manifest")
    p_run.set_defaults(func=cmd_run)

    p_resume = sub.add_parser("resume", help="Resume an interrupted batch against its existing DB.")
    p_resume.add_argument("--config", required=True)
    p_resume.add_argument("manifest", nargs="?", default=None)
    p_resume.set_defaults(func=cmd_resume)

    p_status = sub.add_parser("status", help="Print current state counts.")
    p_status.add_argument("--config", required=True)
    p_status.set_defaults(func=cmd_status)

    p_approve = sub.add_parser("approve", help="Approve an asset (e.g. PARKED_APPROVAL) as SAVED.")
    p_approve.add_argument("--config", required=True)
    p_approve.add_argument("asset_id")
    p_approve.set_defaults(func=cmd_approve)

    p_report = sub.add_parser("report", help="Print the morning report for the current DB.")
    p_report.add_argument("--config", required=True)
    p_report.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ConfigError, ManifestError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
