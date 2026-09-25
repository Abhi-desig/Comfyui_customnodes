"""BatchRunner: wires every port together into the overnight batch loop.

This is the one place that knows about every port at once (`ComfyPort`,
`StorePort`, `JudgePort`, `SupervisorPort`, `NotifierPort`) plus the pure
core logic (`ErrorClassifier`, `RetryPolicy`, `CircuitBreaker`,
`GlobalBreaker`, `SequenceRepairPlanner`). Everything it depends on is
frozen/Wave-1; this file only *orchestrates*.

Concurrency model: N worker coroutines each loop `claim_next -> process`.
A worker whose claimed asset's circuit-breaker partition is OPEN just waits
(polling `breaker.allow`) without releasing the claim -- the row stays
CLAIMED in the DB the whole time, which is exactly the state a crash would
leave it in, so `needing_reconciliation()` handles it identically either
way. Other workers keep claiming and finishing assets in OTHER partitions
while this happens, which is what keeps unrelated work moving when one
workflow's breaker trips.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from .adapters.comfy_http import ComfyValidationError, graph_hash, patch_graph
from .adapters.frame_consistency import FrameConsistencyChecker
from .core.policy import CircuitBreaker, ErrorClassifier, GlobalBreaker, RetryPolicy, DEFAULT_CLASSIFIER
from .core.sequence import QCCompare, RegenerateFrame, SequenceRepairPlanner
from .models import (
    Asset,
    AssetKind,
    AssetRecord,
    AssetState,
    ErrorClass,
    ExecutionError,
    FixProposal,
    JobOutcome,
    QCVerdict,
)
from .ports import ComfyPort, JudgePort, JudgeUnavailable, NotifierPort, StorePort, SupervisorPort

log = structlog.get_logger(__name__)

_KSAMPLER_CLASS_TYPE = "KSampler"

# An absolute ceiling on sampler steps, applied at the point the fix is
# written into the graph. `ClaudeJudge._validate_and_clamp` clamps `delta`
# only when `fix_bounds` has an entry for the action -- an empty/partial
# `fix.bounds:` in the rubric silently disables it -- and even a clamped
# delta compounds across QC rounds. This is the backstop that is not
# config-contingent: an overnight batch must never be talked into a
# thousand-step sample.
_MAX_SAMPLER_STEPS = 150

# `rewrite_slot` free text: the only judge output that reaches a generation
# prompt at all. models.py calls it "a fixed template" and rubric.yaml calls
# the slot allowlist "the prompt-injection envelope", so the text must be a
# short design phrase and nothing else -- no newlines, no control characters,
# no markup/quoting/brace characters that could restructure a prompt.
_SLOT_TEXT_EXTRA_CHARS = frozenset(" ,.-'()/&:%+")
_SLOT_TEXT_MAX_CHARS = 280


# Matches the `_qc<n>` / `_f<n>` / `_a<n>` markers the runner appends to a
# job_key for a different QC round, sequence frame, or attempt. Used to tell
# "this file belongs to the attempt in flight" from "this file belongs to a
# sibling round that shares the same base key".
_ROUND_SUFFIX_RE = re.compile(r"_(?:qc|a|f)\d+(?:_|$)")


def _image_integrity_problem(path: Path) -> str | None:
    """None when `path` is a readable, parseable image; a reason string otherwise.

    A `kill -9` in the middle of ComfyUI's `SaveImage` leaves exactly two
    artefacts: a 0-byte file, and a file with a valid header but a truncated
    body. Both look like success to a `glob`.
    """
    try:
        if not path.is_file():
            return "not a regular file"
        if path.stat().st_size == 0:
            return "0-byte file"
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"

    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()  # header + structural check, cheap: never decodes pixels
    except Exception as exc:  # noqa: BLE001 - any parse failure means "don't trust it"
        return f"unparseable image: {type(exc).__name__}: {exc}"
    return None


class WorkflowLoadError(RuntimeError):
    """A workflow file could not be read or parsed as an API-format graph.

    Per-asset and permanent: retrying the same corrupt file changes nothing,
    and one bad file on a shared drive must never abort the batch.
    """


def sanitize_slot_text(text: str) -> str | None:
    """Return `text` cleaned for insertion into a prompt slot, or None to reject.

    Conservative by construction: alphanumerics (any script) plus a small set
    of punctuation that genuinely occurs in a design phrase. Everything else
    -- newlines, tabs, control characters, angle brackets, braces, quotes,
    backslashes, backticks, `$`, `#`, `=`, `;`, `|` -- is grounds for
    rejection rather than stripping, because a phrase that needed those
    characters was not a design phrase.
    """
    if not text:
        return None
    if any(not (ch.isalnum() or ch in _SLOT_TEXT_EXTRA_CHARS) for ch in text):
        return None
    cleaned = " ".join(text.split()).strip(" ,.-")
    if not cleaned:
        return None
    return cleaned[:_SLOT_TEXT_MAX_CHARS]


def compose_slot_text(original: str, addition: str) -> str:
    """The fixed template `rewrite_slot` composes into.

    The judge's phrase is APPENDED to whatever the workflow author wrote in
    that slot; it never replaces it. Wholesale replacement is what made the
    "never emits free text that reaches a generation prompt" claim false --
    one `rewrite_slot` could discard the entire authored prompt and substitute
    model-chosen text. Appending keeps the author's intent as the load-bearing
    part of the slot and reduces the judge to a bounded refinement of it.
    """
    original = (original or "").strip()
    if not original:
        return addition
    separator = "" if original.endswith((",", ";", ".")) else ","
    return f"{original}{separator} {addition}"


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class BudgetLimits:
    """Hard caps enforced by the runner itself. Defence in depth -- the real
    dead-man's switch lives outside this process (ops/deploy, another agent's
    concern); this is what stops a runaway batch *before* that switch fires."""

    max_wall_clock_s: float | None = None
    max_gpu_seconds: float | None = None
    max_judge_usd: float | None = None


@dataclass
class RunnerConfig:
    model_version: str = "default"
    await_outcome_timeout_s: float = 1800.0
    concurrency: int = 4
    tick_interval_s: float = 30.0
    stall_alert_s: float = 900.0
    milestone_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    comfy_output_dir: Path | None = None
    budget: BudgetLimits = field(default_factory=BudgetLimits)
    idle_poll_max_s: float = 5.0

    # ops/deadman.sh layer 2 (the dead-man's switch's heartbeat watchdog)
    # reads the mtime of this file and terminates the rented GPU instance
    # once it goes stale -- see that script's header comment and
    # docs/runbook.md's "Heartbeat contract" section, which is the spec this
    # implements. None disables heartbeat writing entirely (e.g. tests that
    # don't care about it); the default path mirrors HealthSupervisor's own
    # `marker_dir` default so there is one state directory, not two.
    heartbeat_file: Path | None = Path(".comfy_supervisor/heartbeat")
    # How often the file is actually re-written. Throttled independently of
    # any one asset's processing time -- see `_maybe_touch_heartbeat`'s own
    # comment for why this is called from *every* worker-loop iteration
    # rather than from a separate timer task.
    heartbeat_interval_s: float = 30.0


# --------------------------------------------------------------------- helpers


def read_all_records(store: object) -> list[AssetRecord]:
    """Best-effort full-table read for reporting/admin purposes.

    `StorePort` deliberately exposes only claim/transition/summary -- no bulk
    read -- so the hot path never depends on scanning the whole table.
    Reporting is different: the morning report and the CLI's status/report/
    approve commands need every row, not just counts. Rather than widen the
    frozen `StorePort` protocol for an admin-only need, duck-type onto
    `SQLiteStore`'s own connection (the only `StorePort` implementation this
    project has) and reuse its own row-mapping, so there is exactly one place
    that knows the row shape. Any other `StorePort` implementation just won't
    get a detailed report -- `summary()` counts still work everywhere.
    """
    conn = getattr(store, "_conn", None)
    row_to_record = getattr(type(store), "_row_to_record", None)
    if conn is None or row_to_record is None:
        return []
    rows = conn.execute("SELECT * FROM assets ORDER BY id").fetchall()
    return [row_to_record(r) for r in rows]


def _outcome_from_history(prompt_id: str, entry: dict[str, Any]) -> JobOutcome:
    """Minimal, adapter-agnostic reading of a `/history` entry -- the same
    wire shape every `ComfyPort` implementation deals in. Deliberately not
    imported from `ComfyHTTP` (that parsing is a private method there);
    duplicating ~10 lines here is cheaper than reaching into another
    adapter's internals for something this small."""
    status = entry.get("status") or {}
    outputs = entry.get("outputs") or {}
    paths: list[str] = []
    for node_output in outputs.values():
        for image in node_output.get("images", []) or []:
            subfolder = image.get("subfolder") or ""
            filename = image.get("filename", "")
            paths.append(f"{subfolder}/{filename}" if subfolder else filename)
    if status.get("status_str") == "success":
        return JobOutcome(prompt_id=prompt_id, succeeded=True, output_paths=paths)
    return JobOutcome(
        prompt_id=prompt_id,
        succeeded=False,
        output_paths=paths,
        error=ExecutionError(prompt_id=prompt_id, exception_message="execution failed (from history)"),
    )


def _apply_qc_fix(
    graph: dict[str, Any],
    fix: FixProposal,
    *,
    asset: Asset,
    job_key: str,
    seed: int | None,
    rng: random.Random,
) -> tuple[dict[str, Any], int | None]:
    """Apply a judge's (already clamped/validated) `FixProposal` to a graph.

    `patch_graph` (comfy_http.py, frozen) only understands seed/prompt_text/
    filename_prefix -- it has no notion of steps/strength/named prompt slots.
    Rather than extend a frozen file, the extra mutations are applied here,
    directly on the deep copy `patch_graph` already returns (so the cached
    original graph is never touched either way).
    """
    prompt_text = asset.inputs.get("prompt") if isinstance(asset.inputs, dict) else None
    new_seed = rng.getrandbits(32) if fix.action == "reseed" else seed
    patched = patch_graph(graph, seed=new_seed, prompt_text=prompt_text, filename_prefix=job_key)

    if fix.action in ("adjust_steps", "adjust_strength") and fix.delta is not None:
        node = next(
            (n for n in patched.values() if isinstance(n, dict) and n.get("class_type") == _KSAMPLER_CLASS_TYPE),
            None,
        )
        if node is not None:
            inputs = node.setdefault("inputs", {})
            if fix.action == "adjust_steps":
                current = inputs.get("steps", 20)
                # Floor AND ceiling. The judge's own clamp is config-contingent
                # (see _MAX_SAMPLER_STEPS); this one is not.
                inputs["steps"] = max(1, min(_MAX_SAMPLER_STEPS, int(round(current + fix.delta))))
            else:
                current = inputs.get("denoise", 1.0)
                inputs["denoise"] = max(0.0, min(1.0, current + fix.delta))

    if fix.action == "rewrite_slot" and fix.slot and fix.text:
        slot_nodes = asset.inputs.get("slot_nodes", {}) if isinstance(asset.inputs, dict) else {}
        node_id = slot_nodes.get(fix.slot)
        node = patched.get(node_id) if node_id else None
        if not isinstance(node, dict):
            log.warning("qc_fix.rewrite_slot_unmapped", asset_id=asset.id, slot=fix.slot)
        else:
            addition = sanitize_slot_text(fix.text)
            if addition is None:
                log.warning(
                    "qc_fix.rewrite_slot_rejected",
                    asset_id=asset.id,
                    slot=fix.slot,
                    reason="text contains characters not permitted in a design phrase",
                )
            else:
                inputs = node.setdefault("inputs", {})
                original = inputs.get("text")
                inputs["text"] = compose_slot_text(original if isinstance(original, str) else "", addition)

    return patched, new_seed


# ------------------------------------------------------------------ the runner


class BatchRunner:
    def __init__(
        self,
        *,
        comfy: ComfyPort,
        store: StorePort,
        judge: JudgePort,
        supervisor: SupervisorPort,
        notifier: NotifierPort,
        config: RunnerConfig | None = None,
        breaker: CircuitBreaker | None = None,
        global_breaker: GlobalBreaker | None = None,
        retry_policy: RetryPolicy | None = None,
        classifier: ErrorClassifier | None = None,
        seq_planner: SequenceRepairPlanner | None = None,
        frame_consistency: FrameConsistencyChecker | None = None,
        judge_cost_usd_fn: Callable[[], float] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        self.comfy = comfy
        self.store = store
        self.judge = judge
        self.supervisor = supervisor
        self.notifier = notifier
        self.config = config or RunnerConfig()

        self.breaker = breaker or CircuitBreaker(now=clock)
        self.global_breaker = global_breaker or GlobalBreaker(now=clock)
        self.retry_policy = retry_policy or RetryPolicy()
        self.classifier = classifier or DEFAULT_CLASSIFIER
        self.seq_planner = seq_planner or SequenceRepairPlanner()
        # Optional: JudgePort.judge() (frozen) only ever sees one frame at a
        # time, so nothing catches "the product changed between frame 4 and
        # 5" without this. None (the default) preserves prior behaviour
        # exactly -- wire one up (adapters/frame_consistency.py) to enable it.
        self.frame_consistency = frame_consistency
        self._judge_cost_usd_fn = judge_cost_usd_fn or (lambda: 0.0)

        self._clock = clock
        self._sleep = sleep
        self._rng = rng or random.Random()

        self._graph_cache: dict[str, tuple[dict[str, Any], str]] = {}
        # Negative cache: a workflow on a shared drive that fails to parse
        # fails for every asset pointing at it, and re-reading a corrupt file
        # once per asset is pure cost.
        self._graph_cache_errors: dict[str, str] = {}
        self._graph_cache_lock = threading.Lock()

        self._stop_event = asyncio.Event()
        self._stats_lock = asyncio.Lock()
        self._total = 0
        self._done_count = 0
        self._milestones_hit: set[float] = set()
        self._jobs_since_start = 0
        self._gpu_seconds = 0.0
        self._start_time = 0.0
        self._last_progress_at = 0.0
        self._stall_alerted = False

        # Heartbeat (see RunnerConfig.heartbeat_file). `-inf` (not `0.0`: an
        # injected test clock may legitimately start at 0.0, which would
        # make the first call's `now - last < interval` check true and skip
        # the write) so the very first worker-loop iteration always writes
        # immediately rather than waiting a full `heartbeat_interval_s`.
        self._last_heartbeat_write_at = float("-inf")
        self._heartbeat_iteration = 0
        self._heartbeat_lock = asyncio.Lock()

    # ------------------------------------------------------------------ run()

    async def run(self, assets: list[Asset] | None = None) -> None:
        if assets:
            await self.store.load_batch(assets)

        await self._reconcile_startup()

        summary = await self.store.summary()
        self._total = sum(summary.values())
        self._done_count = sum(c for s, c in summary.items() if s.is_terminal)
        self._milestones_hit = {
            f for f in self.config.milestone_fractions if self._total and self._done_count / self._total >= f
        }
        self._start_time = self._clock()
        self._last_progress_at = self._start_time
        self._stall_alerted = False
        self._jobs_since_start = 0
        self._gpu_seconds = 0.0
        self._stop_event = asyncio.Event()

        await self._safe_notify(self.notifier.batch_started, self._total)

        sup_task = asyncio.create_task(self._supervisor_loop())
        workers = [
            asyncio.create_task(self._worker_loop(i)) for i in range(max(1, self.config.concurrency))
        ]
        try:
            # Plain `gather` leaves the surviving workers running detached
            # when one raises, so a single worker failure would orphan its
            # siblings mid-claim -- rows stuck CLAIMED with nobody finishing
            # them, and a coroutine still writing to a store the caller is
            # about to close. Stop and drain them all before propagating.
            await asyncio.gather(*workers)
        except BaseException:
            self._stop_event.set()
            for worker in workers:
                worker.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            raise
        finally:
            self._stop_event.set()
            sup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sup_task

        final_summary = await self.store.summary()
        records = read_all_records(self.store)
        await self._safe_notify(self.notifier.morning_report, final_summary, records)
        # `run()` returning means every worker loop has exited, so nothing
        # calls `_maybe_touch_heartbeat` again -- the heartbeat file is left
        # exactly where it was at the last touch, and will (correctly) start
        # aging from here. That is deliberately NOT how a normal finished
        # batch tears the box down: `ops/on_controller_stop.sh`
        # (deploy/comfy-controller.service's `ExecStopPost=`) fires the
        # instant this process exits cleanly and terminates the instance
        # right away via `ops/deadman.sh terminate`, rather than making a
        # successful overnight run sit idle-and-billing until
        # `DEADMAN_HEARTBEAT_STALE_MIN` elapses. The heartbeat's staleness
        # timer exists for the *wedged-but-still-running* case; a clean exit
        # is handled deliberately, not left for that timer to eventually
        # notice.

    # -------------------------------------------------------------- workers

    async def _worker_loop(self, worker_id: int) -> None:
        idle_rounds = 0
        while not self._stop_event.is_set():
            # The primary heartbeat checkpoint: every worker's own control
            # loop, whether it's about to claim fresh work or is idle-polling
            # because there's none yet. See `_maybe_touch_heartbeat`'s own
            # comment for why this lives here and not in a detached task.
            await self._maybe_touch_heartbeat()

            reason = await self._budget_exceeded()
            if reason:
                await self._safe_notify(self.notifier.alert, "Budget exceeded", reason)
                self._stop_event.set()
                return
            if self.global_breaker.is_tripped():
                await self._safe_notify(
                    self.notifier.alert, "Batch stopped", self.global_breaker.reason or "global breaker tripped"
                )
                self._stop_event.set()
                return

            record = await self.store.claim_next(job_key_fn=self._job_key_fn)
            if record is None:
                summary = await self.store.summary()
                outstanding = sum(c for s, c in summary.items() if not s.is_terminal)
                if outstanding == 0:
                    return
                idle_rounds += 1
                await self._sleep(min(0.5 * idle_rounds, self.config.idle_poll_max_s))
                continue
            idle_rounds = 0

            try:
                if record.asset.kind is AssetKind.SEQUENCE:
                    await self._process_sequence(record)
                else:
                    await self._process_single(record)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad asset must never kill a worker
                log.exception("worker.unhandled_exception", asset_id=record.asset.id, worker=worker_id)
                with contextlib.suppress(Exception):
                    terminal = (
                        AssetState.PARKED_APPROVAL
                        if record.asset.kind is AssetKind.SEQUENCE
                        else AssetState.SKIPPED_REVIEW
                    )
                    await self.store.transition(
                        record.asset.id,
                        terminal,
                        error=ExecutionError(
                            prompt_id=record.comfy_prompt_id or "",
                            exception_type=type(exc).__name__,
                            exception_message=str(exc),
                        ),
                    )
                await self._mark_progress_and_check_milestones()

            async with self._stats_lock:
                self._jobs_since_start += 1

    # --------------------------------------------------------------- job_key

    def _job_key_fn(self, asset: Asset, attempt: int) -> str:
        """Deterministic filename_prefix, including the attempt suffix so a
        retry never collides with a previous attempt's output on disk.

        This runs INSIDE `claim_next`'s transaction, so it must not raise: a
        `WorkflowLoadError` here would roll the claim back and propagate out
        of the worker, aborting the whole batch over one corrupt file. Instead
        it returns a key derived without the graph hash; `_process_single` /
        `_process_sequence` hit the same load failure a moment later, outside
        the transaction, where it can be recorded as a per-asset
        PERMANENT_ERROR.
        """
        try:
            _, ghash = self._load_graph(asset.workflow)
        except WorkflowLoadError:
            return f"{asset.job_key(self.config.model_version, 'unloadable')}_a{attempt}"
        base = asset.job_key(self.config.model_version, ghash)
        return f"{base}_a{attempt}"

    def _load_graph(self, workflow_path: str) -> tuple[dict[str, Any], str]:
        """Read + parse a workflow, caching both successes and failures.

        Raises `WorkflowLoadError` for anything wrong with the file -- missing,
        unreadable, not JSON, or JSON that isn't an object. Callers turn that
        into a terminal state for the one asset involved.
        """
        with self._graph_cache_lock:
            cached = self._graph_cache.get(workflow_path)
            if cached is not None:
                return cached
            prior_error = self._graph_cache_errors.get(workflow_path)
            if prior_error is not None:
                raise WorkflowLoadError(prior_error)
            try:
                graph = json.loads(Path(workflow_path).read_text())
            except (OSError, ValueError) as exc:
                message = f"could not load workflow {workflow_path!r}: {type(exc).__name__}: {exc}"
                self._graph_cache_errors[workflow_path] = message
                raise WorkflowLoadError(message) from exc
            if not isinstance(graph, dict):
                message = (
                    f"could not load workflow {workflow_path!r}: expected an API-format "
                    f"object, got {type(graph).__name__}"
                )
                self._graph_cache_errors[workflow_path] = message
                raise WorkflowLoadError(message)
            ghash = graph_hash(graph)
            self._graph_cache[workflow_path] = (graph, ghash)
            return graph, ghash

    async def _finish_unloadable_workflow(self, asset: Asset, exc: WorkflowLoadError) -> None:
        log.error("workflow.load_failed", asset_id=asset.id, workflow=asset.workflow, error=str(exc))
        self.breaker.record_failure(asset.workflow, ErrorClass.PERMANENT)
        await self.store.transition(
            asset.id,
            AssetState.PERMANENT_ERROR,
            error=ExecutionError(
                prompt_id="", exception_type="WorkflowLoadError", exception_message=str(exc)
            ),
        )
        await self._safe_notify(
            self.notifier.alert, f"{asset.workflow}: workflow unreadable", f"{asset.id}: {exc}"
        )
        await self._mark_progress_and_check_milestones()

    # --------------------------------------------------------- single assets

    async def _process_single(self, record: AssetRecord) -> None:
        asset = record.asset
        partition = asset.workflow
        await self._await_breaker(partition)

        try:
            graph, _ = self._load_graph(asset.workflow)
        except WorkflowLoadError as exc:
            await self._finish_unloadable_workflow(asset, exc)
            return

        seed = asset.seed
        fix = FixProposal(action="none")
        qc_round = 0

        while True:
            # The QC-repair loop submits a fresh generation per round. Without
            # this check the only budget/stop gate is between whole assets
            # (_worker_loop), so a stop requested mid-asset is ignored for as
            # many rounds as the retry policy allows.
            halt = await self._halt_reason()
            if halt is not None:
                await self._yield_claim_for_halt(asset, halt, where="qc_retry_loop", detail=qc_round)
                return

            job_key = record.job_key if qc_round == 0 else f"{record.job_key}_qc{qc_round}"
            patched, seed = _apply_qc_fix(graph, fix, asset=asset, job_key=job_key, seed=seed, rng=self._rng)

            t0 = self._clock()
            try:
                prompt_id = await self.comfy.submit(patched, job_key)
            except ComfyValidationError as exc:
                await self._finish_unfixable_submit_error(asset, exc)
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
                await self._handle_submit_exception(record, exc, partition)
                return

            await self.store.record_submission(asset.id, prompt_id)
            try:
                outcome = await self.comfy.await_outcome(prompt_id, timeout=self.config.await_outcome_timeout_s)
            except asyncio.TimeoutError:
                outcome = JobOutcome(
                    prompt_id=prompt_id,
                    succeeded=False,
                    error=ExecutionError(
                        prompt_id=prompt_id, exception_type="TimeoutError", exception_message="await_outcome timed out"
                    ),
                )
            async with self._stats_lock:
                self._gpu_seconds += self._clock() - t0
            # Checkpoint: the submit/poll round-trip just completed, so this
            # worker is demonstrably not wedged, even mid-asset on a single
            # very long QC-repair round. See _maybe_touch_heartbeat.
            await self._maybe_touch_heartbeat()

            if not outcome.succeeded:
                await self._handle_execution_failure(record, outcome, partition)
                return

            self.breaker.record_success(partition)

            verdict = await self._judge_or_park(asset, outcome.output_paths, prompt_id=prompt_id)
            if verdict is None:
                await self._mark_progress_and_check_milestones()
                return

            if verdict.passed:
                await self.store.transition(
                    asset.id, AssetState.SAVED, verdict=verdict, output_paths=outcome.output_paths
                )
                await self._mark_progress_and_check_milestones()
                return

            if "_low_confidence" in verdict.failed_checks or verdict.fix.action == "none":
                # Judge couldn't agree with itself, or failed with nothing
                # fixable to try -- either way this needs a human, not a
                # bounded auto-retry.
                await self.store.transition(
                    asset.id, AssetState.PARKED_APPROVAL, verdict=verdict, output_paths=outcome.output_paths
                )
                await self._mark_progress_and_check_milestones()
                return

            decision = self.retry_policy.qc_retry_decision(qc_round, asset.kind)
            if not decision.should_retry:
                await self.store.transition(
                    asset.id, decision.terminal_state, verdict=verdict, output_paths=outcome.output_paths
                )
                await self._mark_progress_and_check_milestones()
                return

            fix = verdict.fix
            qc_round += 1

    async def _handle_execution_failure(self, record: AssetRecord, outcome: JobOutcome, partition: str) -> None:
        asset = record.asset
        error = outcome.error or ExecutionError(prompt_id=outcome.prompt_id, exception_message="unknown failure")
        cls = self.classifier.classify(error)
        self.breaker.record_failure(partition, cls)

        decision = self.retry_policy.next_action(cls, attempt=max(record.attempt - 1, 0), kind=asset.kind)
        if decision.should_retry:
            await self.store.transition(
                asset.id, AssetState.RETRY_WAIT, error=error, retry_after_s=decision.delay_s
            )
            await self._safe_notify(
                self.notifier.alert, f"{asset.workflow}: {cls.value} failure (retrying)", f"{asset.id}: {error.exception_message}"
            )
        else:
            assert decision.terminal_state is not None
            await self.store.transition(asset.id, decision.terminal_state, error=error)
            await self._safe_notify(
                self.notifier.alert, f"{asset.workflow}: {cls.value} failure ({decision.terminal_state.value})",
                f"{asset.id}: {error.exception_message}",
            )
            await self._mark_progress_and_check_milestones()

    async def _handle_submit_exception(self, record: AssetRecord, exc: Exception, partition: str) -> None:
        """Route a non-validation submit failure through the same
        classifier/RetryPolicy path an execution failure takes.

        Only `ComfyValidationError` means "this graph is wrong"; everything
        else -- a refused connection, a read timeout, a 5xx after the HTTP
        adapter's own short retry budget -- is about the server, not the
        asset. Submitting into a supervisor restart is the ordinary case:
        `SupervisorConfig.backoff_cap` is 300s against a ~10s submit budget,
        so without this a perfectly good asset was skipped permanently.
        """
        error = ExecutionError(
            prompt_id="", exception_type=type(exc).__name__, exception_message=str(exc)
        )
        log.warning(
            "submit.failed",
            asset_id=record.asset.id,
            workflow=record.asset.workflow,
            exception_type=type(exc).__name__,
            error=str(exc),
        )
        outcome = JobOutcome(prompt_id="", succeeded=False, error=error)
        await self._handle_execution_failure(record, outcome, partition)

    async def _finish_unfixable_submit_error(self, asset: Asset, exc: Exception) -> None:
        error = ExecutionError(prompt_id="", exception_type=type(exc).__name__, exception_message=str(exc))
        self.breaker.record_failure(asset.workflow, ErrorClass.PERMANENT)
        terminal = AssetState.PARKED_APPROVAL if asset.kind is AssetKind.SEQUENCE else AssetState.PERMANENT_ERROR
        await self.store.transition(asset.id, terminal, error=error)
        await self._safe_notify(self.notifier.alert, f"{asset.workflow}: graph rejected", f"{asset.id}: {exc}")
        await self._mark_progress_and_check_milestones()

    def resolve_output_path(self, output_path: str) -> Path:
        """Turn one ComfyUI-reported output path into a path on this machine.

        ComfyUI reports outputs as `subfolder/filename` RELATIVE TO ITS OWN
        OUTPUT DIRECTORY (see `ComfyHTTP._extract_output_paths`). Handing that
        string straight to `Path()` resolves it against the *controller's* CWD
        instead, where nothing exists -- which made `prefilter.evaluate`'s
        `Image.open` raise `FileNotFoundError` for every asset, an error class
        no caller expects from a judge.
        """
        path = Path(output_path)
        base = self.config.comfy_output_dir
        if base is not None and not path.is_absolute():
            path = base / path
        return path

    def resolve_output_paths(self, output_paths: list[str]) -> list[Path]:
        return [self.resolve_output_path(p) for p in output_paths]

    def _unreadable_outputs(self, paths: list[Path]) -> list[str]:
        """Which of `paths` a judge could not possibly open, and why.

        Only enforced once `comfy_output_dir` is configured -- without it the
        controller has no way to resolve a relative path to real bytes and has
        no business asserting anything about the filesystem.
        """
        if self.config.comfy_output_dir is None:
            return []
        problems: list[str] = []
        if not paths:
            return ["job reported success but produced no output files"]
        for path in paths:
            try:
                stat = path.stat()
            except OSError as exc:
                problems.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            if stat.st_size == 0:
                problems.append(f"{path}: 0-byte file")
        return problems

    async def _fail_unreadable_output(
        self, asset: Asset, output_paths: list[str], problems: list[str], *, prompt_id: str
    ) -> None:
        detail = "; ".join(problems)
        # Distinct from every other failure: the generation reported success,
        # so this is neither a rubric failure nor a judge-infrastructure
        # failure. Letting it reach the worker's blanket `except Exception`
        # instead buried it as an unexplained SKIPPED_REVIEW.
        log.error(
            "judge.output_unreadable",
            asset_id=asset.id,
            prompt_id=prompt_id,
            output_paths=output_paths,
            problems=problems,
        )
        terminal = (
            AssetState.PARKED_APPROVAL if asset.kind is AssetKind.SEQUENCE else AssetState.SKIPPED_REVIEW
        )
        await self.store.transition(
            asset.id,
            terminal,
            error=ExecutionError(
                prompt_id=prompt_id,
                exception_type="OutputUnreadable",
                exception_message=f"output file(s) missing or unreadable: {detail}",
            ),
            output_paths=output_paths,
        )
        await self._safe_notify(self.notifier.alert, "Output unreadable", f"{asset.id}: {detail}")

    async def _judge_or_park(self, asset: Asset, output_paths: list[str], *, prompt_id: str) -> QCVerdict | None:
        resolved = self.resolve_output_paths(output_paths)

        problems = self._unreadable_outputs(resolved)
        if problems:
            await self._fail_unreadable_output(asset, output_paths, problems, prompt_id=prompt_id)
            return None

        try:
            return await self.judge.judge(asset, resolved)
        except JudgeUnavailable as exc:
            # Never fail open into auto-approval, and never treat this as the
            # asset having failed the rubric -- it's an infra problem.
            await self.store.transition(
                asset.id,
                AssetState.JUDGE_UNAVAILABLE,
                error=ExecutionError(prompt_id=prompt_id, exception_type="JudgeUnavailable", exception_message=str(exc)),
                output_paths=output_paths,
            )
            await self._safe_notify(self.notifier.alert, "Judge unavailable", f"{asset.id}: {exc}")
            return None
        except OSError as exc:
            # The judge got as far as opening the file and the filesystem said
            # no (deleted under us, permission denied, truncated mid-read).
            # Same explicit, distinctly-logged outcome as the pre-check above,
            # never an anonymous escape to the worker's blanket handler.
            await self._fail_unreadable_output(
                asset, output_paths, [f"{type(exc).__name__}: {exc}"], prompt_id=prompt_id
            )
            return None

    # -------------------------------------------------------------- sequences

    async def _process_sequence(self, record: AssetRecord) -> None:
        """Generate every frame, then run `SequenceRepairPlanner` rounds until
        the sequence is fixed or parked.

        `QCCompare` (planner.py) asks to compare a regenerated frame against
        its stable neighbours. `JudgePort.judge()` (frozen) takes one asset
        plus its own image list -- it has no notion of "compare to a
        reference frame" -- so a `QCCompare` step is implemented as
        "re-judge this frame now that its neighbour has stabilised" via the
        same per-frame call, rather than a true pairwise diff. That is a
        deliberate simplification forced by the frozen `JudgePort` signature.
        """
        asset = record.asset
        partition = asset.workflow
        frame_count = asset.frame_count or 0
        if frame_count <= 0:
            await self.store.transition(asset.id, AssetState.PERMANENT_ERROR,
                                         error=ExecutionError(prompt_id="", exception_message="sequence with frame_count<=0"))
            await self._mark_progress_and_check_milestones()
            return

        await self._await_breaker(partition)
        try:
            graph, _ = self._load_graph(asset.workflow)
        except WorkflowLoadError as exc:
            await self._finish_unloadable_workflow(asset, exc)
            return

        frame_paths: dict[int, str] = {}
        attempt_counts: dict[int, int] = {}

        async def generate_frame(i: int) -> bool:
            """Returns True on success; on a hard execution failure, finalises
            the whole sequence's terminal state and returns False."""
            job_key = f"{record.job_key}_f{i}_{attempt_counts.get(i, 0)}"
            seed = (asset.seed or 0) + i
            fix = FixProposal(action="none")
            patched, _ = _apply_qc_fix(graph, fix, asset=asset, job_key=job_key, seed=seed, rng=self._rng)
            try:
                prompt_id = await self.comfy.submit(patched, job_key)
            except ComfyValidationError as exc:
                await self._finish_unfixable_submit_error(asset, exc)
                return False
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified below, never swallowed
                # Same reasoning as _handle_submit_exception: only a
                # validation error is the asset's fault. A connection failure
                # here must go through the RetryPolicy ladder, not skip an
                # otherwise-good sequence.
                error = ExecutionError(
                    prompt_id="", exception_type=type(exc).__name__, exception_message=str(exc)
                )
                log.warning(
                    "submit.failed",
                    asset_id=asset.id,
                    workflow=asset.workflow,
                    frame_index=i,
                    exception_type=type(exc).__name__,
                    error=str(exc),
                )
                return await handle_frame_failure(i, error)

            await self.store.record_submission(asset.id, prompt_id)
            outcome = await self.comfy.await_outcome(prompt_id, timeout=self.config.await_outcome_timeout_s)
            # Same checkpoint as _process_single: one frame's submit/poll
            # round-trip just completed, so this worker is still alive even
            # mid-sequence. See _maybe_touch_heartbeat.
            await self._maybe_touch_heartbeat()
            if not outcome.succeeded:
                error = outcome.error or ExecutionError(prompt_id=prompt_id, exception_message="unknown failure")
                return await handle_frame_failure(i, error)
            self.breaker.record_success(partition)
            frame_paths[i] = outcome.output_paths[0] if outcome.output_paths else ""
            return True

        async def handle_frame_failure(i: int, error: ExecutionError) -> bool:
            cls = self.classifier.classify(error)
            self.breaker.record_failure(partition, cls)
            decision = self.retry_policy.next_action(
                cls, attempt=attempt_counts.get(i, 0), kind=AssetKind.SEQUENCE
            )
            if decision.should_retry:
                attempt_counts[i] = attempt_counts.get(i, 0) + 1
                return await generate_frame(i)
            await self.store.transition(asset.id, AssetState.PARKED_APPROVAL, error=error)
            await self._mark_progress_and_check_milestones()
            return False

        for i in range(frame_count):
            # A manifest row is free to ask for thousands of frames; without
            # this the budget and the stop event are not consulted again until
            # every last one has been generated.
            halt = await self._halt_reason()
            if halt is not None:
                await self._yield_claim_for_halt(asset, halt, where="sequence_frame_loop", detail=i)
                return
            if not await generate_frame(i):
                return

        failed_frames = set()
        for i in range(frame_count):
            verdict = await self._judge_or_park(asset, [frame_paths[i]], prompt_id="")
            if verdict is None:
                return
            if not verdict.passed:
                failed_frames.add(i)

        if self.frame_consistency is not None and frame_count >= 2:
            failed_frames |= self._check_sequence_consistency(asset, frame_paths, frame_count)

        while True:
            halt = await self._halt_reason()
            if halt is not None:
                await self._yield_claim_for_halt(asset, halt, where="sequence_repair_loop")
                return

            plan = self.seq_planner.plan(frame_count, failed_frames, attempt_counts)
            if plan.outcome == "fixed":
                await self.store.transition(
                    asset.id, AssetState.SAVED, output_paths=[frame_paths[i] for i in range(frame_count)]
                )
                await self._mark_progress_and_check_milestones()
                return
            if plan.outcome == "parked":
                await self.store.transition(
                    asset.id, AssetState.PARKED_APPROVAL, output_paths=[frame_paths[i] for i in range(frame_count)]
                )
                await self._mark_progress_and_check_milestones()
                return

            newly_failed: set[int] = set()
            for step in plan.steps:
                if isinstance(step, RegenerateFrame):
                    attempt_counts[step.frame_index] = attempt_counts.get(step.frame_index, 0) + 1
                    if not await generate_frame(step.frame_index):
                        return
                elif isinstance(step, QCCompare):
                    verdict = await self._judge_or_park(asset, [frame_paths[step.frame_index]], prompt_id="")
                    if verdict is None:
                        return
                    if not verdict.passed:
                        newly_failed.add(step.frame_index)
                    elif self.frame_consistency is not None:
                        # Design doc: "QC the new frame N against both N-1 and
                        # N+1" -- `_judge_or_park` above only re-scores frame N
                        # in isolation (JudgePort.judge() is frozen and has no
                        # notion of a reference frame), so it can't see the
                        # frame having drifted from its now-stable neighbours.
                        # `step.against` is exactly that neighbour set.
                        for neighbor in step.against:
                            pair = self.frame_consistency.check_pair(
                                self.resolve_output_path(frame_paths[step.frame_index]),
                                self.resolve_output_path(frame_paths[neighbor]),
                                index_a=step.frame_index,
                                index_b=neighbor,
                            )
                            if not pair.passed:
                                log.info(
                                    "sequence.qc_compare_consistency_failed",
                                    asset_id=asset.id,
                                    frame_index=step.frame_index,
                                    neighbor=neighbor,
                                    tripped=pair.tripped_signals,
                                )
                                newly_failed.add(step.frame_index)
            failed_frames = newly_failed

    def _check_sequence_consistency(
        self, asset: Asset, frame_paths: dict[int, str], frame_count: int
    ) -> set[int]:
        """Design doc: sequences must be checked for "consistency across
        frames (product, colours, no flicker)". `JudgePort.judge()` only ever
        sees one frame at a time, so this is the only place that catches e.g.
        "the product changed between frame 4 and 5" on the very first pass,
        before any repair round has even started. Runs the cheap, VLM-free
        `FrameConsistencyChecker` (adapters/frame_consistency.py) across every
        adjacent pair plus whole-sequence colour/brightness drift.
        """
        report = self.frame_consistency.check_sequence(
            [self.resolve_output_path(frame_paths[i]) for i in range(frame_count)]
        )
        failed: set[int] = set()
        for pair in report.pairs:
            if not pair.passed:
                # The later frame is the one that broke continuity with its
                # (assumed-good) predecessor, so it's what the repair loop
                # should regenerate -- matching "regenerate frame N using N-1
                # as reference".
                failed.add(pair.frame_b)
                log.info(
                    "sequence.consistency_failed",
                    asset_id=asset.id,
                    frame_a=pair.frame_a,
                    frame_b=pair.frame_b,
                    tripped=pair.tripped_signals,
                    changed_regions=pair.changed_regions,
                )
        if report.brightness_drift.tripped or report.color_drift.tripped:
            # Cumulative drift isn't pinned to one pair -- the last frame is
            # furthest from the sequence's own baseline, so it's the natural
            # one to regenerate first.
            failed.add(frame_count - 1)
            log.info(
                "sequence.cumulative_drift_flagged",
                asset_id=asset.id,
                brightness_drift=report.brightness_drift.value,
                color_drift=report.color_drift.value,
            )
        return failed

    # --------------------------------------------------------- reconciliation

    async def _reconcile_startup(self) -> None:
        pending = await self.store.needing_reconciliation()
        for rec in pending:
            try:
                await self._reconcile_one(rec)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad row must never abort the resume
                # `run()` has not started a single worker yet, so anything
                # raised here takes the WHOLE batch down before it begins:
                # `asyncio.run` propagates it out of `cli._run_batch`, the
                # process exits non-zero, no asset is reconciled, no morning
                # report is sent -- and the row stays `reconciling=1`, so the
                # next resume wedges in exactly the same place. For an
                # unattended overnight batch that turns one unreachable
                # ComfyUI into a dead night. Requeue the row instead and let
                # the ordinary claim/retry path deal with it; `crash_count`
                # still bounds how many times this can repeat.
                log.exception("reconciliation.failed", asset_id=rec.asset.id, error=str(exc))
                with contextlib.suppress(Exception):
                    await self.store.transition(
                        rec.asset.id, AssetState.RETRY_WAIT, retry_after_s=0.0,
                        error=ExecutionError(
                            prompt_id=rec.comfy_prompt_id or "",
                            exception_type=type(exc).__name__,
                            exception_message=str(exc),
                        ),
                    )

    async def _probe_history(self, prompt_id: str) -> dict[str, Any] | None:
        """`comfy.history()` for reconciliation, where an unreachable server
        is an ordinary condition rather than an error.

        Resume runs precisely when something went wrong, which is exactly
        when ComfyUI is most likely to be down, restarting, or still loading.
        The adapter already retries transient failures on a bounded budget;
        once that is spent, "I could not ask" is not "it did not happen", so
        this reports no answer and lets the caller fall through to the
        on-disk check and, failing that, the never-guess requeue.
        """
        try:
            return await self.comfy.history(prompt_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "reconciliation.history_probe_failed",
                prompt_id=prompt_id,
                exception_type=type(exc).__name__,
                error=str(exc),
            )
            return None

    async def _reconcile_one(self, rec: AssetRecord) -> None:
        asset = rec.asset
        if rec.crash_count >= 2:
            log.warning("reconciliation.giving_up_after_crashes", asset_id=asset.id, crash_count=rec.crash_count)
            terminal = AssetState.PARKED_APPROVAL if asset.kind is AssetKind.SEQUENCE else AssetState.SKIPPED_REVIEW
            await self.store.transition(
                asset.id, terminal,
                error=ExecutionError(prompt_id=rec.comfy_prompt_id or "", exception_type="ReconciliationExhausted",
                                      exception_message=f"crashed {rec.crash_count} times; giving up"),
            )
            return

        if asset.kind is AssetKind.SEQUENCE:
            # Frames depend on their predecessor -- partial progress can't be
            # trusted, so the whole sequence reruns from scratch.
            await self.store.transition(asset.id, AssetState.RETRY_WAIT, retry_after_s=0.0)
            return

        found: JobOutcome | None = None
        if rec.comfy_prompt_id:
            entry = await self._probe_history(rec.comfy_prompt_id)
            if entry is not None:
                found = _outcome_from_history(rec.comfy_prompt_id, entry)

        if found is None and rec.job_key and self.config.comfy_output_dir is not None:
            matches = self._reconcile_outputs_on_disk(rec.job_key, self.config.comfy_output_dir)
            if matches:
                found = JobOutcome(
                    prompt_id=rec.comfy_prompt_id or "",
                    succeeded=True,
                    output_paths=[str(p) for p in matches],
                )

        if found is None:
            # Genuinely ambiguous -- never guess. Requeue for a fresh attempt.
            await self.store.transition(asset.id, AssetState.RETRY_WAIT, retry_after_s=0.0)
            return

        if found.succeeded:
            verdict = await self._judge_or_park(asset, found.output_paths, prompt_id=found.prompt_id)
            if verdict is None:
                return
            if verdict.passed:
                await self.store.transition(asset.id, AssetState.SAVED, verdict=verdict, output_paths=found.output_paths)
            else:
                await self.store.transition(asset.id, AssetState.RETRY_WAIT, retry_after_s=0.0, verdict=verdict)
        else:
            error = found.error or ExecutionError(prompt_id=found.prompt_id, exception_message="unknown failure")
            cls = self.classifier.classify(error)
            decision = self.retry_policy.next_action(cls, attempt=max(rec.attempt - 1, 0), kind=asset.kind)
            if decision.should_retry:
                await self.store.transition(asset.id, AssetState.RETRY_WAIT, error=error, retry_after_s=decision.delay_s)
            else:
                assert decision.terminal_state is not None
                await self.store.transition(asset.id, decision.terminal_state, error=error)

    def _reconcile_outputs_on_disk(self, job_key: str, out_dir: Path) -> list[Path]:
        """Find the outputs of the attempt that was actually in flight, or none.

        Three things the bare `glob(f"{job_key}*")` got wrong:

        * **Over-matching across rounds.** A QC-repair round writes under
          `{job_key}_qc{n}` and a sequence frame under `{job_key}_f{i}_{n}`,
          all of which the prefix glob happily returns. Accepting round 0's
          file as "the answer" for a crash that happened during round 2 hands
          the judge an image the judge already rejected.
        * **Existence is not success.** A `kill -9` partway through
          `SaveImage` leaves a 0-byte or truncated file on disk. The old code
          read that as a completed generation and judged it.
        * **Silence on ambiguity.** Finding later-round files proves a later
          round ran, which means nothing on disk can be trusted to represent
          the in-flight attempt. That has to requeue, not guess.
        """
        candidates = sorted(out_dir.glob(f"{job_key}*"))
        exact: list[Path] = []
        other_rounds: list[Path] = []
        for path in candidates:
            # What ComfyUI appended: `_00001_.png` for the attempt itself,
            # versus `_qc2_00001_.png` / `_f3_1_00001_.png` for a different
            # round or frame of the same claim.
            suffix = path.name[len(job_key):]
            if _ROUND_SUFFIX_RE.search(suffix):
                other_rounds.append(path)
            else:
                exact.append(path)

        if other_rounds:
            log.warning(
                "reconciliation.ambiguous_rounds_on_disk",
                job_key=job_key,
                later_round_files=[p.name for p in other_rounds][:10],
                detail="outputs from another QC round/frame exist; cannot tell which attempt was in flight",
            )
            return []

        intact: list[Path] = []
        for path in exact:
            problem = _image_integrity_problem(path)
            if problem is None:
                intact.append(path)
            else:
                log.warning("reconciliation.rejected_corrupt_output", job_key=job_key, path=str(path), problem=problem)

        if exact and not intact:
            # Every candidate was junk -- a mid-write kill. Requeueing is the
            # only honest answer; claiming success here would save a corrupt
            # asset.
            return []
        return intact

    # ------------------------------------------------------------- breaker

    async def _await_breaker(self, partition: str) -> None:
        while not self.breaker.allow(partition):
            if self._stop_event.is_set():
                raise asyncio.CancelledError()
            await self._sleep(1.0)

    # ---------------------------------------------------------- supervision

    async def _supervisor_loop(self) -> None:
        prior_restarts = getattr(self.supervisor, "restart_count", 0)
        while not self._stop_event.is_set():
            await self._sleep(self.config.tick_interval_s)
            if self._stop_event.is_set():
                return

            tick = getattr(self.supervisor, "tick", None)
            try:
                verdict = await tick() if tick is not None else await self.supervisor.assess()
            except Exception:
                log.exception("supervisor_tick_failed")
                continue

            current_restarts = getattr(self.supervisor, "restart_count", prior_restarts)
            if current_restarts > prior_restarts:
                self.global_breaker.record_crash()
                await self._safe_notify(
                    self.notifier.alert, "ComfyUI restarted",
                    f"verdict={getattr(verdict, 'value', verdict)} restart#{current_restarts}",
                )
                prior_restarts = current_restarts

            if getattr(self.supervisor, "held", False):
                await self._safe_notify(
                    self.notifier.alert, "Restart intensity ceiling hit",
                    "ComfyUI supervisor is holding; stopping the run for operator intervention.",
                )
                self._stop_event.set()
                return

            async with self._stats_lock:
                jobs = self._jobs_since_start
                stalled = (not self._stall_alerted) and (self._clock() - self._last_progress_at > self.config.stall_alert_s)
                if stalled:
                    self._stall_alerted = True
            if stalled:
                await self._safe_notify(
                    self.notifier.alert, "Batch appears stalled",
                    f"no asset has finished in over {self.config.stall_alert_s:.0f}s",
                )

            recycled = await self.supervisor.maybe_recycle(jobs)
            if recycled:
                await self._safe_notify(self.notifier.alert, "Proactive recycle", f"restarted after {jobs} jobs")

    # ----------------------------------------------------------- heartbeat

    async def _maybe_touch_heartbeat(self) -> None:
        """Refresh `RunnerConfig.heartbeat_file`'s mtime, throttled to at most
        once per `heartbeat_interval_s`.

        Called from the WORKER loop itself (`_worker_loop`'s own iteration,
        plus the two checkpoints inside `_process_single`/`_process_sequence`
        right after a `comfy.await_outcome` call resolves) rather than from a
        separate `asyncio.create_task`'d timer. That distinction is the whole
        point of this being a dead-man's-switch signal and not theatre: a
        detached task whose only job is "sleep `interval`; touch; repeat"
        keeps ticking on its own schedule even if the actual work -- the
        worker loops that claim and process assets -- has deadlocked or
        wedged on something that doesn't block the rest of the event loop
        (an `await` that never resolves, two coroutines stuck on each other).
        That would make the heartbeat lie about the one thing
        `ops/deadman.sh` needs it to tell the truth about. Piggy-backing the
        write on the workers' own forward progress means it can only look
        healthy when the thing it's certifying -- the batch actually
        advancing -- is itself still true.

        Every call is cheap when throttled (one clock read under a lock), so
        calling it from several places, including per QC-round/per-frame
        checkpoints for a single very long-running asset, is deliberate: with
        `concurrency=1` a single slow-but-healthy job would otherwise starve
        the heartbeat for its entire `await_outcome_timeout_s`, which on a
        long sequence is longer than a plausible `DEADMAN_HEARTBEAT_STALE_MIN`.
        """
        path = self.config.heartbeat_file
        if path is None:
            return
        now = self._clock()
        async with self._heartbeat_lock:
            if now - self._last_heartbeat_write_at < self.config.heartbeat_interval_s:
                return
            self._last_heartbeat_write_at = now
            self._heartbeat_iteration += 1
            iteration = self._heartbeat_iteration
        try:
            await asyncio.to_thread(self._write_heartbeat_sync, path, iteration)
        except OSError:
            # Best-effort: a failed heartbeat write must never take down the
            # batch. Worst case ops/deadman.sh's staleness check fires late
            # (still fail-safe, never fail-open) -- it never fires *never*,
            # because the file being unwritable is exactly as bad as the
            # controller being dead from the dead-man's switch's point of
            # view, and it will correctly go stale on its own.
            log.exception("heartbeat.write_failed", path=str(path))

    @staticmethod
    def _write_heartbeat_sync(path: Path, iteration: int) -> None:
        """Atomic temp-file + rename so `ops/deadman.sh`'s reader
        (`stat`/`os.path.getmtime` on `path`) never observes a half-written
        file -- only ever the previous complete one, or the new complete one.
        Content is a nice-to-have for a human running `cat`; only the
        resulting mtime is part of the actual contract (see
        docs/runbook.md's "Heartbeat contract").
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}-{threading.get_ident()}")
        tmp.write_text(json.dumps({"ts": time.time(), "iteration": iteration}))
        os.replace(tmp, path)

    # -------------------------------------------------------------- budget

    async def _halt_reason(self) -> str | None:
        """Why an in-progress asset should stop looping, or None to continue.

        `_worker_loop` only consults the budget and the stop event BETWEEN
        assets. Anything that loops WITHIN one asset -- QC-repair rounds, and
        every frame of a sequence -- has to consult them too, or a single
        manifest row can keep submitting long after the batch was told to
        stop or blew its budget.
        """
        if self._stop_event.is_set():
            return "stop requested"
        return await self._budget_exceeded()

    async def _yield_claim_for_halt(
        self, asset: Asset, reason: str, *, where: str, detail: Any = None
    ) -> None:
        """Give a claimed, partially-processed asset back as RETRY_WAIT.

        Not a terminal state: nothing is wrong with the asset, the run just
        ran out of budget or was asked to stop. RETRY_WAIT leaves it
        immediately claimable by the next run rather than stranded CLAIMED
        for startup reconciliation to puzzle over.
        """
        log.warning("runner.halted_mid_asset", asset_id=asset.id, reason=reason, where=where, detail=detail)
        with contextlib.suppress(Exception):
            await self.store.transition(
                asset.id,
                AssetState.RETRY_WAIT,
                retry_after_s=0.0,
                error=ExecutionError(
                    prompt_id="",
                    exception_type="BatchHalted",
                    exception_message=f"halted during {where}: {reason}",
                ),
            )
        self._stop_event.set()

    async def _budget_exceeded(self) -> str | None:
        b = self.config.budget
        async with self._stats_lock:
            elapsed = self._clock() - self._start_time
            gpu_seconds = self._gpu_seconds
        if b.max_wall_clock_s is not None and elapsed >= b.max_wall_clock_s:
            return f"wall-clock budget exceeded ({elapsed:.0f}s >= {b.max_wall_clock_s:.0f}s)"
        if b.max_gpu_seconds is not None and gpu_seconds >= b.max_gpu_seconds:
            return f"GPU-time budget exceeded ({gpu_seconds:.0f}s >= {b.max_gpu_seconds:.0f}s)"
        if b.max_judge_usd is not None:
            spent = self._judge_cost_usd_fn()
            if spent >= b.max_judge_usd:
                return f"judge cost budget exceeded (${spent:.2f} >= ${b.max_judge_usd:.2f})"
        return None

    # ------------------------------------------------------------ progress

    async def _mark_progress_and_check_milestones(self) -> None:
        async with self._stats_lock:
            self._done_count += 1
            self._last_progress_at = self._clock()
            self._stall_alerted = False
            done, total = self._done_count, self._total
            to_notify = []
            for frac in self.config.milestone_fractions:
                if frac not in self._milestones_hit and total > 0 and done / total >= frac:
                    self._milestones_hit.add(frac)
                    to_notify.append(frac)
        for _ in to_notify:
            await self._safe_notify(self.notifier.milestone, done, total)

    async def _safe_notify(self, fn: Callable[..., Any], *args: Any) -> None:
        try:
            await fn(*args)
        except Exception:
            log.exception("notifier_call_failed")
