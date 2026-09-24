"""End-to-end tests for BatchRunner against FakeComfy (via the real ComfyHTTP
adapter) + FakeJudge -- no network, no GPU. One test (`test_breaker_...`) uses
a small hand-rolled in-process ComfyPort stub instead of FakeComfy, because
FakeComfy's fault injection is a single global setting applied uniformly to
every submission -- it cannot make one workflow fail while another succeeds,
which is exactly what partition-isolation needs to prove.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Callable

import pytest

from comfy_controller.adapters.sqlite_store import SQLiteStore
from comfy_controller.adapters.comfy_http import ComfyHTTP
from comfy_controller.core.policy import CircuitBreaker
from comfy_controller.models import (
    Asset,
    AssetKind,
    AssetState,
    ExecutionError,
    FixProposal,
    JobOutcome,
)
from comfy_controller.runner import BatchRunner, BudgetLimits, RunnerConfig, read_all_records
from comfy_controller.supervisor import HealthSupervisor
from comfy_controller.testing.fake_comfy import Fault, FakeComfy, FakeComfyState
from comfy_controller.testing.fake_judge import FakeJudge


# --------------------------------------------------------------------- helpers


def _workflow_json(tmp_path: Path, name: str = "wf.json", *, tag: str | None = None) -> str:
    graph: dict[str, Any] = {
        "1": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20, "denoise": 1.0}},
        "9": {"class_type": "SaveImage", "inputs": {}},
    }
    if tag is not None:
        graph["__test_workflow_tag__"] = tag
    path = tmp_path / name
    path.write_text(json.dumps(graph))
    return str(path)


async def _make_comfy_pair(state: FakeComfyState | None = None) -> tuple[FakeComfy, ComfyHTTP]:
    fc = FakeComfy(state)
    base = await fc.start()
    client = ComfyHTTP(base, client_id="runner-test")
    return fc, client


async def _teardown_comfy(fc: FakeComfy, client: ComfyHTTP) -> None:
    await client.close()
    await fc.stop()


def _make_runner(
    comfy,
    judge,
    db_path: Path,
    *,
    concurrency: int = 2,
    breaker: CircuitBreaker | None = None,
    await_timeout: float = 20.0,
) -> tuple[BatchRunner, SQLiteStore]:
    store = SQLiteStore(db_path)
    supervisor = HealthSupervisor(comfy, launch_argv=None, stall_timeout=600.0)
    from comfy_controller.adapters.console_notifier import ConsoleNotifier

    config = RunnerConfig(
        concurrency=concurrency,
        tick_interval_s=3600.0,  # tick cadence irrelevant to these tests; keep it out of the way
        await_outcome_timeout_s=await_timeout,
        idle_poll_max_s=0.5,
    )
    runner = BatchRunner(
        comfy=comfy,
        store=store,
        judge=judge,
        supervisor=supervisor,
        notifier=ConsoleNotifier(),
        config=config,
        breaker=breaker,
    )
    return runner, store


async def _summary_counts(store: SQLiteStore) -> dict[AssetState, int]:
    return await store.summary()


# -------------------------------------------------------------- happy path


async def test_happy_path_all_saved(tmp_path):
    wf = _workflow_json(tmp_path)
    assets = [Asset(id=f"a{i}", workflow=wf, seed=i) for i in range(10)]

    fc, comfy = await _make_comfy_pair(FakeComfyState(fault=Fault.NONE))
    runner, store = _make_runner(comfy, FakeJudge(), tmp_path / "s.db")
    try:
        await runner.run(assets)
        summary = await _summary_counts(store)
        assert summary.get(AssetState.SAVED) == 10
        assert sum(summary.values()) == 10
    finally:
        await store.close()
        await _teardown_comfy(fc, comfy)


# --------------------------------------------------------- node error -> skip


async def test_node_error_retried_then_skipped_review(tmp_path):
    """NODE_ERROR ('RuntimeError: node blew up') doesn't match any TEMPORARY/
    PERMANENT rule, so it classifies UNKNOWN: exactly one retry, then
    SKIPPED_REVIEW. The retry backoff (10s, jittered) is real wall-clock time
    -- SQLiteStore gates `retry_after` against `time.time()`, which isn't
    injectable -- so this test genuinely waits for it."""
    wf = _workflow_json(tmp_path)
    assets = [Asset(id="a0", workflow=wf)]

    fc, comfy = await _make_comfy_pair(FakeComfyState(fault=Fault.NODE_ERROR, fault_after_n_jobs=0))
    runner, store = _make_runner(comfy, FakeJudge(), tmp_path / "s.db", concurrency=1)
    try:
        await asyncio.wait_for(runner.run(assets), timeout=30.0)
        summary = await _summary_counts(store)
        assert summary.get(AssetState.SKIPPED_REVIEW) == 1

        records = read_all_records(store)
        rec = next(r for r in records if r.asset.id == "a0")
        assert rec.attempt == 2  # first attempt + exactly one retry
    finally:
        await store.close()
        await _teardown_comfy(fc, comfy)


# ------------------------------------------------------ missing file -> permanent


async def test_missing_file_permanent_immediately(tmp_path):
    wf = _workflow_json(tmp_path)
    assets = [Asset(id="a0", workflow=wf)]

    fc, comfy = await _make_comfy_pair(FakeComfyState(fault=Fault.MISSING_FILE, fault_after_n_jobs=0))
    runner, store = _make_runner(comfy, FakeJudge(), tmp_path / "s.db", concurrency=1)
    try:
        await asyncio.wait_for(runner.run(assets), timeout=10.0)
        summary = await _summary_counts(store)
        assert summary.get(AssetState.PERMANENT_ERROR) == 1

        records = read_all_records(store)
        rec = next(r for r in records if r.asset.id == "a0")
        assert rec.attempt == 1  # no retry at all
    finally:
        await store.close()
        await _teardown_comfy(fc, comfy)


# ---------------------------------------------------------------- OOM -> retry


async def test_oom_retried_and_no_restart_requested(tmp_path):
    """OOM classifies TEMPORARY and must never be treated as a worker-crash
    signal (see core/policy.py). The fault is flipped off after the first
    failed submission so the retry succeeds without waiting through the full
    TEMPORARY backoff ladder more than once."""
    wf = _workflow_json(tmp_path)
    assets = [Asset(id="a0", workflow=wf)]

    fc, comfy = await _make_comfy_pair(FakeComfyState(fault=Fault.OOM, fault_after_n_jobs=0))

    async def _disable_fault_after_first_submission() -> None:
        while fc.state.submitted_count < 1:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.1)  # let the (near-instant) OOM branch resolve
        fc.state.fault = Fault.NONE

    disabler = asyncio.create_task(_disable_fault_after_first_submission())

    runner, store = _make_runner(comfy, FakeJudge(), tmp_path / "s.db", concurrency=1)
    try:
        await asyncio.wait_for(runner.run(assets), timeout=30.0)
        summary = await _summary_counts(store)
        assert summary.get(AssetState.SAVED) == 1

        assert runner.supervisor.restart_count == 0
        assert runner.supervisor.held is False
    finally:
        await disabler
        await store.close()
        await _teardown_comfy(fc, comfy)


# -------------------------------------------------------- judge fail -> fix -> pass


async def test_judge_fails_then_fix_applied_then_passes(tmp_path):
    wf = _workflow_json(tmp_path)
    assets = [Asset(id="a0", workflow=wf, seed=42)]

    fc, comfy = await _make_comfy_pair(FakeComfyState(fault=Fault.NONE))
    judge = FakeJudge()
    judge.queue_fail(name="anatomy_correct", evidence="six fingers", fix=FixProposal(action="reseed"))
    judge.queue_pass()

    runner, store = _make_runner(comfy, judge, tmp_path / "s.db", concurrency=1)
    try:
        await asyncio.wait_for(runner.run(assets), timeout=10.0)
        summary = await _summary_counts(store)
        assert summary.get(AssetState.SAVED) == 1
        assert len(judge.calls) == 2  # first fails, second (post-fix) passes
    finally:
        await store.close()
        await _teardown_comfy(fc, comfy)


# --------------------------------------------------------------- judge unavailable


async def test_judge_unavailable_is_not_saved_and_not_dropped(tmp_path):
    wf = _workflow_json(tmp_path)
    assets = [Asset(id="a0", workflow=wf)]

    fc, comfy = await _make_comfy_pair(FakeComfyState(fault=Fault.NONE))
    judge = FakeJudge()
    judge.queue_unavailable("simulated 529")

    runner, store = _make_runner(comfy, judge, tmp_path / "s.db", concurrency=1)
    try:
        await asyncio.wait_for(runner.run(assets), timeout=10.0)
        summary = await _summary_counts(store)
        assert summary.get(AssetState.JUDGE_UNAVAILABLE) == 1
        assert summary.get(AssetState.SAVED, 0) == 0
        assert sum(summary.values()) == 1  # never silently dropped
    finally:
        await store.close()
        await _teardown_comfy(fc, comfy)


# -------------------------------------------------------------- kill and resume


async def test_kill_and_resume_every_asset_reaches_one_terminal_state(tmp_path):
    """Simulates a `kill -9`: two rows are left CLAIMED/SUBMITTED (mid-flight,
    with no way for a fresh process to know what happened to them) by talking
    to the store directly, bypassing the runner. A brand new BatchRunner is
    then built against the same DB file and must reconcile + finish every
    asset, with no asset left non-terminal and no duplicate prompt_id (the
    latter is additionally enforced structurally by a UNIQUE constraint)."""
    wf = _workflow_json(tmp_path)
    assets = [Asset(id=f"a{i}", workflow=wf) for i in range(6)]
    db_path = tmp_path / "s.db"

    store1 = SQLiteStore(db_path)
    await store1.load_batch(assets)

    def job_key_fn(asset: Asset, attempt: int) -> str:
        return f"{asset.id}_a{attempt}"

    claimed = await store1.claim_next(job_key_fn=job_key_fn)  # left CLAIMED
    submitted_rec = await store1.claim_next(job_key_fn=job_key_fn)
    await store1.record_submission(submitted_rec.asset.id, "prompt-that-never-happened")  # left SUBMITTED
    assert claimed is not None and submitted_rec is not None
    await store1.close()  # simulated crash: nothing else flushed

    fc, comfy = await _make_comfy_pair(FakeComfyState(fault=Fault.NONE))
    runner, store2 = _make_runner(comfy, FakeJudge(), db_path, concurrency=3)
    try:
        await asyncio.wait_for(runner.run(), timeout=30.0)  # no `assets` -- DB already has them

        summary = await _summary_counts(store2)
        assert sum(summary.values()) == 6
        assert all(state.is_terminal for state in summary)
        assert summary.get(AssetState.SAVED) == 6

        rows = store2._conn.execute(
            "SELECT comfy_prompt_id, COUNT(*) c FROM assets WHERE comfy_prompt_id IS NOT NULL "
            "GROUP BY comfy_prompt_id HAVING c > 1"
        ).fetchall()
        assert rows == []
    finally:
        await store2.close()
        await _teardown_comfy(fc, comfy)


# ------------------------------------------------------------------ breaker


class _ScriptedComfy:
    """Minimal in-process ComfyPort stub. Unlike FakeComfy, outcomes are
    decided per-submission from the graph's own content, which is what lets
    this one test make one workflow fail every time while a different
    workflow succeeds every time -- something FakeComfy's single global
    fault setting cannot express."""

    def __init__(self, decide: Callable[[str, dict[str, Any]], JobOutcome]) -> None:
        self._decide = decide
        self._n = 0
        self._pending: dict[str, dict[str, Any]] = {}

    async def submit(self, graph: dict[str, Any], job_key: str, front: bool = False) -> str:
        self._n += 1
        prompt_id = f"p{self._n}"
        self._pending[prompt_id] = graph
        return prompt_id

    async def events(self):
        return
        yield {}  # pragma: no cover - never iterated by BatchRunner directly

    async def await_outcome(self, prompt_id: str, timeout: float) -> JobOutcome:
        graph = self._pending.pop(prompt_id, {})
        return self._decide(prompt_id, graph)

    async def queue_depth(self) -> tuple[int, int]:
        return (0, 0)

    async def history(self, prompt_id: str):
        return None

    async def is_responding(self) -> bool:
        return True

    async def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        return None

    async def interrupt(self, prompt_id: str | None = None) -> None:
        return None


async def test_breaker_opens_on_one_partition_other_partitions_keep_running(tmp_path):
    wf_a = _workflow_json(tmp_path, "wfA.json", tag="wfA")
    wf_b = _workflow_json(tmp_path, "wfB.json", tag="wfB")

    def decide(prompt_id: str, graph: dict[str, Any]) -> JobOutcome:
        if graph.get("__test_workflow_tag__") == "wfA":
            return JobOutcome(
                prompt_id=prompt_id,
                succeeded=False,
                error=ExecutionError(
                    prompt_id=prompt_id,
                    exception_type="FileNotFoundError",
                    exception_message="No such file or directory: input.png",
                ),
            )
        return JobOutcome(prompt_id=prompt_id, succeeded=True, output_paths=[f"{prompt_id}.png"])

    comfy = _ScriptedComfy(decide)

    # 6 assets on the failing partition (enough to trip the threshold-5
    # breaker and leave one stuck waiting on it), 3 on a healthy partition.
    assets = [Asset(id=f"a{i}", workflow=wf_a) for i in range(6)] + [
        Asset(id=f"b{i}", workflow=wf_b) for i in range(3)
    ]

    breaker = CircuitBreaker(now=time.monotonic, failure_threshold=5, cooldown_s=1.0)
    runner, store = _make_runner(comfy, FakeJudge(), tmp_path / "s.db", concurrency=3, breaker=breaker)
    try:
        await asyncio.wait_for(runner.run(assets), timeout=15.0)

        records = {r.asset.id: r for r in read_all_records(store)}
        b_states = [records[f"b{i}"].state for i in range(3)]
        assert b_states == [AssetState.SAVED] * 3  # other partition unaffected

        a_states = [records[f"a{i}"].state for i in range(6)]
        assert a_states.count(AssetState.PERMANENT_ERROR) == 6  # every one eventually resolves terminal
    finally:
        await store.close()


# ------------------------------------------------------------- job_key_fn fix


async def test_claim_next_uses_deterministic_job_key_fn(tmp_path):
    """Directly exercises fix #1: a supplied `job_key_fn` drives the claimed
    row's job_key, and it is reproducible across separate claims of the same
    (asset, attempt) pair -- unlike the uuid4 fallback."""
    store = SQLiteStore(tmp_path / "s.db")
    await store.load_batch([Asset(id="x", workflow="w.json", inputs={"p": "hi"}, seed=1)])

    def job_key_fn(asset: Asset, attempt: int) -> str:
        return f"{asset.job_key('modelV', 'graphH')}_a{attempt}"

    rec = await store.claim_next(job_key_fn=job_key_fn)
    assert rec is not None
    expected = f"{rec.asset.job_key('modelV', 'graphH')}_a1"
    assert rec.job_key == expected

    # Fallback path (no job_key_fn) still works and is opaque/unique.
    await store.load_batch([Asset(id="y", workflow="w.json")])
    rec2 = await store.claim_next()
    assert rec2 is not None
    assert rec2.job_key is not None and rec2.job_key != expected

    await store.close()
