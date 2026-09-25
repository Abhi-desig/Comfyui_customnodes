"""The 50-asset chaos acceptance suite.

Every scenario the task brief calls out is covered, split across a couple of
focused `BatchRunner.run()` calls rather than crammed into one single
mega-run:

* `test_bulk_chaos_run_every_asset_reaches_one_terminal_state` -- ~50 assets
  in ONE run, injecting (simultaneously, via `RoutingComfy`/`RoutingJudge`
  per-asset scripting -- see conftest.py) OOM, a permanent missing-file
  error, a QC failure that gets fixed and one that gets parked, a judge 429
  storm, and one workflow partition tripping its circuit breaker while
  another partition keeps completing. Also the primary carrier of the
  headline invariant.
* `test_zombie_detected_restarts_and_loses_no_asset` -- the #16312 zombie,
  which needs a *real* FakeComfy+ComfyHTTP+HealthSupervisor (WS events, a
  front=True canary, HTTP always-200) rather than the scripted stand-ins the
  bulk run uses, kept separate so its timing isn't at the mercy of ~50
  concurrent scripted jobs.

Splitting it this way is a deliberate choice, not a shortcut: cramming a
genuine zombie/restart timeline into the same run as ~50 fast scripted
assets would make the zombie assertions racy against unrelated scheduling
noise, which is exactly the kind of flaky-by-construction test this suite
must not be.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path

import pytest
from PIL import Image

from comfy_controller.adapters.comfy_http import ComfyHTTP
from comfy_controller.adapters.console_notifier import ConsoleNotifier
from comfy_controller.adapters.sqlite_store import SQLiteStore
from comfy_controller.core.policy import BreakerState, CircuitBreaker, GlobalBreaker
from comfy_controller.core.sequence import SequenceRepairPlanner
from comfy_controller.models import Asset, AssetState, CheckResult, FixProposal, QCVerdict
from comfy_controller.ports import JudgeUnavailable
from comfy_controller.runner import BatchRunner, RunnerConfig, read_all_records
from comfy_controller.supervisor import HealthSupervisor
from comfy_controller.testing.fake_comfy import Fault, FakeComfy, FakeComfyState
from comfy_controller.testing.fake_judge import FakeJudge

from .conftest import (
    RoutingComfy,
    RoutingJudge,
    always_fail,
    always_succeed,
    assert_all_terminal,
    fail_once_then_succeed,
    tagged_workflow,
)


# ============================================================= the bulk run


def _assert_genuinely_readable_png(path: Path) -> None:
    """Not just "the file exists": genuinely decodable, matching what
    `prefilter.evaluate`'s own `Image.open(path); img.load()` does. A stub
    like `b"not-a-real-png"` (fine for the kill/resume suite, which only
    checks existence) would fail this and must never be mistaken for
    coverage of the judge/prefilter read-the-file path."""
    assert path.is_file(), f"{path}: not a regular file"
    assert path.stat().st_size > 0, f"{path}: 0-byte file"
    with Image.open(path) as img:
        img.load()


@pytest.mark.slow
async def test_bulk_chaos_run_every_asset_reaches_one_terminal_state(tmp_path):
    # `comfy_output_dir` configured end-to-end (FakeComfy's own write side AND
    # the runner's read side) so this run genuinely exercises the judge/
    # prefilter read-the-file path instead of leaving `comfy_output_dir`
    # unset -- which is exactly the gap that let a real bug ship unnoticed
    # (see RoutingComfy's own docstring in conftest.py for how scripted
    # "succeeded" outcomes are backed by real files too, not just names).
    output_dir = tmp_path / "comfy_outputs"
    output_dir.mkdir()

    fc = FakeComfy(FakeComfyState(output_dir=output_dir))
    base = await fc.start()
    real_client = ComfyHTTP(base, client_id="chaos-bulk")

    # ---- build the ~50-asset manifest, grouped by scripted behaviour ------
    assets: list[Asset] = []

    wf_happy = tagged_workflow(tmp_path, "happy")
    assets += [Asset(id=f"happy-{i}", workflow=wf_happy, seed=i) for i in range(30)]

    oom_ids = [f"oom-{i}" for i in range(3)]
    for oid in oom_ids:
        assets.append(Asset(id=oid, workflow=tagged_workflow(tmp_path, oid)))

    wf_missing = tagged_workflow(tmp_path, "missing_file")
    missing_ids = [f"missing-{i}" for i in range(3)]
    assets += [Asset(id=aid, workflow=wf_missing) for aid in missing_ids]

    wf_qcfix = tagged_workflow(tmp_path, "qcfix")
    qcfix_ids = [f"qcfix-{i}" for i in range(2)]
    assets += [Asset(id=aid, workflow=wf_qcfix, seed=i) for i, aid in enumerate(qcfix_ids)]

    wf_qcpark = tagged_workflow(tmp_path, "qcpark")
    qcpark_ids = [f"qcpark-{i}" for i in range(2)]
    assets += [Asset(id=aid, workflow=wf_qcpark, seed=i) for i, aid in enumerate(qcpark_ids)]

    wf_storm = tagged_workflow(tmp_path, "judgestorm")
    storm_ids = [f"judgestorm-{i}" for i in range(3)]
    assets += [Asset(id=aid, workflow=wf_storm) for aid in storm_ids]

    wf_breaker_bad = tagged_workflow(tmp_path, "breaker_bad")
    breaker_bad_ids = [f"breaker-bad-{i}" for i in range(6)]
    assets += [Asset(id=aid, workflow=wf_breaker_bad) for aid in breaker_bad_ids]

    wf_breaker_good = tagged_workflow(tmp_path, "breaker_good")
    breaker_good_ids = [f"breaker-good-{i}" for i in range(3)]
    assets += [Asset(id=aid, workflow=wf_breaker_good) for aid in breaker_good_ids]

    total = len(assets)
    assert total == 30 + 3 + 3 + 2 + 2 + 3 + 6 + 3  # == 52, "~50"

    # ---- script the ComfyPort side ----------------------------------------
    scripts = {
        "happy": always_succeed(),
        "missing_file": always_fail("FileNotFoundError", "No such file or directory: input.png"),
        "qcfix": always_succeed(),
        "qcpark": always_succeed(),
        "judgestorm": always_succeed(),
        "breaker_bad": always_fail("FileNotFoundError", "No such file or directory: input.png"),
        "breaker_good": always_succeed(),
    }
    for oid in oom_ids:
        scripts[oid] = fail_once_then_succeed("torch.cuda.OutOfMemoryError", "CUDA out of memory")

    comfy = RoutingComfy(real_client, scripts, output_dir=output_dir)

    # ---- script the JudgePort side ----------------------------------------
    judge = RoutingJudge()
    for aid in qcfix_ids:
        judge.script(
            aid,
            QCVerdict(
                checks=[CheckResult(name="anatomy_correct", evidence="six fingers", passed=False)],
                fix=FixProposal(action="reseed"),
            ),
            QCVerdict(checks=[CheckResult(name="anatomy_correct", evidence="fixed", passed=True)]),
        )
    for aid in qcpark_ids:
        judge.script(
            aid,
            QCVerdict(
                checks=[CheckResult(name="matches_brief", evidence="doesn't match the brief", passed=False)],
                fix=FixProposal(action="none"),
            ),
        )
    for aid in storm_ids:
        judge.script(aid, JudgeUnavailable("simulated 429 storm"))

    # ---- wire the runner ----------------------------------------------------
    store = SQLiteStore(tmp_path / "chaos.db")
    supervisor = HealthSupervisor(comfy, launch_argv=None, stall_timeout=600.0)
    breaker = CircuitBreaker(now=time.monotonic, failure_threshold=5, cooldown_s=1.0)
    global_breaker = GlobalBreaker(now=time.monotonic)
    config = RunnerConfig(
        concurrency=8,
        tick_interval_s=2.0,
        await_outcome_timeout_s=20.0,
        idle_poll_max_s=0.5,
        comfy_output_dir=output_dir,
    )
    runner = BatchRunner(
        comfy=comfy,
        store=store,
        judge=judge,
        supervisor=supervisor,
        notifier=ConsoleNotifier(),
        config=config,
        breaker=breaker,
        global_breaker=global_breaker,
        seq_planner=SequenceRepairPlanner(),
    )

    try:
        # OOM's real backoff ladder is genuine wall-clock time (SQLiteStore
        # gates retry_after on time.time(), which isn't injectable -- see
        # test_runner.py's own note on this) and the breaker's cooldown adds
        # another ~1s of real polling, so this genuinely takes a handful of
        # seconds even though nothing else in this run sleeps.
        await asyncio.wait_for(runner.run(assets), timeout=60.0)

        summary = await assert_all_terminal(store, total)
        records = {r.asset.id: r for r in read_all_records(store)}

        # -- happy path --------------------------------------------------------
        assert all(records[f"happy-{i}"].state == AssetState.SAVED for i in range(30))

        # -- OOM: retried at the backoff ladder, never a restart signal ---------
        for oid in oom_ids:
            rec = records[oid]
            assert rec.state == AssetState.SAVED, f"{oid}: {rec.state}"
            assert rec.attempt == 2, f"{oid}: expected exactly one retry, got attempt={rec.attempt}"
        assert runner.supervisor.restart_count == 0
        assert runner.supervisor.held is False

        # -- missing file: PERMANENT_ERROR immediately, zero retries -------------
        for mid in missing_ids:
            rec = records[mid]
            assert rec.state == AssetState.PERMANENT_ERROR
            assert rec.attempt == 1

        # -- QC fix path vs. QC park path -----------------------------------------
        for aid in qcfix_ids:
            assert records[aid].state == AssetState.SAVED
        for aid in qcpark_ids:
            assert records[aid].state == AssetState.PARKED_APPROVAL

        # -- judge 429 storm: never SAVED, never dropped --------------------------
        for aid in storm_ids:
            assert records[aid].state == AssetState.JUDGE_UNAVAILABLE

        # -- breaker: partition A trips OPEN, partition B keeps completing ---------
        for aid in breaker_bad_ids:
            assert records[aid].state == AssetState.PERMANENT_ERROR
        for aid in breaker_good_ids:
            assert records[aid].state == AssetState.SAVED
        assert breaker.state(wf_breaker_bad) is BreakerState.OPEN
        assert breaker.state(wf_breaker_good) is BreakerState.CLOSED

        assert summary.get(AssetState.SAVED, 0) == 30 + len(oom_ids) + len(qcfix_ids) + len(breaker_good_ids)

        # -- the read-the-file path itself: `comfy_output_dir` is configured
        # for this whole run, so every SAVED asset's `output_paths` must
        # resolve (via `BatchRunner.resolve_output_path`) to a real,
        # Pillow-decodable file under `output_dir` -- not merely a filename
        # string that happens to satisfy the `AssetState.SAVED` check. This
        # is the exact path (`_unreadable_outputs` + the judge opening the
        # image) that a shipped bug was previously free to break unnoticed,
        # because the rest of this suite never configured `comfy_output_dir`
        # at all. Covers every SAVED family: plain happy path, a
        # once-then-succeeds OOM retry, a QC-fixed asset, and the surviving
        # breaker partition.
        saved_ids = (
            [f"happy-{i}" for i in range(30)] + oom_ids + qcfix_ids + breaker_good_ids
        )
        assert saved_ids, "test setup: expected at least one SAVED asset"
        for aid in saved_ids:
            rec = records[aid]
            assert rec.state == AssetState.SAVED
            assert rec.output_paths, f"{aid}: SAVED with no recorded output_paths"
            for rel_path in rec.output_paths:
                _assert_genuinely_readable_png(runner.resolve_output_path(rel_path))

        # And the judge itself was hand a real, already-resolved, existing
        # path -- not a bare filename resolved against nothing (the old,
        # vacuously-passing shape) -- for every asset whose scripted outcome
        # reached the judge at all (SAVED, PARKED_APPROVAL, and the judge-
        # unavailable storm all get this far before deciding).
        judged_ids = saved_ids + qcpark_ids + storm_ids
        judge_calls_by_asset: dict[str, list] = {}
        for aid, paths in judge.calls:
            judge_calls_by_asset.setdefault(aid, []).extend(paths)
        for aid in judged_ids:
            paths = judge_calls_by_asset.get(aid)
            assert paths, f"{aid}: judge was never called with any image paths"
            for path in paths:
                assert path.is_absolute(), f"{aid}: {path} was never resolved against comfy_output_dir"
                assert path.exists(), f"{aid}: judge was handed a path with nothing on disk: {path}"
    finally:
        await store.close()
        await real_client.close()
        await fc.stop()


# ============================================================= zombie / restart


@pytest.mark.slow
async def test_zombie_detected_restarts_and_loses_no_asset(tmp_path):
    """#16312 in miniature, at the BatchRunner level: HTTP keeps answering
    200, the queue keeps accepting jobs, GPU reads idle, no WS progress ever
    arrives. `hysteresis_n=3` means it takes three consecutive bad
    assessments before the supervisor actually restarts -- one bad sample
    alone must never be enough. (This is what "three consecutive failures ->
    restart triggered" in the brief describes: the hysteresis gate on this
    same zombie-detection mechanism, not a second, independent trigger --
    there is no other code path in `HealthSupervisor`/`BatchRunner` that
    restarts ComfyUI at all.) Every asset submitted while the worker was
    wedged must still reach a terminal state once the (simulated) restart
    recovers it -- none may be silently lost.
    """
    # `output_dir` configured on both sides (FakeComfy's write side, the
    # runner's read side below) so the victims that eventually complete after
    # the restart go through the genuine judge/prefilter read-the-file path
    # too, same rationale as the bulk run test.
    output_dir = tmp_path / "comfy_outputs"
    output_dir.mkdir()

    fc = FakeComfy(FakeComfyState(fault=Fault.ZOMBIE, fault_after_n_jobs=0, output_dir=output_dir))
    base = await fc.start()
    comfy = ComfyHTTP(base, client_id="chaos-zombie")

    store = SQLiteStore(tmp_path / "zombie.db")

    # gpu_looks_idle starts True so the very first assessments escalate
    # straight to the canary (exactly what a real idle-GPU zombie looks
    # like); the healer below flips it once the restart lands, so the
    # *simulated recovery* doesn't itself trigger an endless storm of
    # canary re-checks fighting the real retry traffic for FakeComfy's
    # single-job-at-a-time worker.
    gpu_state = {"idle": True}

    supervisor = HealthSupervisor(
        comfy,
        stall_timeout=0.05,
        canary_timeout=0.2,
        hysteresis_n=3,
        hard_ceiling=100.0,
        # Restart discipline (see supervisor.py) normally backs off for real
        # wall-clock seconds before actually restarting -- irrelevant to what
        # this test is proving (detection + hysteresis + no lost asset), and
        # left at its production default it races against the victims' own
        # ~10s TEMPORARY retry ladder closely enough to be flaky under load.
        backoff_base=0.05,
        backoff_cap=0.05,
        gpu_util_fn=lambda: (1.0 if gpu_state["idle"] else 95.0),
        cpu_util_fn=lambda: (1.0 if gpu_state["idle"] else 95.0),
        launch_argv=None,  # dry-run: no real process to spawn/kill
    )

    restart_events: list[int] = []

    async def _watch_restart_and_heal() -> None:
        """Stands in for what a *real* ComfyUI restart would do: the old
        process is gone and a fresh one takes its place. FakeComfy's
        dry-run restart (`launch_argv=None`) doesn't touch the fake server
        itself, so this coroutine performs the equivalent by hand -- clear
        the wedged queue, turn the fault off, and stop reporting idle GPU --
        the moment `restart_count` actually increments."""
        prior = 0
        while True:
            await asyncio.sleep(0.02)
            current = supervisor.restart_count
            if current > prior:
                restart_events.append(current)
                fc.state.fault = Fault.NONE
                fc._queue.clear()
                fc._running = None
                gpu_state["idle"] = False
                return
            prior = current

    healer = asyncio.create_task(_watch_restart_and_heal())

    wf = tagged_workflow(tmp_path, "zombie_victim")
    assets = [Asset(id=f"victim-{i}", workflow=wf) for i in range(4)]

    config = RunnerConfig(
        concurrency=4, tick_interval_s=0.05, await_outcome_timeout_s=2.0, idle_poll_max_s=0.1,
        comfy_output_dir=output_dir,
    )
    judge = FakeJudge()
    runner = BatchRunner(
        comfy=comfy,
        store=store,
        judge=judge,
        supervisor=supervisor,
        notifier=ConsoleNotifier(),
        config=config,
    )

    try:
        await asyncio.wait_for(runner.run(assets), timeout=45.0)

        assert restart_events == [1], f"expected exactly one restart, got {restart_events}"
        assert supervisor.held is False

        summary = await assert_all_terminal(store, len(assets))
        assert summary.get(AssetState.SAVED, 0) == len(assets)

        # Every recovered victim's output is genuinely readable, not just a
        # filename that satisfies AssetState.SAVED -- and the judge itself
        # was handed the real, resolved, on-disk path.
        records = {r.asset.id: r for r in read_all_records(store)}
        judge_calls_by_asset: dict[str, list] = {}
        for aid, paths in judge.calls:
            judge_calls_by_asset.setdefault(aid, []).extend(paths)
        for asset in assets:
            rec = records[asset.id]
            assert rec.output_paths, f"{asset.id}: SAVED with no recorded output_paths"
            for rel_path in rec.output_paths:
                _assert_genuinely_readable_png(runner.resolve_output_path(rel_path))
            paths = judge_calls_by_asset.get(asset.id)
            assert paths, f"{asset.id}: judge was never called"
            for path in paths:
                assert path.is_absolute() and path.exists()
    finally:
        healer.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await healer
        await store.close()
        await comfy.close()
        await fc.stop()
