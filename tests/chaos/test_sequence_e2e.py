"""End-to-end coverage for `AssetKind.SEQUENCE` through the real `BatchRunner`.

Before this file, sequences were "unit-tested but never exercised whole":
`tests/test_sequence.py` drives `SequenceRepairPlanner` directly with no I/O
at all (it only ever *proposes* steps, per its own module docstring), and
`grep -rn "AssetKind.SEQUENCE\\|frame_count" tests/test_runner.py` (checked
before writing this file) turns up nothing -- no test ever builds a sequence
`Asset` and pushes it through `BatchRunner._process_sequence`. That means the
`RegenerateFrame`/`QCCompare` steps the planner emits were never proven to be
consumed correctly by the runner, and the design doc's crash-recovery
contract for sequences ("frames depend on their predecessor, so a crash can't
be resumed frame-by-frame -- the whole sequence reruns from frame 0",
`core/sequence.py`) was never proven to actually happen end to end either.

Two scenarios, both driving a real `BatchRunner` against a real `SQLiteStore`
(only the ComfyUI HTTP boundary is faked, via `RoutingComfy`/`RoutingJudge`
from `conftest.py`, same as the rest of this chaos suite):

  * `test_sequence_frame_fails_qc_and_is_repaired_against_neighbours` -- a
    4-frame sequence where frame 2 fails QC, gets regenerated against frame 1
    (its predecessor), and is re-judged (the `QCCompare` step) before the
    whole sequence is accepted.
  * `test_sequence_crash_mid_generation_reruns_every_frame_from_scratch` -- a
    row left CLAIMED/SUBMITTED by talking to the store directly (the same
    "simulated crash" technique `tests/test_runner.py`'s own single-asset
    kill/resume test uses), proving reconciliation reruns the ENTIRE
    sequence rather than trying to resume mid-way. This is not a design
    choice being tested weakly: `_process_sequence`'s per-frame progress
    (`frame_paths`/`attempt_counts`) is PURELY in-memory and never persisted
    anywhere, so a crash at any point during sequence generation is
    indistinguishable to reconciliation from a crash before frame 0 even
    started -- rerunning everything is the only behaviour the persistence
    model allows, and this proves it actually happens (every frame judged
    again, not just the ones "after" the simulated crash point) rather than
    silently losing frames or hanging.
"""

from __future__ import annotations

import asyncio

import pytest

from comfy_controller.adapters.comfy_http import ComfyHTTP
from comfy_controller.adapters.console_notifier import ConsoleNotifier
from comfy_controller.adapters.sqlite_store import SQLiteStore
from comfy_controller.core.sequence import SequenceRepairPlanner
from comfy_controller.models import Asset, AssetKind, AssetState, CheckResult, FixProposal, QCVerdict
from comfy_controller.runner import BatchRunner, RunnerConfig, read_all_records
from comfy_controller.supervisor import HealthSupervisor
from comfy_controller.testing.fake_comfy import FakeComfy, FakeComfyState

from .conftest import RoutingComfy, RoutingJudge, always_succeed, assert_all_terminal, tagged_workflow

FRAME_COUNT = 4


def _make_runner_config() -> RunnerConfig:
    return RunnerConfig(
        concurrency=2,
        # Large on purpose: keeps HealthSupervisor's periodic tick out of the
        # way, same rationale as test_kill_resume.py's own config.
        tick_interval_s=3600.0,
        await_outcome_timeout_s=10.0,
        idle_poll_max_s=0.2,
    )


def _make_runner(comfy, judge, store) -> BatchRunner:
    supervisor = HealthSupervisor(comfy, launch_argv=None, stall_timeout=600.0)
    return BatchRunner(
        comfy=comfy,
        store=store,
        judge=judge,
        supervisor=supervisor,
        notifier=ConsoleNotifier(),
        config=_make_runner_config(),
        seq_planner=SequenceRepairPlanner(),
    )


def _pass(evidence: str) -> QCVerdict:
    return QCVerdict(checks=[CheckResult(name="anatomy_correct", evidence=evidence, passed=True)])


# ==================================================== QC failure + repair


@pytest.mark.slow
async def test_sequence_frame_fails_qc_and_is_repaired_against_neighbours(tmp_path):
    fc = FakeComfy(FakeComfyState())
    base = await fc.start()
    real_client = ComfyHTTP(base, client_id="seq-e2e-repair")

    asset_id = "sequence-repair"
    wf = tagged_workflow(tmp_path, asset_id)
    # Generation itself always "succeeds" here -- QC pass/fail is entirely a
    # judge-side decision (RoutingJudge, scripted below), matching how
    # comfy_controller actually separates "did ComfyUI produce an image" from
    # "does the image pass the rubric".
    comfy = RoutingComfy(real_client, {asset_id: always_succeed()})

    judge = RoutingJudge()
    fail_frame_2 = QCVerdict(
        checks=[CheckResult(name="anatomy_correct", evidence="six fingers on frame 2", passed=False)],
        fix=FixProposal(action="reseed"),
    )
    # `_process_sequence`'s initial pass judges frames 0..3 IN ORDER (a plain
    # sequential for-loop, no concurrency), so these five scripted verdicts
    # play back in exactly this order: frame0 pass, frame1 pass, frame2
    # FAIL (triggers repair), frame3 pass, then the repair round's QCCompare
    # re-judge of the regenerated frame2, which must pass to end the round.
    judge.script(
        asset_id,
        _pass("frame 0 clean"),
        _pass("frame 1 clean"),
        fail_frame_2,
        _pass("frame 3 clean"),
        _pass("regenerated frame 2 now matches its neighbours"),
    )

    store = SQLiteStore(tmp_path / "seq_repair.db")
    runner = _make_runner(comfy, judge, store)
    asset = Asset(id=asset_id, kind=AssetKind.SEQUENCE, workflow=wf, frame_count=FRAME_COUNT)

    try:
        await asyncio.wait_for(runner.run([asset]), timeout=30.0)

        await assert_all_terminal(store, 1)
        records = {r.asset.id: r for r in read_all_records(store)}
        rec = records[asset_id]

        assert rec.state == AssetState.SAVED, f"expected SAVED, got {rec.state} (error={rec.last_error})"
        assert len(rec.output_paths) == FRAME_COUNT, rec.output_paths

        # Every frame really was judged, in order, including the extra
        # QCCompare re-check of the regenerated frame 2 -- proof the planner's
        # RegenerateFrame/QCCompare steps were actually consumed by the
        # runner, not merely proposed by the pure planner (tests/test_sequence.py).
        judged = [paths for aid, paths in judge.calls if aid == asset_id]
        assert len(judged) == 5, f"expected 5 judge calls (4 initial + 1 repair re-check), got {len(judged)}"
        for paths in judged:
            assert len(paths) == 1  # one frame per judge call -- JudgePort.judge() is frozen, no reference-frame arg

        # Frame 2 specifically was regenerated: its attempt count in the
        # planner's own bookkeeping is invisible from here, but the fact that
        # a 5th judge call happened at all (rather than stopping at 4) proves
        # the repair round actually ran a RegenerateFrame + QCCompare pair.
    finally:
        await store.close()
        await real_client.close()
        await fc.stop()


# ============================================================ crash mid-sequence


@pytest.mark.slow
@pytest.mark.parametrize(
    "leave_state",
    ["claimed", "submitted"],
    ids=["crash-before-any-frame-submitted", "crash-mid-frame-submission"],
)
async def test_sequence_crash_mid_generation_reruns_every_frame_from_scratch(tmp_path, leave_state):
    asset_id = "sequence-crash"
    wf = tagged_workflow(tmp_path, asset_id)
    asset = Asset(id=asset_id, kind=AssetKind.SEQUENCE, workflow=wf, frame_count=FRAME_COUNT)
    db_path = tmp_path / "seq_crash.db"

    # ---- simulate the crash: talk to the store directly, never through a
    # live BatchRunner -- the same technique test_runner.py's own single-
    # asset kill/resume test uses, and the only technique that makes sense
    # here at all, since `_process_sequence`'s per-frame progress is never
    # persisted (see module docstring): there is nothing "further along" a
    # more elaborate simulated crash could leave behind for reconciliation to
    # find.
    store1 = SQLiteStore(db_path)
    await store1.load_batch([asset])

    def job_key_fn(a: Asset, attempt: int) -> str:
        return f"{a.id}_a{attempt}"

    rec = await store1.claim_next(job_key_fn=job_key_fn)  # -> CLAIMED
    assert rec is not None
    if leave_state == "submitted":
        # A crash slightly later: some frame's /prompt POST landed, but the
        # process died before that frame (or any later one) ever resolved.
        # record_submission() accepts 'claimed' OR 'submitted' as the prior
        # state specifically because a sequence submits many prompts, one per
        # frame, under the ONE claim (see sqlite_store.py's own comment on
        # this) -- so this is a faithful stand-in for "died partway into
        # frame 0", not a fabricated state the real system can't reach.
        await store1.record_submission(asset_id, "prompt-that-never-resolved")
    await store1.close()  # simulated crash: nothing else flushed

    # ---- resume: a brand new BatchRunner/store pair against the same file.
    fc = FakeComfy(FakeComfyState())
    base = await fc.start()
    real_client = ComfyHTTP(base, client_id="seq-e2e-crash")
    comfy = RoutingComfy(real_client, {asset_id: always_succeed()})
    judge = RoutingJudge()  # unscripted for this asset -> every frame passes QC first try

    store2 = SQLiteStore(db_path)
    runner = _make_runner(comfy, judge, store2)

    try:
        await asyncio.wait_for(runner.run(), timeout=30.0)  # no `assets` arg: resume against the existing DB

        await assert_all_terminal(store2, 1)
        records = {r.asset.id: r for r in read_all_records(store2)}
        result = records[asset_id]

        assert result.state == AssetState.SAVED, f"expected SAVED, got {result.state} (error={result.last_error})"
        assert len(result.output_paths) == FRAME_COUNT
        assert result.crash_count >= 1, "reconciliation must have recorded the crash it recovered from"

        # The headline claim: ALL FOUR frames were (re)judged in this single
        # resumed run, not just the ones "after" wherever the simulated crash
        # happened. A resume that tried (and failed) to preserve partial
        # progress would show fewer than 4 calls here; one that silently
        # duplicated work would show more.
        judged = [paths for aid, paths in judge.calls if aid == asset_id]
        assert len(judged) == FRAME_COUNT, (
            f"expected exactly {FRAME_COUNT} judge calls (every frame regenerated from scratch "
            f"post-crash), got {len(judged)}"
        )
    finally:
        await store2.close()
        await real_client.close()
        await fc.stop()
