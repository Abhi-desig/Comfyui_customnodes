"""Real `SIGKILL` against a genuine `comfy_controller.cli` subprocess.

`tests/test_runner.py::test_kill_and_resume_every_asset_reaches_one_terminal_state`
already covers the *simulated* crash path (rows left `claimed`/`submitted` by
talking to the store directly). What that can't prove -- because nothing was
ever actually killed -- is that the real thing survives an in-flight
`fsync`/WAL boundary: a `kill -9` lands wherever the OS scheduler happens to
be, not wherever a test author finds it convenient. This file adds exactly
that: a real subprocess, a real signal, at several different points in the
claim -> submit -> outcome lifecycle.

The two `FakeComfy` timing knobs (`slow_down_prompt_endpoint`,
`FakeComfyState.step_delay`) exist purely to widen otherwise-microsecond
windows (CLAIMED-but-not-yet-SUBMITTED, SUBMITTED-but-not-yet-resolved) to
something a polling loop from a separate process can reliably catch -- they
do not change *what* is being proved, only how reliably the test can arrange
to be standing in the right place when the signal lands.
"""

from __future__ import annotations

import signal

import pytest

from comfy_controller.adapters.sqlite_store import SQLiteStore
from comfy_controller.models import AssetState
from comfy_controller.runner import read_all_records
from comfy_controller.testing.fake_comfy import FakeComfyState

from .conftest import (
    assert_all_terminal,
    poll_db_until,
    reap,
    sigkill_and_wait,
    slow_down_prompt_endpoint,
    spawn_controller,
    start_output_writing_comfy,
    tagged_workflow,
    wait_for_exit,
    write_manifest,
    write_yaml,
)

N_ASSETS = 6

# Each SQL: first column truthy on at least one row -> the checkpoint has
# been reached and it's safe to send SIGKILL.
CHECKPOINTS = {
    # At least two assets have left PENDING -- with concurrency=2 this means
    # both workers currently have something claimed/submitted at once.
    "claimed_n": "SELECT COUNT(*) >= 2 FROM assets WHERE state != 'pending'",
    # A row has been claimed (job_key assigned) but the submission that
    # would move it to 'submitted' hasn't landed yet -- the kill lands
    # between claim_next() and record_submission().
    "mid_submit": "SELECT COUNT(*) >= 1 FROM assets WHERE state = 'claimed' AND job_key IS NOT NULL",
    # A row has a durably-recorded comfy_prompt_id but hasn't resolved to a
    # terminal state yet -- the kill lands after record_submission() but
    # before the outcome (success or failure) is ever transitioned.
    "post_submit_pre_outcome": "SELECT COUNT(*) >= 1 FROM assets WHERE state = 'submitted'",
}


def _config_dict(base_url: str, db_path, output_dir) -> dict:
    return {
        "comfy_base_url": base_url,
        "db_path": str(db_path),
        "comfy_output_dir": str(output_dir),
        "judge_backend": "fake",
        "notifier": "console",
        "concurrency": 2,
        "await_outcome_timeout_s": 15.0,
        "tick_interval_s": 3600.0,  # keep the supervisor out of the way
    }


@pytest.mark.slow
@pytest.mark.parametrize("checkpoint", sorted(CHECKPOINTS))
async def test_real_sigkill_then_resume_every_asset_terminal(tmp_path, checkpoint):
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    # step_delay widens the SUBMITTED-but-not-yet-resolved window;
    # slow_down_prompt_endpoint widens the CLAIMED-but-not-yet-SUBMITTED one.
    # Both must be set up before fc.start() (see conftest docstrings).
    fc = start_output_writing_comfy(output_dir, FakeComfyState(step_delay=0.3))
    slow_down_prompt_endpoint(fc, delay_s=0.5)
    base_url = await fc.start()

    try:
        wf = tagged_workflow(tmp_path, "kill-resume")
        manifest_path = tmp_path / "manifest.yaml"
        write_manifest(manifest_path, [{"id": f"a{i}", "workflow": wf} for i in range(N_ASSETS)])

        db_path = tmp_path / "kr.db"
        config_path = tmp_path / "config.yaml"
        write_yaml(config_path, _config_dict(base_url, db_path, output_dir))

        proc = spawn_controller(config_path, manifest_path, mode="run")
        try:
            await poll_db_until(db_path, CHECKPOINTS[checkpoint], timeout=20.0)
            returncode = await sigkill_and_wait(proc, timeout=10.0)
            # A genuinely killed process, not one that happened to exit
            # cleanly before we got to it -- if this ever reads 0, the
            # checkpoint fired too late to prove anything.
            assert returncode == -signal.SIGKILL, f"expected SIGKILL, got returncode={returncode}"
        finally:
            await reap(proc)

        # Resume against the SAME db, no manifest -- the real operational
        # path (docs/runbook.md §3): "resuming IS just starting it again
        # against the same DB."
        proc2 = spawn_controller(config_path, mode="resume")
        try:
            returncode2 = await wait_for_exit(proc2, timeout=60.0)
        finally:
            await reap(proc2)
        assert returncode2 == 0, f"resume subprocess exited {returncode2}"

        store = SQLiteStore(db_path)
        try:
            summary = await assert_all_terminal(store, N_ASSETS)
            records = {r.asset.id: r for r in read_all_records(store)}

            # No asset both SAVED and missing its output file.
            saved = [r for r in records.values() if r.state == AssetState.SAVED]
            for rec in saved:
                assert rec.output_paths, f"{rec.asset.id}: SAVED with no recorded output_paths"
                for rel_path in rec.output_paths:
                    assert (output_dir / rel_path).exists(), (
                        f"{rec.asset.id}: SAVED but output file missing: {output_dir / rel_path}"
                    )

            # crash_count bookkeeping: exactly one kill happened, so nothing
            # should show more than one recorded crash, and -- this is the
            # part that would silently pass if the checkpoint never actually
            # caught anything mid-flight -- at least one asset must show it
            # really did.
            assert all(r.crash_count <= 1 for r in records.values()), {
                r.asset.id: r.crash_count for r in records.values() if r.crash_count > 1
            }
            assert any(r.crash_count == 1 for r in records.values()), (
                f"no asset shows crash_count==1 -- the '{checkpoint}' checkpoint "
                "never actually caught anything mid-flight"
            )
            assert summary.get(AssetState.SAVED, 0) == N_ASSETS
        finally:
            await store.close()
    finally:
        await fc.stop()


@pytest.mark.slow
async def test_crash_count_exhaustion_skips_asset_after_two_crashes(tmp_path):
    """design doc: an asset that crashes (found needing reconciliation) twice
    is given up on rather than retried forever (`_reconcile_one`,
    `sqlite_store.py`'s `crash_count` bookkeeping). Forcing a *third*
    real-kill-while-claimed deterministically needs two things this test
    controls on purpose:

    * `comfy_output_dir` is left unset in the config, so a row killed before
      its `comfy_prompt_id` is durably recorded (the "mid-submit" window)
      has no way to be resolved by reconciliation except "ambiguous, requeue"
      -- exactly the repeatable per-crash outcome this test needs, with no
      dependence on whether the real HTTP request happened to finish
      landing on the FakeComfy side before the kill.
    * `concurrency=1` and a single asset removes any ambiguity about which
      row a `SIGKILL` catches.

    Three real kills, each at the same "claimed, not yet submitted"
    checkpoint, on the one row in the DB: the first two must each resolve
    ambiguously (`crash_count` 0->1->2); the third must be recognised at
    startup reconciliation as already having crashed twice and skip the
    asset without attempting anything further.
    """
    fc = start_output_writing_comfy(tmp_path / "unused-outputs", FakeComfyState(step_delay=0.3))
    slow_down_prompt_endpoint(fc, delay_s=0.5)
    base_url = await fc.start()

    try:
        wf = tagged_workflow(tmp_path, "crash-exhaustion")
        manifest_path = tmp_path / "manifest.yaml"
        write_manifest(manifest_path, [{"id": "lone", "workflow": wf}])

        db_path = tmp_path / "kr.db"
        config_path = tmp_path / "config.yaml"
        cfg = {
            "comfy_base_url": base_url,
            "db_path": str(db_path),
            # deliberately no comfy_output_dir -- see docstring
            "judge_backend": "fake",
            "notifier": "console",
            "concurrency": 1,
            "await_outcome_timeout_s": 15.0,
            "tick_interval_s": 3600.0,
        }
        write_yaml(config_path, cfg)

        # The kill always lands on a row that is CLAIMED with a job_key, so
        # that alone cannot be the checkpoint: the PREVIOUS round's corpse
        # already satisfies it, and a resume polled on it gets SIGKILLed
        # before it has reconciled anything (crash_count never advances, and
        # the process contributes nothing but a wasted round). `attempt` is
        # what distinguishes them -- claim_next() bumps it, so waiting for
        # `attempt >= n` is waiting for *this* process to have reconciled the
        # corpse and taken a fresh claim of its own.
        def mid_submit_sql(attempt: int) -> str:
            return (
                "SELECT COUNT(*) >= 1 FROM assets "
                f"WHERE state = 'claimed' AND job_key IS NOT NULL AND attempt >= {attempt}"
            )

        # Crash #1: the only real run that also has to load the manifest.
        proc = spawn_controller(config_path, manifest_path, mode="run")
        try:
            await poll_db_until(db_path, mid_submit_sql(1), timeout=20.0)
            rc = await sigkill_and_wait(proc, timeout=10.0)
        finally:
            await reap(proc)
        assert rc == -signal.SIGKILL

        # Crash #2 and crash #3: `resume` claims the (now RETRY_WAIT, then
        # re-CLAIMED) row again each time; its own startup reconciliation
        # from the *previous* crash runs before the new claim, so by the
        # time we kill it here the row already carries crash_count from
        # every prior round.
        for attempt in (2, 3):
            proc = spawn_controller(config_path, mode="resume")
            try:
                await poll_db_until(db_path, mid_submit_sql(attempt), timeout=20.0)
                rc = await sigkill_and_wait(proc, timeout=10.0)
            finally:
                await reap(proc)
            assert rc == -signal.SIGKILL

        # Final resume: startup reconciliation now finds crash_count>=2 and
        # must give up immediately rather than attempt a fourth submission.
        proc = spawn_controller(config_path, mode="resume")
        try:
            rc = await wait_for_exit(proc, timeout=30.0)
        finally:
            await reap(proc)
        assert rc == 0

        store = SQLiteStore(db_path)
        try:
            summary = await assert_all_terminal(store, 1)
            records = {r.asset.id: r for r in read_all_records(store)}
            rec = records["lone"]
            assert rec.crash_count >= 2, f"expected crash_count>=2, got {rec.crash_count}"
            assert rec.state == AssetState.SKIPPED_REVIEW, (
                f"expected SKIPPED_REVIEW after two crashes, got {rec.state}"
            )
            assert summary == {AssetState.SKIPPED_REVIEW: 1}
        finally:
            await store.close()
    finally:
        await fc.stop()
