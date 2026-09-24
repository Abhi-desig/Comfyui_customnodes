"""Tests for the SQLite StorePort adapter.

Two layers:
  * plain unit tests for the specific mechanical guarantees called out in the
    design (pragmas, the conditional-UPDATE lock, retry gating, run_log);
  * a Hypothesis RuleBasedStateMachine that throws randomized sequences of
    claim/submit/transition/crash at the store and checks the invariants that
    actually define correctness here: no asset in two states, no shared
    comfy_prompt_id, claimed-but-unsubmitted rows always recoverable, and
    every asset terminal by the end.
"""

from __future__ import annotations

import asyncio
import gc
import sqlite3
import time
import uuid
from pathlib import Path

from hypothesis import HealthCheck, settings
from hypothesis.stateful import RuleBasedStateMachine, Bundle, consumes, invariant, multiple, rule
from hypothesis import strategies as st

import pytest

from comfy_controller.adapters.sqlite_store import SQLiteStore, SubmissionNotRecorded
from comfy_controller.models import Asset, AssetState, ExecutionError, QCVerdict, CheckResult


def run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------- unit tests


def test_pragmas_applied_on_every_connection(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    conn = store._conn
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL == 1
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000

    # Reopening (simulating a restart) must reapply them -- they are per
    # connection, not stored in the file.
    run(store.close())
    store2 = SQLiteStore(tmp_path / "s.db")
    conn2 = store2._conn
    assert conn2.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    assert conn2.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_double_claim_returns_none(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    run(store.load_batch([Asset(id="x", workflow="w.json")]))

    rec1 = run(store.claim_next())
    assert rec1 is not None
    assert rec1.state == AssetState.CLAIMED
    assert rec1.attempt == 1
    assert rec1.job_key is not None

    rec2 = run(store.claim_next())
    assert rec2 is None


def test_conditional_update_affects_zero_rows_once_claimed(tmp_path):
    """Exercises the exact UPDATE from the design doc directly, on a second
    raw connection, to show the WHERE clause -- not just our in-process lock
    -- is what prevents a double claim."""
    db_path = tmp_path / "s.db"
    store = SQLiteStore(db_path)
    run(store.load_batch([Asset(id="x", workflow="w.json")]))
    run(store.claim_next())

    other = sqlite3.connect(db_path)
    try:
        cur = other.execute(
            """UPDATE assets SET state='claimed', attempt=attempt+1, job_key=?
               WHERE id=? AND state IN ('pending', 'retry_wait')
                 AND (retry_after IS NULL OR retry_after <= ?)""",
            ("intruder-token", "x", time.time()),
        )
        assert cur.rowcount == 0
    finally:
        other.close()


def test_retry_after_gating(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    run(store.load_batch([Asset(id="x", workflow="w.json")]))
    rec = run(store.claim_next())
    run(store.transition(rec.asset.id, AssetState.RETRY_WAIT, retry_after_s=10))

    # Not due yet.
    assert run(store.claim_next()) is None

    # Move the clock back (without touching pragmas/migrations) to simulate
    # the retry becoming due.
    store._conn.execute("UPDATE assets SET retry_after=? WHERE id='x'", (time.time() - 1,))
    rec2 = run(store.claim_next())
    assert rec2 is not None
    assert rec2.asset.id == "x"
    assert rec2.attempt == 2


def test_run_log_appends_one_row_per_transition(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    run(store.load_batch([Asset(id="x", workflow="w.json")]))
    run(store.claim_next())
    run(store.record_submission("x", "prompt-1"))
    run(store.transition("x", AssetState.SAVED, output_paths=["o.png"]))

    rows = store._conn.execute(
        "SELECT state, prompt_id FROM run_log WHERE asset_id='x' ORDER BY log_id"
    ).fetchall()
    assert [r["state"] for r in rows] == ["claimed", "submitted", "saved"]
    assert rows[1]["prompt_id"] == "prompt-1"
    assert rows[2]["prompt_id"] == "prompt-1"  # carried forward for audit


def test_needing_reconciliation_and_crash_count(tmp_path):
    db_path = tmp_path / "s.db"
    store = SQLiteStore(db_path)
    run(store.load_batch([Asset(id="x", workflow="w.json")]))
    run(store.claim_next())

    # Simulate `kill -9`: drop the store, reopen a fresh one on the same file.
    run(store.close())
    del store
    gc.collect()
    store2 = SQLiteStore(db_path)

    pending = run(store2.needing_reconciliation())
    assert [r.asset.id for r in pending] == ["x"]
    assert pending[0].state == AssetState.CLAIMED

    # Caller checked ComfyUI, found nothing, schedules a retry.
    run(store2.transition("x", AssetState.RETRY_WAIT, retry_after_s=0))
    rec = run(store2.claim_next())
    assert rec is not None
    assert rec.crash_count == 1  # this attempt was a reconciled restart

    # A normal claim -> submit -> terminal in the *same* session must NOT
    # bump crash_count -- it never went through needing_reconciliation().
    run(store2.record_submission("x", "prompt-2"))
    run(store2.transition("x", AssetState.SAVED, output_paths=["o.png"]))
    final = store2._conn.execute("SELECT crash_count, state FROM assets WHERE id='x'").fetchone()
    assert final["crash_count"] == 1
    assert final["state"] == "saved"


def test_no_two_assets_share_prompt_id(tmp_path):
    store = SQLiteStore(tmp_path / "s.db")
    run(store.load_batch([Asset(id="a", workflow="w.json"), Asset(id="b", workflow="w.json")]))
    run(store.claim_next())
    run(store.claim_next())
    run(store.record_submission("a", "shared-id"))
    try:
        run(store.record_submission("b", "shared-id"))
        raised = False
    except sqlite3.IntegrityError:
        raised = True
    assert raised, "UNIQUE constraint on comfy_prompt_id must reject a duplicate"


# ------------------------------------------------------------------ stateful test

ASSET_IDS = ["a0", "a1", "a2"]
TERMINAL_STATES = [
    AssetState.SAVED,
    AssetState.PARKED_APPROVAL,
    AssetState.JUDGE_UNAVAILABLE,
    AssetState.SKIPPED_REVIEW,
    AssetState.PERMANENT_ERROR,
]


def _sample_verdict(passed: bool) -> QCVerdict:
    return QCVerdict(checks=[CheckResult(name="check", evidence="e", passed=passed)])


def _sample_error() -> ExecutionError:
    return ExecutionError(prompt_id="p", exception_type="RuntimeError", exception_message="boom")


class StoreMachine(RuleBasedStateMachine):
    """Bundles track *sequencing* only (which asset id to act on next) --
    invariants below always re-read the real DB rather than trust bundle
    bookkeeping, since a crash_and_restart can resolve a bundle's item behind
    its back (exactly as a real reconciliation pass would)."""

    claimed = Bundle("claimed")      # holds asset_id
    submitted = Bundle("submitted")  # holds (asset_id, prompt_id)

    def __init__(self) -> None:
        super().__init__()
        import tempfile

        self._tmpdir = tempfile.mkdtemp()
        self.db_path = Path(self._tmpdir) / "store.db"
        self.store = SQLiteStore(self.db_path)
        self.epoch = 0
        run(self.store.load_batch([Asset(id=i, workflow="w.json") for i in ASSET_IDS]))

    # ------------------------------------------------------------------ rules

    @rule(target=claimed)
    def do_claim(self):
        rec = run(self.store.claim_next())
        if rec is None:
            return multiple()
        assert rec.state == AssetState.CLAIMED
        # Tagged with the current epoch: a crash_and_restart resolves every
        # in-flight row behind the bundle's back, which makes any id minted
        # before it stale.
        return (rec.asset.id, self.epoch)

    @rule(target=submitted, item=consumes(claimed))
    def do_submit(self, item):
        aid, _epoch = item
        prompt_id = f"p-{uuid.uuid4().hex}"
        # A bundle entry can go stale behind our back: crash_and_restart
        # reconciles in-flight rows to RETRY_WAIT, and the asset may since
        # have been re-claimed under a *different* entry. So assert against
        # the row's real state rather than predicting it -- recording a
        # prompt is legal only from claimed/submitted, and the store must
        # refuse loudly anywhere else rather than drop it silently.
        row = self.store._conn.execute(
            "SELECT state FROM assets WHERE id=?", (aid,)
        ).fetchone()
        if row["state"] not in ("claimed", "submitted"):
            with pytest.raises(SubmissionNotRecorded):
                run(self.store.record_submission(aid, prompt_id))
            return multiple()
        run(self.store.record_submission(aid, prompt_id))
        return (aid, prompt_id)

    @rule(item=consumes(submitted), term_state=st.sampled_from(TERMINAL_STATES))
    def do_finish(self, item, term_state):
        aid, _prompt_id = item
        kwargs: dict = {}
        if term_state == AssetState.SAVED:
            kwargs["output_paths"] = [f"{aid}.png"]
            kwargs["verdict"] = _sample_verdict(True)
        elif term_state == AssetState.PERMANENT_ERROR:
            kwargs["error"] = _sample_error()
        elif term_state == AssetState.JUDGE_UNAVAILABLE:
            pass
        elif term_state in (AssetState.PARKED_APPROVAL, AssetState.SKIPPED_REVIEW):
            kwargs["verdict"] = _sample_verdict(False)
        run(self.store.transition(aid, term_state, **kwargs))

    @rule(item=consumes(claimed))
    def do_retry_from_claimed(self, item):
        aid, _epoch = item
        run(self.store.transition(aid, AssetState.RETRY_WAIT, retry_after_s=0))

    @rule(item=consumes(submitted))
    def do_retry_from_submitted(self, item):
        aid, _prompt_id = item
        run(self.store.transition(aid, AssetState.RETRY_WAIT, retry_after_s=0))

    @rule()
    def crash_and_restart(self):
        # Simulate `kill -9`: nothing is flushed deliberately, the object is
        # just dropped and a fresh one opened against the same file.
        run(self.store.close())
        del self.store
        gc.collect()
        self.store = SQLiteStore(self.db_path)
        self.epoch += 1
        pending = run(self.store.needing_reconciliation())
        for rec in pending:
            # Stand-in for "checked ComfyUI, couldn't confirm either way" --
            # the store must never decide this on its own (req #4), so the
            # test plays the caller's role here explicitly.
            run(self.store.transition(rec.asset.id, AssetState.RETRY_WAIT, retry_after_s=0))

    # ------------------------------------------------------------- invariants

    @invariant()
    def every_asset_appears_exactly_once(self):
        rows = self.store._conn.execute("SELECT id FROM assets").fetchall()
        ids = [r["id"] for r in rows]
        assert sorted(ids) == sorted(ASSET_IDS)
        assert len(ids) == len(set(ids))  # PRIMARY KEY already guarantees this

    @invariant()
    def states_are_valid(self):
        rows = self.store._conn.execute("SELECT state FROM assets").fetchall()
        for r in rows:
            AssetState(r["state"])  # raises ValueError if it isn't a real state

    @invariant()
    def no_shared_prompt_ids(self):
        rows = self.store._conn.execute(
            """SELECT comfy_prompt_id, COUNT(*) AS c FROM assets
               WHERE comfy_prompt_id IS NOT NULL
               GROUP BY comfy_prompt_id HAVING c > 1"""
        ).fetchall()
        assert rows == []

    @invariant()
    def claimed_rows_are_recoverable(self):
        # A CLAIMED/SUBMITTED row must carry everything reconciliation needs:
        # a job_key from the claim, and an Asset that still deserializes.
        rows = self.store._conn.execute(
            "SELECT * FROM assets WHERE state IN ('claimed', 'submitted')"
        ).fetchall()
        for r in rows:
            assert r["job_key"] is not None
            Asset.model_validate_json(r["asset_json"])  # must not raise
            if r["state"] == "submitted":
                assert r["comfy_prompt_id"] is not None

    def teardown(self):
        # Drain whatever the random sequence left behind so we can assert the
        # end-to-end guarantee: every asset reaches exactly one terminal
        # state. Mirrors what the real caller's reconciliation + retry loop
        # would eventually do.
        for _ in range(4 * len(ASSET_IDS) + 10):
            pending = run(self.store.needing_reconciliation())
            for rec in pending:
                run(self.store.transition(rec.asset.id, AssetState.RETRY_WAIT, retry_after_s=0))

            rec = run(self.store.claim_next())
            if rec is None:
                summary = run(self.store.summary())
                if all(AssetState(s).is_terminal for s in summary):
                    break
                continue
            run(self.store.record_submission(rec.asset.id, f"drain-{uuid.uuid4().hex}"))
            run(self.store.transition(rec.asset.id, AssetState.SAVED, output_paths=["done.png"]))

        summary = run(self.store.summary())
        for state, count in summary.items():
            assert state.is_terminal, f"{state} still has {count} non-terminal row(s) at teardown"
        assert sum(summary.values()) == len(ASSET_IDS)

        run(self.store.close())


StoreMachine.TestCase.settings = settings(
    max_examples=25,
    stateful_step_count=40,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    deadline=None,
)
TestStoreMachine = StoreMachine.TestCase
