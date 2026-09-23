"""SQLite adapter for StorePort.

The DB is the sole source of truth (req #8): every decision the caller makes —
what to claim next, what needs reconciliation, whether an asset has crashed
too many times — is answered by reading this file, never by parsing logs.

Correctness bar: a `kill -9` at any instant must resume with every asset
reaching exactly one terminal state, with no duplicate generation and no lost
asset. The design leans on three things to get there:
  * WAL + synchronous=NORMAL, so a committed transaction survives a process
    kill (only an OS/power-loss event could lose it, which is out of scope);
  * a single persistent connection, so SQLite's single-writer rule is
    trivially respected without us fighting SQLITE_BUSY under our own load;
  * conditional UPDATEs (`WHERE state IN (...)`) as the actual lock, so even
    if that single-connection assumption is ever relaxed, two racing claims
    still can't both win.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path

from ..models import (
    Asset,
    AssetRecord,
    AssetState,
    ExecutionError,
    QCVerdict,
)

# Forward-only migrations: each entry is applied exactly once, in order, the
# first time a DB is opened at a version below it. Kept minimal but real —
# no down-migrations, no branching, just "get an old file to the current
# shape."
_MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE assets (
            id                 TEXT PRIMARY KEY,
            asset_json         TEXT NOT NULL,
            state              TEXT NOT NULL,
            attempt            INTEGER NOT NULL DEFAULT 0,
            job_key            TEXT,
            comfy_prompt_id    TEXT UNIQUE,   -- SQLite: multiple NULLs are not a conflict
            output_paths_json  TEXT NOT NULL DEFAULT '[]',
            last_error_json    TEXT,
            last_verdict_json  TEXT,
            crash_count        INTEGER NOT NULL DEFAULT 0,
            retry_after        REAL,
            -- Set by needing_reconciliation() on the rows it hands back, and
            -- cleared by transition(). Lets transition() tell "this row is
            -- completing normally after claim->submit" apart from "this row
            -- is being finalized after a crash found it mid-flight" —
            -- exactly the distinction req #5's crash_count needs, since both
            -- cases otherwise look identical (a CLAIMED/SUBMITTED row being
            -- transitioned out).
            reconciling        INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX idx_assets_state ON assets(state);

        -- Append-only: one row per attempt transition. This is the
        -- failure-rate table; nothing reads the process logs to compute it.
        CREATE TABLE run_log (
            log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_id        TEXT NOT NULL REFERENCES assets(id),
            attempt         INTEGER NOT NULL,
            ts              REAL NOT NULL,
            state           TEXT NOT NULL,
            prompt_id       TEXT,
            error_type      TEXT,
            verdict_summary TEXT,
            duration        REAL
        );
        CREATE INDEX idx_run_log_asset ON run_log(asset_id);
        """,
    ),
]


def _migrate(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
    row = conn.execute("SELECT version FROM schema_version").fetchone()
    current = row[0] if row else 0
    for version, sql in _MIGRATIONS:
        if version > current:
            conn.executescript(sql)
            current = version
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (current,))
    elif current != row[0]:
        conn.execute("UPDATE schema_version SET version=?", (current,))


class SQLiteStore:
    """Implements StorePort. Async methods are thin `asyncio.to_thread`
    wrappers around sync sqlite3 calls, serialized through `self._lock`."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        # Real single-writer enforcement: every public op takes this before
        # touching self._conn, whether it reads or writes. Cheap here (local
        # file, short critical sections) and it sidesteps SQLITE_BUSY retries
        # entirely rather than relying on busy_timeout to paper over races
        # between threads in *our own* process.
        self._lock = threading.Lock()
        self._conn = self._connect()
        with self._lock:
            _migrate(self._conn)

    def _connect(self) -> sqlite3.Connection:
        # isolation_level=None -> autocommit; we drive BEGIN IMMEDIATE/COMMIT/
        # ROLLBACK explicitly so the conditional UPDATEs below are exactly the
        # transactions req #2/#3 describe, not whatever implicit transaction
        # sqlite3's default wrapping would pick.
        conn = sqlite3.connect(self._db_path, check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        # Pragmas are per-connection, not persisted in the DB file, so they
        # must be reissued every time we open one -- including the fresh
        # connection built after a simulated crash.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    async def close(self) -> None:
        await asyncio.to_thread(self._conn.close)

    # ------------------------------------------------------------- mapping

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(
            asset=Asset.model_validate_json(row["asset_json"]),
            state=AssetState(row["state"]),
            attempt=row["attempt"],
            job_key=row["job_key"],
            comfy_prompt_id=row["comfy_prompt_id"],
            output_paths=json.loads(row["output_paths_json"]),
            last_error=(
                ExecutionError.model_validate_json(row["last_error_json"])
                if row["last_error_json"] is not None
                else None
            ),
            last_verdict=(
                QCVerdict.model_validate_json(row["last_verdict_json"])
                if row["last_verdict_json"] is not None
                else None
            ),
            crash_count=row["crash_count"],
        )

    def _log_run(
        self,
        asset_id: str,
        attempt: int,
        state: AssetState,
        *,
        prompt_id: str | None,
        error_type: str | None,
        verdict_summary: str | None,
        duration: float | None,
    ) -> None:
        self._conn.execute(
            """INSERT INTO run_log
                 (asset_id, attempt, ts, state, prompt_id, error_type, verdict_summary, duration)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (asset_id, attempt, time.time(), state.value, prompt_id, error_type, verdict_summary, duration),
        )

    # --------------------------------------------------------------- StorePort

    async def load_batch(self, assets: list[Asset]) -> None:
        await asyncio.to_thread(self._load_batch_sync, assets)

    def _load_batch_sync(self, assets: list[Asset]) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                for asset in assets:
                    # Idempotent: reloading the same manifest after a restart
                    # must not reset progress on assets already in flight.
                    self._conn.execute(
                        """INSERT INTO assets (id, asset_json, state)
                           VALUES (?, ?, ?)
                           ON CONFLICT(id) DO NOTHING""",
                        (asset.id, asset.model_dump_json(), AssetState.PENDING.value),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    async def claim_next(self) -> AssetRecord | None:
        return await asyncio.to_thread(self._claim_next_sync)

    def _claim_next_sync(self) -> AssetRecord | None:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute(
                    """SELECT * FROM assets
                       WHERE state IN ('pending', 'retry_wait')
                         AND (retry_after IS NULL OR retry_after <= ?)
                       ORDER BY id
                       LIMIT 1""",
                    (now,),
                ).fetchone()
                if row is None:
                    self._conn.execute("COMMIT")
                    return None

                asset_id = row["id"]
                # Opaque per-attempt claim token. Not the deterministic
                # content-hash from Asset.job_key(model_version, graph_hash)
                # -- the store has neither of those -- just a unique value
                # written atomically with the claim so record_submission and
                # reconciliation have something to correlate against.
                job_key = uuid.uuid4().hex

                # The WHERE clause here -- not the BEGIN IMMEDIATE lock -- is
                # what req #2 actually requires: even a second connection
                # racing us with the identical statement must affect 0 rows.
                cur = self._conn.execute(
                    """UPDATE assets SET state='claimed', attempt=attempt+1, job_key=?
                       WHERE id=? AND state IN ('pending', 'retry_wait')
                         AND (retry_after IS NULL OR retry_after <= ?)""",
                    (job_key, asset_id, now),
                )
                if cur.rowcount == 0:
                    # Lost a race between the SELECT and the UPDATE.
                    self._conn.execute("COMMIT")
                    return None

                new_row = self._conn.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
                self._log_run(
                    asset_id, new_row["attempt"], AssetState.CLAIMED,
                    prompt_id=None, error_type=None, verdict_summary=None, duration=None,
                )
                self._conn.execute("COMMIT")
                return self._row_to_record(new_row)
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    async def record_submission(self, asset_id: str, prompt_id: str) -> None:
        await asyncio.to_thread(self._record_submission_sync, asset_id, prompt_id)

    def _record_submission_sync(self, asset_id: str, prompt_id: str) -> None:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                # This commit is separate from claim_next's on purpose (req
                # #3). The gap between them is the one genuinely ambiguous
                # window in the whole system: if the process dies after
                # POSTing /prompt but before this UPDATE lands, ComfyUI may
                # already be running the job with no durable record of its
                # prompt_id here. That's exactly why needing_reconciliation()
                # exists -- it hands the still-CLAIMED row back so the caller
                # can probe ComfyUI's /history via the job_key-derived
                # filename_prefix rather than the store ever guessing.
                cur = self._conn.execute(
                    "UPDATE assets SET state='submitted', comfy_prompt_id=? WHERE id=? AND state='claimed'",
                    (prompt_id, asset_id),
                )
                if cur.rowcount:
                    row = self._conn.execute(
                        "SELECT attempt FROM assets WHERE id=?", (asset_id,)
                    ).fetchone()
                    self._log_run(
                        asset_id, row["attempt"], AssetState.SUBMITTED,
                        prompt_id=prompt_id, error_type=None, verdict_summary=None, duration=None,
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    async def transition(
        self,
        asset_id: str,
        state: AssetState,
        *,
        error: ExecutionError | None = None,
        verdict: QCVerdict | None = None,
        output_paths: list[str] | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        await asyncio.to_thread(
            self._transition_sync, asset_id, state, error, verdict, output_paths, retry_after_s
        )

    def _transition_sync(
        self,
        asset_id: str,
        state: AssetState,
        error: ExecutionError | None,
        verdict: QCVerdict | None,
        output_paths: list[str] | None,
        retry_after_s: float | None,
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT * FROM assets WHERE id=?", (asset_id,)).fetchone()
                if row is None:
                    self._conn.execute("ROLLBACK")
                    raise KeyError(f"unknown asset id: {asset_id!r}")

                prior_state = AssetState(row["state"])
                crash_count = row["crash_count"]
                # req #5: only count it as a crash when this row was actually
                # flagged by needing_reconciliation() -- i.e. we are
                # finalizing a claim/submission left over from a restart, not
                # completing one that ran start-to-finish this session.
                if row["reconciling"] and prior_state in (AssetState.CLAIMED, AssetState.SUBMITTED):
                    crash_count += 1

                retry_after = (now + retry_after_s) if retry_after_s is not None else None
                error_json = error.model_dump_json() if error is not None else None
                verdict_json = verdict.model_dump_json() if verdict is not None else None
                paths_json = json.dumps(
                    output_paths if output_paths is not None else json.loads(row["output_paths_json"])
                )

                self._conn.execute(
                    """UPDATE assets
                       SET state=?, last_error_json=?, last_verdict_json=?,
                           output_paths_json=?, retry_after=?, crash_count=?, reconciling=0
                       WHERE id=?""",
                    (state.value, error_json, verdict_json, paths_json, retry_after, crash_count, asset_id),
                )

                verdict_summary = None
                if verdict is not None:
                    verdict_summary = "pass" if verdict.passed else "fail:" + ",".join(verdict.failed_checks)

                self._log_run(
                    asset_id, row["attempt"], state,
                    prompt_id=row["comfy_prompt_id"],
                    error_type=(error.exception_type if error is not None else None),
                    verdict_summary=verdict_summary,
                    duration=None,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    async def needing_reconciliation(self) -> list[AssetRecord]:
        return await asyncio.to_thread(self._needing_reconciliation_sync)

    def _needing_reconciliation_sync(self) -> list[AssetRecord]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                rows = self._conn.execute(
                    "SELECT * FROM assets WHERE state IN ('claimed', 'submitted')"
                ).fetchall()
                if rows:
                    self._conn.executemany(
                        "UPDATE assets SET reconciling=1 WHERE id=?",
                        [(r["id"],) for r in rows],
                    )
                self._conn.execute("COMMIT")
                # Never silently assume success or failure here -- just hand
                # back exactly what's ambiguous and let the caller verify
                # against ComfyUI (req #4).
                return [self._row_to_record(r) for r in rows]
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    async def summary(self) -> dict[AssetState, int]:
        return await asyncio.to_thread(self._summary_sync)

    def _summary_sync(self) -> dict[AssetState, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT state, COUNT(*) AS c FROM assets GROUP BY state"
            ).fetchall()
        return {AssetState(r["state"]): r["c"] for r in rows}
