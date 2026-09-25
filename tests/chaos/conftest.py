"""Shared fixtures/helpers for the chaos acceptance suite.

Two test doubles live here that the individual chaos scenarios build on:

* `RoutingComfy` -- a `ComfyPort` that scripts a deterministic, per-asset
  outcome for any graph carrying a `__chaos_id__` marker (baked into that
  asset's own workflow JSON by `tagged_workflow()`), while transparently
  forwarding anything else -- in practice, only `HealthSupervisor`'s own
  untagged canary graph -- to a real `FakeComfy` + `ComfyHTTP` pair. This is
  what lets one single `BatchRunner.run()` call exercise many *simultaneous*
  and *independent* fault behaviours (OOM here, a permanent error there, a
  healthy partition elsewhere) even though `FakeComfyState.fault` is only a
  single global switch -- the scripted assets never touch that switch at
  all, and the real pair stays free for scenarios (zombie detection) that
  really do need genuine WS/HTTP dynamics.

* `RoutingJudge` -- a `JudgePort` that scripts a per-asset-id sequence of
  verdicts/exceptions (unlike `FakeJudge`, whose single shared queue can't
  be pinned to a particular asset under concurrency).

Also provides `assert_all_terminal()`, the headline invariant every chaos
scenario must satisfy, and `unique_prompt_ids()`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import sqlite3
import subprocess
import sys
import time as _time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml

from comfy_controller.adapters.comfy_http import ComfyHTTP
from comfy_controller.adapters.sqlite_store import SQLiteStore
from comfy_controller.models import (
    Asset,
    AssetState,
    CheckResult,
    ExecutionError,
    JobOutcome,
    QCVerdict,
)
from comfy_controller.ports import ComfyPort
from comfy_controller.testing.fake_comfy import FakeComfy, FakeComfyState

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _default_verdict() -> QCVerdict:
    return QCVerdict(checks=[CheckResult(name="ok", evidence="routing judge default pass", passed=True)])


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: real wall-clock waits and/or a genuine subprocess kill -- slower than the unit suite"
    )


# --------------------------------------------------------------------- workflows


def tagged_workflow(tmp_path: Path, chaos_id: str, *, name: str | None = None) -> str:
    """A minimal API-format graph, distinguishable to `RoutingComfy` by its
    `__chaos_id__` marker. Several `Asset`s may point at the *same* file
    (e.g. every asset in one circuit-breaker partition) when their scripted
    behaviour is identical and asset-agnostic; give assets that need their
    own independently-tracked attempt counter (e.g. "fail once, then
    succeed") their own file instead.
    """
    graph: dict[str, Any] = {
        "1": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20, "denoise": 1.0}},
        "9": {"class_type": "SaveImage", "inputs": {}},
        "__chaos_id__": chaos_id,
    }
    path = tmp_path / (name or f"wf_{chaos_id}.json")
    path.write_text(json.dumps(graph))
    return str(path)


# --------------------------------------------------------------------- RoutingComfy


@dataclass
class ScriptedOutcome:
    succeeded: bool
    output_paths: list[str] = field(default_factory=list)
    exception_type: str = ""
    exception_message: str = ""


ScriptFn = Callable[[int], ScriptedOutcome]

# Smallest valid PNG: 1x1, fully transparent -- decodable by Pillow. Same
# bytes `FakeComfyState`'s own `_write_output` writes for the real (non-
# scripted) path; duplicated here rather than imported so this module doesn't
# reach into `fake_comfy`'s private constant.
_SCRIPTED_OUTPUT_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


def fail_once_then_succeed(exception_type: str, exception_message: str) -> ScriptFn:
    """attempt 0 (first submission) fails; every later attempt succeeds --
    e.g. an OOM that clears itself on retry."""

    def fn(attempt: int) -> ScriptedOutcome:
        if attempt == 0:
            return ScriptedOutcome(succeeded=False, exception_type=exception_type, exception_message=exception_message)
        return ScriptedOutcome(succeeded=True, output_paths=[f"out_{attempt}.png"])

    return fn


def always_fail(exception_type: str, exception_message: str) -> ScriptFn:
    def fn(attempt: int) -> ScriptedOutcome:
        return ScriptedOutcome(succeeded=False, exception_type=exception_type, exception_message=exception_message)

    return fn


def always_succeed() -> ScriptFn:
    def fn(attempt: int) -> ScriptedOutcome:
        return ScriptedOutcome(succeeded=True, output_paths=[f"out_{attempt}.png"])

    return fn


class RoutingComfy:
    """Implements `ComfyPort`. See module docstring.

    `output_dir`, when given, makes every SCRIPTED "succeeded" outcome back
    its claimed `output_paths` with a real, decodable PNG on disk -- not just
    a filename string. Without this, turning `RunnerConfig.comfy_output_dir`
    on for a run built from `RoutingComfy` would make `_unreadable_outputs`'s
    real filesystem check fail every scripted asset (the file genuinely
    doesn't exist), and leaving it off (the previous state of this suite)
    means the judge/prefilter read-the-file path never actually runs for any
    asset here -- exactly the gap that let a real bug ship unnoticed.
    """

    def __init__(self, real: ComfyPort, scripts: dict[str, ScriptFn], *, output_dir: Path | None = None):
        self._real = real
        self._scripts = scripts
        self._attempts: dict[str, int] = {}
        self._pending: dict[str, tuple[str, int]] = {}
        self._n = 0
        self._output_dir = output_dir

    @staticmethod
    def _chaos_id(graph: dict[str, Any]) -> str | None:
        value = graph.get("__chaos_id__")
        return value if isinstance(value, str) else None

    async def submit(self, graph: dict[str, Any], job_key: str, front: bool = False) -> str:
        chaos_id = self._chaos_id(graph)
        if chaos_id is None or chaos_id not in self._scripts:
            return await self._real.submit(graph, job_key, front=front)

        self._n += 1
        prompt_id = f"scripted-{chaos_id}-{self._n}"
        attempt = self._attempts.get(chaos_id, 0)
        self._attempts[chaos_id] = attempt + 1
        self._pending[prompt_id] = (chaos_id, attempt)
        return prompt_id

    async def events(self):
        async for evt in self._real.events():
            yield evt

    async def await_outcome(self, prompt_id: str, timeout: float) -> JobOutcome:
        pending = self._pending.pop(prompt_id, None)
        if pending is None:
            return await self._real.await_outcome(prompt_id, timeout)

        chaos_id, attempt = pending
        outcome = self._scripts[chaos_id](attempt)
        if outcome.succeeded:
            self._write_scripted_outputs(outcome.output_paths)
            return JobOutcome(prompt_id=prompt_id, succeeded=True, output_paths=outcome.output_paths)
        return JobOutcome(
            prompt_id=prompt_id,
            succeeded=False,
            error=ExecutionError(
                prompt_id=prompt_id,
                exception_type=outcome.exception_type,
                exception_message=outcome.exception_message,
            ),
        )

    def _write_scripted_outputs(self, output_paths: list[str]) -> None:
        """No-op when `output_dir` isn't configured (most existing chaos
        scenarios don't care about real bytes on disk). When it IS
        configured, every path a scripted "succeeded" outcome claims gets a
        real, Pillow-decodable PNG underneath it, so
        `BatchRunner._judge_or_park`'s `resolve_output_paths` +
        `_unreadable_outputs` check, and the judge/prefilter's own
        `Image.open`, exercise genuine bytes instead of passing vacuously."""
        if self._output_dir is None:
            return
        for rel in output_paths:
            path = self._output_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.write_bytes(_SCRIPTED_OUTPUT_PNG)

    async def queue_depth(self) -> tuple[int, int]:
        return await self._real.queue_depth()

    async def history(self, prompt_id: str):
        if prompt_id.startswith("scripted-"):
            return None
        return await self._real.history(prompt_id)

    async def is_responding(self) -> bool:
        return await self._real.is_responding()

    async def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        await self._real.free(unload_models=unload_models, free_memory=free_memory)

    async def interrupt(self, prompt_id: str | None = None) -> None:
        await self._real.interrupt(prompt_id)


# --------------------------------------------------------------------- RoutingJudge


class RoutingJudge:
    """Implements `JudgePort`. Plays back a scripted sequence of verdicts (or
    `JudgeUnavailable`) *per asset id*, unlike `FakeJudge`'s single shared
    queue which concurrent, interleaved calls from different assets would
    scramble. Once an asset's scripted sequence is down to one entry, that
    last entry repeats forever -- so "always fails" / "always unavailable"
    only needs one queued item.
    """

    def __init__(self) -> None:
        self._scripts: dict[str, deque[QCVerdict | Exception]] = {}
        self.calls: list[tuple[str, list[Path]]] = []

    def script(self, asset_id: str, *outcomes: QCVerdict | Exception) -> None:
        assert outcomes, "script() needs at least one outcome"
        self._scripts[asset_id] = deque(outcomes)

    async def judge(self, asset: Asset, image_paths: list[Path]) -> QCVerdict:
        self.calls.append((asset.id, list(image_paths)))
        seq = self._scripts.get(asset.id)
        if not seq:
            return _default_verdict()
        outcome = seq.popleft() if len(seq) > 1 else seq[0]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --------------------------------------------------------------------- comfy lifecycle


@pytest.fixture
async def real_comfy_pair():
    """A real FakeComfy server + the genuine ComfyHTTP adapter talking to it
    -- for whatever in a chaos scenario needs authentic WS/HTTP dynamics
    (the supervisor's canary, a genuine zombie)."""
    fc = FakeComfy(FakeComfyState())
    base = await fc.start()
    client = ComfyHTTP(base, client_id="chaos-test")
    try:
        yield fc, client
    finally:
        await client.close()
        await fc.stop()


# --------------------------------------------------------------------- invariant helpers


async def assert_all_terminal(store: SQLiteStore, expected_total: int) -> dict[AssetState, int]:
    """The headline invariant: every submitted asset lands in exactly one
    terminal state, the terminal count equals the submitted count, and no
    two assets share a `comfy_prompt_id`."""
    summary = await store.summary()
    total = sum(summary.values())
    assert total == expected_total, f"expected {expected_total} assets, store has {total}: {summary}"
    non_terminal = {s: c for s, c in summary.items() if not s.is_terminal and c}
    assert not non_terminal, f"asset(s) stuck non-terminal: {non_terminal}"

    dupes = store._conn.execute(
        "SELECT comfy_prompt_id, COUNT(*) c FROM assets "
        "WHERE comfy_prompt_id IS NOT NULL GROUP BY comfy_prompt_id HAVING c > 1"
    ).fetchall()
    assert list(dupes) == [], f"duplicate comfy_prompt_id(s): {list(dupes)}"
    return summary


# --------------------------------------------------------------------- kill/resume


def start_output_writing_comfy(output_dir: Path, state: FakeComfyState | None = None) -> FakeComfy:
    """A `FakeComfy` whose successful jobs also write a (tiny, real) output
    file to `output_dir` -- FakeComfy itself only ever fabricates the
    *reference* to an output path, never the file, which is fine for
    in-process tests but not for kill/resume: a real controller subprocess
    is going to be asked whether a `SAVED` asset's output file genuinely
    exists, and this is what makes that question answerable. Must be called
    *before* `await fc.start()` -- the instance attribute has to shadow the
    class method before `start()` registers the aiohttp route.
    """
    fc = FakeComfy(state)
    original_execute = fc._execute

    async def _execute_and_persist(job: Any) -> None:
        await original_execute(job)
        entry = fc._history.get(job.prompt_id)
        if not entry or entry.get("status", {}).get("status_str") != "success":
            return
        for node_output in entry.get("outputs", {}).values():
            for image in node_output.get("images", []) or []:
                subfolder = image.get("subfolder") or ""
                filename = image.get("filename", "")
                target = output_dir / subfolder / filename if subfolder else output_dir / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"not-a-real-png")

    fc._execute = _execute_and_persist  # type: ignore[method-assign]
    return fc


def slow_down_prompt_endpoint(fc: FakeComfy, delay_s: float) -> None:
    """Widen the CLAIMED-but-not-yet-SUBMITTED window (the "mid-submit" kill
    checkpoint) by delaying `/prompt`'s response. Must be called before
    `await fc.start()`, same reason as `start_output_writing_comfy`."""
    original_post_prompt = fc._post_prompt

    async def _delayed_post_prompt(request: Any) -> Any:
        import asyncio

        await asyncio.sleep(delay_s)
        return await original_post_prompt(request)

    fc._post_prompt = _delayed_post_prompt  # type: ignore[method-assign]


def write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data))


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    write_yaml(path, {"assets": rows})


def _poll_db_until_blocking(
    db_path: Path, sql: str, *, timeout: float = 15.0, poll_interval: float = 0.02
) -> None:
    """Poll `db_path` -- via a *separate* read connection; WAL mode lets a
    reader and the controller subprocess's own writer connection coexist --
    until `sql` (a `SELECT` whose first column is truthy on at least one row)
    is satisfied. Real `time.sleep`, deliberately: this is coordinating with
    a genuine OS subprocess, not something an injected clock can stand in
    for. Raises `TimeoutError` (with the last error/row seen) if the
    condition never becomes true -- never silently gives up.

    NEVER call this straight from a coroutine: see `poll_db_until`.
    """
    deadline = _time.monotonic() + timeout
    last_seen: Any = None
    while _time.monotonic() < deadline:
        if db_path.exists():
            try:
                conn = sqlite3.connect(str(db_path))
                try:
                    row = conn.execute(sql).fetchone()
                    last_seen = row
                    if row and row[0]:
                        return
                finally:
                    conn.close()
            except sqlite3.Error as exc:
                last_seen = exc
        _time.sleep(poll_interval)
    raise TimeoutError(f"condition never true within {timeout}s: {sql!r} (last seen: {last_seen!r})")


# Every one of the helpers below waits on *wall clock* -- a polling loop, a
# `waitpid`. The `FakeComfy` these tests spawn their controller subprocesses
# against is an aiohttp server running on the test's OWN event loop, so any
# such wait performed inline in a coroutine freezes ComfyUI for its whole
# duration: the subprocess under test finds a socket that accepts but never
# answers, its submit budget expires, and the resulting `TimeoutError` gets
# (correctly) classified TEMPORARY and backed off -- a fault entirely
# manufactured by the harness. Pushing the wait onto a worker thread keeps
# the loop -- and therefore ComfyUI -- live throughout.


async def poll_db_until(db_path: Path, sql: str, *, timeout: float = 15.0, poll_interval: float = 0.02) -> None:
    await asyncio.to_thread(
        _poll_db_until_blocking, db_path, sql, timeout=timeout, poll_interval=poll_interval
    )


def spawn_controller(
    config_path: Path, manifest_path: Path | None = None, *, mode: str = "run"
) -> subprocess.Popen:
    """A genuine `python -m comfy_controller.cli` subprocess -- not an
    in-process call -- so a `SIGKILL` against it is a real process death
    (severed sockets, no chance for `finally`/`atexit` cleanup), which is
    the entire point of this file versus `tests/test_runner.py`'s simulated
    crash."""
    args = [sys.executable, "-m", "comfy_controller.cli", mode, "--config", str(config_path)]
    if manifest_path is not None:
        args.append(str(manifest_path))
    return subprocess.Popen(args, cwd=str(_REPO_ROOT))


async def sigkill_and_wait(proc: subprocess.Popen, *, timeout: float = 10.0) -> int:
    import os
    import signal

    os.kill(proc.pid, signal.SIGKILL)
    return await wait_for_exit(proc, timeout=timeout)


async def wait_for_exit(proc: subprocess.Popen, *, timeout: float) -> int:
    """`proc.wait(timeout)` without stalling the event loop (see above)."""
    return await asyncio.to_thread(proc.wait, timeout)


async def reap(proc: subprocess.Popen) -> None:
    """Make sure a spawned controller can never outlive its test -- an
    orphan keeps hammering a `FakeComfy` the teardown is about to close,
    and its log lines land in whatever test runs next."""
    if proc.poll() is None:
        proc.kill()
        with contextlib.suppress(Exception):
            await wait_for_exit(proc, timeout=10.0)
