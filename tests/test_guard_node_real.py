"""Faithful harness proving the #16312 guard node actually protects
`prompt_worker`, instead of just trusting that catching the exception in
`PromptExecutor.execute`/`execute_async` is enough.

The reviewer's concern: `comfyui_guard_node` sets `executor.success = False`
on a caught framework-level exception, but ComfyUI's real `prompt_worker`
(main.py) reads MORE than `.success` off the executor immediately afterwards:

    e.execute(item[2], prompt_id, extra_data, item[4])
    q.task_done(item_id, e.history_result, status=execution.PromptQueue.ExecutionStatus(
        status_str='success' if e.success else 'error',
        completed=e.success,
        messages=e.status_messages))

`e.history_result` in particular is never set by `PromptExecutor.reset()` --
only by the success path deep inside `execute_async`'s try body, AFTER
everything that can raise. If the guard only fixed `.success`, an
`AttributeError` (first prompt of a process) or a stale previous job's
outputs (any later prompt) reading `e.history_result` would fire in
`prompt_worker` itself -- outside the guard's own try/except, since that read
happens after the guarded call already returned -- and `prompt_worker`'s own
`except BaseException: ... raise` would kill the thread anyway. That would
make the guard worse than useless: a false sense of safety.

This file does NOT trust that reasoning by inspection. It builds:

  * `StubPromptExecutor` -- reduced to exactly the control-flow shape of the
    real `execution.PromptExecutor` that matters here: `reset()` sets
    `success`/`status_messages` but not `history_result`; `execute()` is the
    real sync shim (`asyncio.run(self.execute_async(...))`); `execute_async()`
    has the same three-phase shape (set up status_messages/`execution_start`
    -> "do work" -> assign `history_result` -> optional `execution_success`),
    with an injectable failure point before or after the `history_result`
    assignment; `add_message()` mirrors the real method (append to
    `status_messages`, and call `self.server.send_sync(...)` when a client_id
    is set). Verified against ComfyUI's own source
    (Comfy-Org/ComfyUI @ master, `main.py`/`execution.py`, fetched
    2026-09-25) -- see `comfyui_guard_node/__init__.py`'s module docstring
    for the exact excerpts this is built from.
  * `StubPromptQueue` -- reduced to the load-bearing shape of
    `execution.PromptQueue.task_done`: it MERGES `history_result` into the
    history entry via `.update(history_result)`, which is exactly the line
    that breaks if `history_result` is ever `None` or missing.
  * `run_prompt_worker` -- the trimmed control-flow shape of the real
    `prompt_worker` (GC/model-unload/asset-scan bookkeeping stripped out,
    since it's irrelevant to what this guard fixes): call `executor.execute`,
    then read `.history_result`/`.success`/`.status_messages` straight off
    the executor to build `ExecutionStatus` for `task_done`, inside the same
    `except BaseException: ... raise` shape that is what actually kills the
    real thread on an unhandled read.

The guard itself is patched onto `StubPromptExecutor` through the real
`comfyui_guard_node.patch_prompt_executor()` (by injecting a fake `execution`
module into `sys.modules`), not reimplemented -- these tests exercise the
actual shipped guard code.
"""

from __future__ import annotations

import asyncio
import queue
import sys
import threading
import time
import types
from typing import Any, NamedTuple

import pytest

import comfyui_guard_node

# --------------------------------------------------------------------- stand-ins


class ExecutionStatus(NamedTuple):
    """Verbatim shape of `execution.PromptQueue.ExecutionStatus`."""

    status_str: str
    completed: bool
    messages: list[Any]


class FailAt:
    NONE = "none"
    BEFORE_HISTORY_RESULT = "before_history_result"
    AFTER_HISTORY_RESULT = "after_history_result"


class StubServer:
    """Stand-in for ComfyUI's `PromptServer`: exposes exactly what
    `PromptExecutor.add_message`/the guard touch -- `.client_id`,
    `.last_node_id`, and a synchronous `.send_sync(event, data, sid)` (real
    ComfyUI's is thread-safe via `call_soon_threadsafe`; this one is
    synchronous, which is a stronger guarantee for the purpose of this test,
    not a weaker one)."""

    def __init__(self, client_id: "str | None" = "client-1") -> None:
        self.client_id = client_id
        self.last_node_id: "str | None" = None
        self.sent_events: list[tuple[str, dict, "str | None"]] = []

    def send_sync(self, event: str, data: dict, sid: "str | None" = None) -> None:
        self.sent_events.append((event, dict(data), sid))


def make_executor_class() -> type:
    """Returns a FRESH class object each call -- so patching one test's
    executor class can never leave `_comfyui_guard_node_patched` markers on a
    function object another test's class also happens to reference."""

    class StubPromptExecutor:
        """Reduced-but-faithful stand-in for `execution.PromptExecutor`. See
        module docstring for exactly which real behaviour this reproduces."""

        def __init__(self, server: StubServer, fail_at: str = FailAt.NONE) -> None:
            self.server = server
            self.fail_at = fail_at
            self.finally_ran = False
            self.reset()

        def reset(self) -> None:
            # Verbatim: real PromptExecutor.reset() sets exactly these two,
            # and deliberately does NOT set history_result.
            self.status_messages: list[Any] = []
            self.success = True

        def add_message(self, event: str, data: dict, broadcast: bool) -> None:
            data = {**data, "timestamp": int(time.time() * 1000)}
            self.status_messages.append((event, data))
            if self.server.client_id is not None or broadcast:
                self.server.send_sync(event, data, self.server.client_id)

        def execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None) -> None:
            # Verbatim: the real sync `execute` is a thin shim over execute_async.
            asyncio.run(self.execute_async(prompt, prompt_id, extra_data or {}, execute_outputs or []))

        async def execute_async(self, prompt, prompt_id, extra_data=None, execute_outputs=None) -> None:
            extra_data = extra_data or {}
            if "client_id" in extra_data:
                self.server.client_id = extra_data["client_id"]
            self.status_messages = []
            self.add_message("execution_start", {"prompt_id": prompt_id}, broadcast=False)

            try:
                if self.fail_at == FailAt.BEFORE_HISTORY_RESULT:
                    raise RuntimeError(f"boom before history_result: {prompt_id}")

                self.success = True
                # Same line, same place in the control flow, as the real
                # `self.history_result = {"outputs": ui_outputs, "meta": meta_outputs}`.
                self.history_result = {"outputs": {"9": {"images": [{"filename": f"{prompt_id}.png"}]}}, "meta": {}}

                if self.fail_at == FailAt.AFTER_HISTORY_RESULT:
                    raise RuntimeError(f"boom after history_result: {prompt_id}")

                self.add_message("execution_success", {"prompt_id": prompt_id}, broadcast=False)
            finally:
                self.finally_ran = True

    return StubPromptExecutor


class StubPromptQueue:
    """Reduced to the load-bearing shape of `execution.PromptQueue`: `get`,
    `get_flags`, and -- critically -- a `task_done` that merges
    `history_result` into the history entry exactly the way the real one
    does, via `dict.update`, which is the line that raises/misbehaves if
    `history_result` isn't a real dict."""

    def __init__(self) -> None:
        self.history: dict[str, dict] = {}
        self.currently_running: dict[int, tuple] = {}
        self._items: "queue.Queue" = queue.Queue()
        self._next_item_id = 0

    def put(self, prompt_id: str, prompt: dict, extra_data: "dict | None" = None, outputs: "list | None" = None) -> None:
        # ComfyUI's real queue item shape: (number, prompt_id, prompt, extra_data, outputs_to_execute).
        item = (0.0, prompt_id, prompt, extra_data or {}, outputs or [])
        item_id = self._next_item_id
        self._next_item_id += 1
        self._items.put((item, item_id))

    def get(self, timeout: float):
        try:
            return self._items.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_flags(self) -> dict:
        return {}

    def task_done(self, item_id, history_result, status: "ExecutionStatus | None") -> None:
        # Verbatim shape of the real PromptQueue.task_done (execution.py,
        # Comfy-Org/ComfyUI @ master, fetched 2026-09-25):
        prompt = self.currently_running.pop(item_id)
        status_dict = dict(status._asdict()) if status is not None else None
        self.history[prompt[1]] = {
            "prompt": prompt,
            "outputs": {},
            "status": status_dict,
        }
        self.history[prompt[1]].update(history_result)


def run_prompt_worker(
    q: StubPromptQueue,
    executor: Any,
    *,
    max_items: int,
    stop_after_idle_s: float = 0.3,
) -> tuple[threading.Thread, "list[BaseException]", "list[int]"]:
    """Trimmed to the load-bearing control-flow shape of ComfyUI's real
    `prompt_worker` (main.py, fetched 2026-09-25): call `e.execute(...)`,
    then read `e.history_result`/`e.success`/`e.status_messages` STRAIGHT OFF
    THE EXECUTOR to build `ExecutionStatus` for `task_done`. GC / model
    unload / asset-scan bookkeeping is irrelevant to what this guard fixes
    and is omitted. The `except BaseException: ... raise` shape is kept
    UNCHANGED -- that is exactly what turns "reading e.history_result raised"
    into "the worker thread dies" in the real bug, and is the thing this test
    proves does NOT happen once the guard is patched in.

    Runs the loop on a background thread and returns (thread, errors,
    drained_counter) so the caller can assert the thread is alive/dead and
    how many items were actually drained, without the test itself running on
    the thread whose survival is being tested.
    """
    errors: list[BaseException] = []
    drained = [0]

    def _worker() -> None:
        idle_deadline = time.monotonic() + stop_after_idle_s
        try:
            while drained[0] < max_items and time.monotonic() < idle_deadline:
                queue_item = q.get(timeout=0.05)
                if queue_item is None:
                    continue
                idle_deadline = time.monotonic() + stop_after_idle_s
                item, item_id = queue_item
                prompt_id = item[1]
                q.currently_running[item_id] = item

                executor.execute(item[2], prompt_id, item[3], item[4])

                q.task_done(
                    item_id,
                    executor.history_result,
                    status=ExecutionStatus(
                        status_str="success" if executor.success else "error",
                        completed=executor.success,
                        messages=executor.status_messages,
                    ),
                )
                drained[0] += 1
        except BaseException as exc:  # noqa: BLE001 - mirrors prompt_worker's `except BaseException: raise`
            errors.append(exc)
            raise

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, errors, drained


@pytest.fixture
def patched_executor_class(monkeypatch: pytest.MonkeyPatch) -> type:
    """Applies the REAL `comfyui_guard_node.patch_prompt_executor()` onto a
    fresh `StubPromptExecutor` class, by injecting a fake `execution` module
    -- exercising the actual shipped guard code, not a reimplementation of
    it."""
    executor_cls = make_executor_class()
    fake_execution = types.ModuleType("execution")
    fake_execution.PromptExecutor = executor_cls
    monkeypatch.setitem(sys.modules, "execution", fake_execution)

    patched = comfyui_guard_node.patch_prompt_executor()
    assert patched is True, "guard failed to patch the stub PromptExecutor"
    assert getattr(executor_cls.execute, comfyui_guard_node._PATCH_MARKER, False)
    return executor_cls


# ============================================================= control: bug is real


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_without_guard_a_pre_history_result_crash_kills_the_worker_thread():
    """Establishes the harness is faithful: WITHOUT the guard patched in, a
    framework-level exception raised before `history_result` is assigned
    propagates all the way out of `execute()`, and the worker thread dies
    without draining the rest of the queue -- reproducing #16312 exactly."""
    executor_cls = make_executor_class()
    server = StubServer()
    executor = executor_cls(server, fail_at=FailAt.BEFORE_HISTORY_RESULT)

    q = StubPromptQueue()
    for i in range(3):
        q.put(f"p{i}", {})

    thread, errors, drained = run_prompt_worker(q, executor, max_items=3, stop_after_idle_s=0.5)
    thread.join(timeout=5.0)

    assert not thread.is_alive(), "worker thread should have died (this is the bug, unguarded)"
    assert errors, "expected the raw RuntimeError to propagate and kill the thread"
    assert isinstance(errors[0], RuntimeError)
    assert drained[0] == 0, "the very first item should have killed the thread before task_done"
    assert q.history == {}, "nothing should have reached history without task_done ever completing"


# ============================================================= the guard, proven


@pytest.mark.parametrize("fail_at", [FailAt.BEFORE_HISTORY_RESULT, FailAt.AFTER_HISTORY_RESULT])
def test_guard_survives_a_single_crashing_job_and_drains_it(patched_executor_class, fail_at):
    server = StubServer()
    executor = patched_executor_class(server, fail_at=fail_at)

    q = StubPromptQueue()
    q.put("only-job", {})

    thread, errors, drained = run_prompt_worker(q, executor, max_items=1)
    thread.join(timeout=5.0)

    assert thread.is_alive() is False  # the worker function returns normally once max_items is drained
    assert not errors, f"worker thread must not raise, got {errors}"
    assert drained[0] == 1, "the crashing job must still be drained (task_done must run)"

    entry = q.history["only-job"]
    assert entry["status"]["status_str"] == "error"
    assert entry["status"]["completed"] is False
    # history_result was merged in without blowing up .update() -- and did
    # NOT silently carry forward pre-crash/stale outputs as if they were valid.
    assert entry["outputs"] == {}


def test_guard_keeps_the_worker_loop_alive_and_the_queue_draining_across_many_jobs(patched_executor_class):
    """The headline claim: a mix of healthy and crashing jobs (crashing both
    before and after history_result is assigned, interleaved with jobs that
    succeed normally) all reach a terminal history entry, and the worker
    thread survives the whole run -- proving the guard, not just a single
    lucky catch."""
    server = StubServer()

    plan = [
        ("ok-1", FailAt.NONE),
        ("crash-before-1", FailAt.BEFORE_HISTORY_RESULT),
        ("ok-2", FailAt.NONE),
        ("crash-after-1", FailAt.AFTER_HISTORY_RESULT),
        ("crash-before-2", FailAt.BEFORE_HISTORY_RESULT),
        ("ok-3", FailAt.NONE),
    ]

    # One executor instance per job, standing in for the single long-lived
    # PromptExecutor `prompt_worker` constructs once and reuses across every
    # prompt in the real process -- fail_at is set per-job on that same
    # instance to simulate "this particular prompt happens to hit the bug".
    executor = patched_executor_class(server, fail_at=FailAt.NONE)

    q = StubPromptQueue()
    for prompt_id, _ in plan:
        q.put(prompt_id, {})

    def _worker() -> None:
        for prompt_id, fail_at in plan:
            executor.fail_at = fail_at
            queue_item = q.get(timeout=2.0)
            assert queue_item is not None
            item, item_id = queue_item
            assert item[1] == prompt_id
            q.currently_running[item_id] = item
            executor.execute(item[2], prompt_id, item[3], item[4])
            q.task_done(
                item_id,
                executor.history_result,
                status=ExecutionStatus(
                    status_str="success" if executor.success else "error",
                    completed=executor.success,
                    messages=executor.status_messages,
                ),
            )

    errors: list[BaseException] = []

    def _guarded() -> None:
        try:
            _worker()
        except BaseException as exc:  # mirrors prompt_worker's except BaseException: raise
            errors.append(exc)
            raise

    thread = threading.Thread(target=_guarded, daemon=True)
    thread.start()
    thread.join(timeout=10.0)

    assert not thread.is_alive()
    assert not errors, f"worker thread must survive every job, got {errors}"
    assert set(q.history.keys()) == {pid for pid, _ in plan}, "every submitted job must reach a history entry"

    for prompt_id, fail_at in plan:
        status = q.history[prompt_id]["status"]
        if fail_at == FailAt.NONE:
            assert status["status_str"] == "success", prompt_id
            assert status["completed"] is True, prompt_id
        else:
            assert status["status_str"] == "error", f"{prompt_id}: {status}"
            assert status["completed"] is False, prompt_id


# ================================================== fast controller visibility


@pytest.mark.parametrize("fail_at", [FailAt.BEFORE_HISTORY_RESULT, FailAt.AFTER_HISTORY_RESULT])
def test_guard_emits_a_real_execution_error_event_not_a_silent_timeout(patched_executor_class, fail_at):
    """The reviewer's second concern: a caught framework crash must not be
    invisible to a controller polling for `execution_error` / reading
    `/history`. Proves the guard reaches `add_message`, which both appends an
    `execution_error` message AND emits it over the (stand-in) event bus --
    exactly like ComfyUI's own per-node error path, not a side channel."""
    server = StubServer(client_id="controller-client")
    executor = patched_executor_class(server, fail_at=fail_at)

    q = StubPromptQueue()
    q.put("crashy", {"client_id": "controller-client"})

    thread, errors, drained = run_prompt_worker(q, executor, max_items=1)
    thread.join(timeout=5.0)
    assert not errors and drained[0] == 1

    # 1. status_messages (what a later /history read sees) carries a real
    #    execution_error entry -- not just the "execution_start" ComfyUI
    #    itself always appends first.
    error_messages = [m for m in executor.status_messages if m[0] == "execution_error"]
    assert error_messages, "expected an execution_error message in status_messages"
    _, data = error_messages[0]
    assert data["prompt_id"] == "crashy"
    assert data["exception_type"] == "RuntimeError"
    assert "crashy" in data["exception_message"]

    # 2. It was also sent over the (stand-in) event bus immediately -- this
    #    is what lets a real controller's await_outcome resolve right away
    #    instead of waiting out its full timeout before falling back to a
    #    /history poll.
    sent_errors = [e for e in server.sent_events if e[0] == "execution_error"]
    assert sent_errors, "expected send_sync('execution_error', ...) to have been called"
    _, sent_data, sid = sent_errors[0]
    assert sid == "controller-client", "execution_error must target the submitting client (broadcast=False)"
    assert sent_data["prompt_id"] == "crashy"

    # 3. And it reached the actual /history-shaped entry task_done wrote.
    history_messages = q.history["crashy"]["status"]["messages"]
    assert any(m[0] == "execution_error" for m in history_messages)


def test_guard_history_result_is_always_a_real_dict_even_on_first_ever_prompt(patched_executor_class):
    """Specifically targets the AttributeError scenario: a brand-new
    executor (as `prompt_worker` constructs exactly once per process) that
    crashes on its very FIRST prompt, before `history_result` has ever been
    assigned even once. Without the guard's fix, `e.history_result` doesn't
    exist as an attribute at all at this point, and `prompt_worker` reading
    it raises AttributeError."""
    server = StubServer()
    executor = patched_executor_class(server, fail_at=FailAt.BEFORE_HISTORY_RESULT)
    assert not hasattr(executor, "history_result"), "test setup: must not exist before the very first execute()"

    q = StubPromptQueue()
    q.put("first-ever", {})

    thread, errors, drained = run_prompt_worker(q, executor, max_items=1)
    thread.join(timeout=5.0)

    assert not errors
    assert drained[0] == 1
    assert hasattr(executor, "history_result")
    assert isinstance(executor.history_result, dict)
    assert q.history["first-ever"]["status"]["status_str"] == "error"
