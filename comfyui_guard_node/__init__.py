"""ComfyUI Guard Node -- mitigation for issue #16312.

Dropped into ComfyUI's `custom_nodes/`. At import time it monkey-patches
`execution.PromptExecutor.execute` (and `.execute_async`, on ComfyUI versions
that have it) so a framework-level exception marks the current prompt failed
instead of propagating out of `prompt_worker` and killing that background
thread -- which is what #16312 actually is: the HTTP server keeps answering
200, `/prompt` keeps accepting jobs, and nothing ever executes again, with no
crash and no log to grep for.

Precedent: Comfy Org's own comfyui-datadog-monitor patches exactly these two
methods for exactly this reason.

Defensive by design: if `execution` can't be imported, or the expected
attribute is missing (upstream rename), this logs a loud warning and gets out
of the way -- it must never prevent ComfyUI from starting. Because a silent
regression here is precisely the failure mode it exists to prevent, a
successful patch is confirmed with an explicit, loud check right after
patching, not assumed.

Two things this guard MUST also do, verified against ComfyUI's own source
(`main.py`'s `prompt_worker`, `execution.py`'s `PromptExecutor`,
Comfy-Org/ComfyUI @ master, fetched 2026-09-25) rather than assumed --
see `tests/test_guard_node_real.py` for the harness that proves both:

  1. `prompt_worker` reads `e.history_result` straight off the executor
     immediately after `e.execute(...)` returns, unconditionally, to hand to
     `PromptQueue.task_done()`:

         e.execute(item[2], prompt_id, extra_data, item[4])
         q.task_done(item_id, e.history_result, status=...)

     `PromptExecutor.reset()` sets `self.success` and `self.status_messages`
     but NEVER sets `self.history_result` -- that attribute is only assigned
     near the very end of `execute_async`'s try body (on the success path,
     after every line that can raise). Catching the exception and setting
     `executor.success = False` is therefore not enough: on the FIRST prompt
     of a process, `e.history_result` doesn't exist as an attribute at all,
     so reading it in `prompt_worker` raises `AttributeError` -- outside
     anything this guard's `try/except` around `execute`/`execute_async` can
     catch, since that read happens in `prompt_worker` itself, after our
     wrapper has already returned. `prompt_worker`'s own
     `except BaseException: ... raise` then kills the thread anyway -- the
     guard would have been worse than useless. On any LATER prompt, the
     missing assignment instead silently carries a PREVIOUS job's outputs
     forward, misattributed to this failed one. Fixed by always setting
     `executor.history_result` to a real (empty) dict when we catch anything.

  2. A caught framework-level exception never reaches
     `PromptExecutor.handle_execution_error` (that's only called from inside
     the node-execution loop this guard's catch sits outside of), so no
     `execution_error` WS event is ever sent the way an ordinary per-node
     failure's is. Left alone, that turns a fast failure into a SILENT wait
     for the controller's full `await_outcome` timeout (up to its configured
     30 minutes) before it falls back to polling `/history/{prompt_id}` on
     its own. Fixed by calling the executor's own `add_message(...)` --
     exactly the method ComfyUI's real error path calls, which both appends
     the message to `status_messages` (so a `/history` read sees it) and
     emits it over the real event bus via `self.server.send_sync(...)`
     (confirmed thread-safe -- `self.loop.call_soon_threadsafe(...)` --
     exactly like every other event `prompt_worker`'s own thread sends). This
     is not reaching around ComfyUI internals from an unexpected context: it
     is the same method, called from the same thread, ComfyUI's own code
     already calls it from. Falls back to writing the message into
     `status_messages` by hand if `add_message` is missing or itself raises,
     so at least a `/history` read still sees a real error.
"""

from __future__ import annotations

import logging
import time
import traceback as _traceback

logger = logging.getLogger("comfyui_guard_node")

# Sentinel attribute marking a callable as already wrapped by us -- both to
# skip a function we've already patched (double-patch guard) and to let the
# post-patch verification step tell "our wrapper" apart from anything else.
_PATCH_MARKER = "_comfyui_guard_node_patched"


def _extract_prompt_id(call_args: tuple, call_kwargs: dict) -> "str | None":
    """`execute`/`execute_async`'s real signature is
    `(self, prompt, prompt_id, extra_data=..., execute_outputs=...)`.
    `call_args`/`call_kwargs` here are what the wrapper received AFTER
    `self` (see `_make_sync_wrapper`/`_make_async_wrapper` below), matching
    how `prompt_worker` actually calls it: `e.execute(item[2], prompt_id,
    extra_data, item[4])` -- positionally, so `prompt_id` is `call_args[1]`.
    Never raises; returns None (an honest "don't know") if the shape doesn't
    match, e.g. an upstream signature change.
    """
    try:
        if "prompt_id" in call_kwargs:
            return call_kwargs["prompt_id"]
        if len(call_args) >= 2:
            return call_args[1]
    except Exception:
        pass
    return None


def _extract_client_id(executor: object) -> "str | None":
    """Best-effort: `execute_async` sets `self.server.client_id` from
    `extra_data` before its try block, so by the time a framework-level
    exception fires inside that try (the case this guard exists for), it is
    already the right value for the in-flight prompt. Never raises."""
    try:
        return getattr(getattr(executor, "server", None), "client_id", None)
    except Exception:
        return None


def _extract_last_node_id(executor: object) -> "str | None":
    """Best-effort hint at which node was in flight when the crash happened.
    None ("unknown") is an honest answer when this isn't available -- this is
    a diagnostic nicety, never load-bearing. Never raises."""
    try:
        return getattr(getattr(executor, "server", None), "last_node_id", None)
    except Exception:
        return None


def _ensure_history_result(executor: object) -> None:
    """See module docstring, point 1. Always OVERWRITE with a real, empty
    result -- never leave a stale previous job's outputs misattributed to
    this failed one, and never leave the attribute missing entirely."""
    executor.history_result = {"outputs": {}, "meta": {}}  # type: ignore[attr-defined]


def _build_execution_error_payload(
    executor: object, exc: BaseException, call_args: tuple, call_kwargs: dict
) -> dict:
    return {
        "prompt_id": _extract_prompt_id(call_args, call_kwargs),
        "node_id": _extract_last_node_id(executor),
        "node_type": None,
        "executed": [],
        "exception_message": str(exc),
        "exception_type": type(exc).__name__,
        "traceback": _traceback.format_exception(type(exc), exc, exc.__traceback__),
        "current_inputs": {},
        "current_outputs": [],
    }


def _record_failure_for_controller(
    executor: object, method_name: str, exc: BaseException, call_args: tuple, call_kwargs: dict
) -> None:
    """See module docstring, point 2. Tries the real `add_message` path
    first (message text + real WS emission in one call, exactly like
    ComfyUI's own error path); falls back to a raw `status_messages` append
    if that method is missing or itself raises. Never propagates."""
    payload = _build_execution_error_payload(executor, exc, call_args, call_kwargs)

    add_message = getattr(executor, "add_message", None)
    if callable(add_message):
        try:
            add_message("execution_error", payload, broadcast=False)
            return
        except Exception:
            logger.warning(
                "comfyui_guard_node: PromptExecutor.add_message raised while "
                "reporting a %s failure -- falling back to a raw history entry.",
                method_name,
            )

    # Fallback: add_message missing/renamed, or it raised. At minimum, make
    # sure a later /history read (the controller's own timeout-path check,
    # or its reconnect-time reconciliation) finds a real error message
    # instead of nothing.
    try:
        messages = getattr(executor, "status_messages", None)
        if not isinstance(messages, list):
            messages = []
        messages.append(("execution_error", {**payload, "timestamp": int(time.time() * 1000)}))
        executor.status_messages = messages  # type: ignore[attr-defined]
    except Exception:
        pass  # never let failure *reporting* become a second failure

    # add_message wasn't available/usable -- still attempt a direct,
    # best-effort WS emission so a live controller isn't left waiting purely
    # because of that. Thread-safe the same way ComfyUI's own send_sync is
    # (call_soon_threadsafe internally); never raise if it isn't there.
    try:
        server = getattr(executor, "server", None)
        send_sync = getattr(server, "send_sync", None)
        if callable(send_sync):
            send_sync("execution_error", payload, _extract_client_id(executor))
    except Exception:
        pass


def _mark_prompt_failed_and_log(
    executor: object, method_name: str, exc: BaseException, call_args: tuple, call_kwargs: dict
) -> None:
    logger.error(
        "comfyui_guard_node: caught a framework-level exception in "
        "PromptExecutor.%s -- marking the prompt failed instead of letting "
        "it kill prompt_worker (see https://github.com/comfyanonymous/ComfyUI/issues/16312).",
        method_name,
        exc_info=exc,
    )
    # Best-effort, in this order. Every step is wrapped separately so one
    # step's failure can never suppress the next -- this whole function must
    # never itself raise.
    try:
        executor.success = False  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        _ensure_history_result(executor)
    except Exception:
        pass
    try:
        _record_failure_for_controller(executor, method_name, exc, call_args, call_kwargs)
    except Exception:
        pass


def _make_sync_wrapper(method_name: str, original):
    def wrapper(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception as exc:
            _mark_prompt_failed_and_log(self, method_name, exc, args, kwargs)
            return None

    return wrapper


def _make_async_wrapper(method_name: str, original):
    async def wrapper(self, *args, **kwargs):
        try:
            return await original(self, *args, **kwargs)
        except Exception as exc:
            _mark_prompt_failed_and_log(self, method_name, exc, args, kwargs)
            return None

    return wrapper


_WRAPPERS = {
    "execute": _make_sync_wrapper,
    "execute_async": _make_async_wrapper,
}


def patch_prompt_executor() -> bool:
    """Patch `execution.PromptExecutor`. Returns True iff the patch is
    confirmed active afterwards. Never raises -- every failure mode is a
    logged warning, because a broken guard must not become a broken ComfyUI.
    """
    try:
        import execution  # ComfyUI's own module; not ours to control.
    except Exception as exc:
        logger.warning(
            "comfyui_guard_node: could not import ComfyUI's `execution` module "
            "(%s) -- the #16312 zombie-worker guard is NOT installed.",
            exc,
        )
        return False

    executor_cls = getattr(execution, "PromptExecutor", None)
    if executor_cls is None:
        logger.warning(
            "comfyui_guard_node: `execution.PromptExecutor` not found (upstream "
            "rename?) -- the #16312 zombie-worker guard is NOT installed."
        )
        return False

    patched_any = False
    for method_name, make_wrapper in _WRAPPERS.items():
        original = getattr(executor_cls, method_name, None)
        if original is None:
            continue  # e.g. execute_async simply doesn't exist on this version

        if getattr(original, _PATCH_MARKER, False):
            # Already ours (module re-imported, custom_nodes reloaded, etc.)
            # -- guard against double-wrapping instead of nesting try/excepts.
            logger.info(
                "comfyui_guard_node: PromptExecutor.%s already patched, skipping.",
                method_name,
            )
            patched_any = True
            continue

        wrapper = make_wrapper(method_name, original)
        setattr(wrapper, _PATCH_MARKER, True)
        wrapper.__name__ = getattr(original, "__name__", method_name)
        wrapper.__wrapped__ = original
        setattr(executor_cls, method_name, wrapper)
        patched_any = True

    if not patched_any:
        logger.warning(
            "comfyui_guard_node: neither `execute` nor `execute_async` was found "
            "on PromptExecutor -- the #16312 zombie-worker guard is NOT installed."
        )
        return False

    # Loud, explicit confirmation the patch actually took. An upstream rename
    # or a shadowing re-import could otherwise silently regress us back to
    # the zombie state with no error anywhere -- so we verify the live
    # attribute, we don't just trust that setattr() above worked.
    current_execute = getattr(executor_cls, "execute", None)
    if getattr(current_execute, _PATCH_MARKER, False):
        logger.critical(
            "comfyui_guard_node: PATCH CONFIRMED -- PromptExecutor.execute is "
            "now guarded against framework-level exceptions (#16312 mitigation active)."
        )
        return True

    logger.warning(
        "comfyui_guard_node: patch verification FAILED -- PromptExecutor.execute "
        "does not carry our marker after patching. The #16312 guard may NOT be active."
    )
    return False


# Patch at import time, exactly once per process (barring an explicit re-call).
patch_prompt_executor()

# This package contributes no nodes of its own -- it exists purely for its
# import-time side effect above.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
