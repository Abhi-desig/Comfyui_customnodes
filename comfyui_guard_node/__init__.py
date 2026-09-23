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
"""

from __future__ import annotations

import logging

logger = logging.getLogger("comfyui_guard_node")

# Sentinel attribute marking a callable as already wrapped by us -- both to
# skip a function we've already patched (double-patch guard) and to let the
# post-patch verification step tell "our wrapper" apart from anything else.
_PATCH_MARKER = "_comfyui_guard_node_patched"


def _mark_prompt_failed_and_log(executor: object, method_name: str) -> None:
    logger.error(
        "comfyui_guard_node: caught a framework-level exception in "
        "PromptExecutor.%s -- marking the prompt failed instead of letting "
        "it kill prompt_worker (see https://github.com/comfyanonymous/ComfyUI/issues/16312).",
        method_name,
        exc_info=True,
    )
    # Best-effort: nudge the executor's own bookkeeping so it doesn't believe
    # this prompt is still "running". Never let this itself raise.
    try:
        executor.success = False  # type: ignore[attr-defined]
    except Exception:
        pass


def _make_sync_wrapper(method_name: str, original):
    def wrapper(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except Exception:
            _mark_prompt_failed_and_log(self, method_name)
            return None

    return wrapper


def _make_async_wrapper(method_name: str, original):
    async def wrapper(self, *args, **kwargs):
        try:
            return await original(self, *args, **kwargs)
        except Exception:
            _mark_prompt_failed_and_log(self, method_name)
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
