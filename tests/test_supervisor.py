"""Tests for HealthSupervisor and the comfyui_guard_node monkeypatch.

Driven against FakeComfy through the real ComfyHTTP adapter (owned by another
agent, only ever imported here -- never modified) so these tests exercise the
supervisor exactly as it will run in production, not a hand-rolled stand-in.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import types

from comfy_controller.adapters.comfy_http import ComfyHTTP
from comfy_controller.models import HealthVerdict
from comfy_controller.supervisor import HealthSupervisor
from comfy_controller.testing.fake_comfy import Fault, FakeComfy, FakeComfyState


async def _make_pair(state: FakeComfyState | None = None) -> tuple[FakeComfy, ComfyHTTP]:
    fc = FakeComfy(state)
    base = await fc.start()
    client = ComfyHTTP(base, client_id="supervisor-test")
    return fc, client


async def _teardown(fc: FakeComfy, client: ComfyHTTP, sup: HealthSupervisor | None = None) -> None:
    if sup is not None:
        await sup.aclose()
    await client.close()
    await fc.stop()


# ------------------------------------------------------------------- assess()


async def test_zombie_is_detected_via_front_running_canary():
    """#16312 in miniature: HTTP stays 200 and the queue keeps accepting jobs
    while the worker thread is dead. Only a front=True canary that never
    reaches `executed` can tell that apart from a merely quiet server."""
    fc, client = await _make_pair(FakeComfyState(fault=Fault.ZOMBIE, fault_after_n_jobs=0))
    sup = HealthSupervisor(
        client,
        stall_timeout=0.05,
        canary_timeout=0.3,
        gpu_util_fn=lambda: 1.0,  # idle -> escalate straight to canary, not BUSY
        cpu_util_fn=lambda: 1.0,
        launch_argv=None,
    )
    try:
        await asyncio.sleep(0.1)  # let WS silence exceed stall_timeout
        verdict = await sup.assess()
        assert verdict is HealthVerdict.ZOMBIE
    finally:
        await _teardown(fc, client, sup)


async def test_slow_job_stays_healthy_or_busy_and_is_never_restarted():
    """The single most important behaviour: a legitimately long-running job
    must NEVER be restarted just because it goes quiet between progress
    ticks. hysteresis_n=1 (maximally trigger-happy) makes this a strong
    check -- even with zero tolerance for a second bad sample, nothing here
    should ever count as a bad sample in the first place."""
    fc, client = await _make_pair(FakeComfyState(fault=Fault.SLOW, fault_after_n_jobs=0, slow_seconds=0.3))
    sup = HealthSupervisor(
        client,
        stall_timeout=0.08,
        canary_timeout=0.2,
        hysteresis_n=1,
        gpu_util_fn=lambda: 95.0,  # pegged: any silence reads as BUSY, never SUSPECT
        cpu_util_fn=lambda: 95.0,
        launch_argv=None,
    )
    restart_calls: list[str] = []
    real_restart = sup.restart

    async def spy_restart(reason: str) -> bool:
        restart_calls.append(reason)
        return await real_restart(reason)

    sup.restart = spy_restart  # type: ignore[method-assign]
    try:
        await client.submit({}, "slow-job", front=False)
        verdicts = []
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline:
            verdicts.append(await sup.tick())
            await asyncio.sleep(0.03)
        assert restart_calls == []
        assert all(v in (HealthVerdict.HEALTHY, HealthVerdict.BUSY) for v in verdicts)
    finally:
        await _teardown(fc, client, sup)


async def test_hysteresis_requires_two_consecutive_bad_assessments():
    fc, client = await _make_pair(FakeComfyState(fault=Fault.ZOMBIE, fault_after_n_jobs=0))
    sup = HealthSupervisor(
        client,
        stall_timeout=0.05,
        canary_timeout=0.2,
        hysteresis_n=2,
        gpu_util_fn=lambda: 1.0,
        cpu_util_fn=lambda: 1.0,
        launch_argv=None,
    )
    try:
        await asyncio.sleep(0.1)

        v1 = await sup.tick()
        assert v1 is HealthVerdict.ZOMBIE
        assert sup.restart_count == 0  # one bad sample: never act alone

        v2 = await sup.tick()
        assert v2 is HealthVerdict.ZOMBIE
        assert sup.restart_count == 1  # second confirming sample: now act
    finally:
        await _teardown(fc, client, sup)


# ------------------------------------------------------------ restart discipline


async def test_backoff_grows_exponentially_and_caps(tmp_path):
    delays: list[float] = []

    async def fake_sleep(d: float) -> None:
        delays.append(d)

    fc, client = await _make_pair()
    sup = HealthSupervisor(
        client,
        launch_argv=None,
        sleep=fake_sleep,
        backoff_base=10.0,
        backoff_cap=300.0,
        intensity_max_restarts=100,
        marker_dir=tmp_path,
    )
    try:
        for _ in range(6):
            assert await sup.restart("test") is True
        assert delays == [10.0, 20.0, 40.0, 80.0, 160.0, 300.0]
    finally:
        await _teardown(fc, client, sup)


async def test_intensity_ceiling_holds_and_stops_restarting(tmp_path):
    async def fake_sleep(_d: float) -> None:
        return None

    fc, client = await _make_pair()
    sup = HealthSupervisor(
        client,
        launch_argv=None,
        sleep=fake_sleep,
        intensity_max_restarts=3,
        intensity_window=600.0,
        marker_dir=tmp_path,
    )
    try:
        results = [await sup.restart("x") for _ in range(4)]
        assert results == [True, True, True, False]
        assert sup.held is True
        # Once held, it stays held -- must not quietly resume on its own.
        assert await sup.restart("x") is False
    finally:
        await _teardown(fc, client, sup)


async def test_maybe_recycle_fires_every_n_jobs(tmp_path):
    async def fake_sleep(_d: float) -> None:
        return None

    fc, client = await _make_pair()
    sup = HealthSupervisor(
        client, launch_argv=None, sleep=fake_sleep, recycle_every_n_jobs=40, marker_dir=tmp_path
    )
    try:
        assert await sup.maybe_recycle(39) is False
        assert await sup.maybe_recycle(40) is True
        assert sup.restart_count == 1
    finally:
        await _teardown(fc, client, sup)


# -------------------------------------------------------------------- guard node


def _stub_execution_module(*, with_async: bool = True) -> tuple[types.ModuleType, dict[str, int]]:
    calls = {"execute": 0, "execute_async": 0}

    class PromptExecutor:
        def execute(self, *a, **k):
            calls["execute"] += 1
            return "ok"

        if with_async:

            async def execute_async(self, *a, **k):
                calls["execute_async"] += 1
                return "ok-async"

    mod = types.ModuleType("execution")
    mod.PromptExecutor = PromptExecutor
    return mod, calls


def test_guard_node_patch_confirmed_and_wrapper_transparent(monkeypatch, caplog):
    mod, calls = _stub_execution_module()
    monkeypatch.setitem(sys.modules, "execution", mod)
    sys.modules.pop("comfyui_guard_node", None)  # force a fresh, observable import
    caplog.set_level(logging.CRITICAL, logger="comfyui_guard_node")

    import comfyui_guard_node as guard

    assert getattr(mod.PromptExecutor.execute, guard._PATCH_MARKER) is True
    assert getattr(mod.PromptExecutor.execute_async, guard._PATCH_MARKER) is True
    assert any("PATCH CONFIRMED" in r.getMessage() for r in caplog.records)

    instance = mod.PromptExecutor()
    assert instance.execute() == "ok"  # wrapper still calls through
    assert calls["execute"] == 1
    assert guard.NODE_CLASS_MAPPINGS == {}
    assert guard.NODE_DISPLAY_NAME_MAPPINGS == {}


def test_guard_node_double_patch_is_a_noop(monkeypatch):
    mod, _calls = _stub_execution_module()
    monkeypatch.setitem(sys.modules, "execution", mod)
    sys.modules.pop("comfyui_guard_node", None)

    import comfyui_guard_node as guard

    first_wrapper = mod.PromptExecutor.execute
    assert guard.patch_prompt_executor() is True  # patch again by hand
    assert mod.PromptExecutor.execute is first_wrapper  # not re-wrapped


def test_guard_node_missing_execution_module_warns_but_does_not_raise(monkeypatch, caplog):
    monkeypatch.delitem(sys.modules, "execution", raising=False)
    sys.modules.pop("comfyui_guard_node", None)
    caplog.set_level(logging.WARNING, logger="comfyui_guard_node")

    import comfyui_guard_node as guard  # must not raise even with no `execution` module

    assert guard.NODE_CLASS_MAPPINGS == {}
    assert any("could not import" in r.getMessage() for r in caplog.records)


def test_guard_node_wrapper_survives_a_framework_level_exception(monkeypatch):
    mod, _calls = _stub_execution_module(with_async=False)

    class Boom:
        def execute(self, *a, **k):
            raise RuntimeError("framework-level explosion")

    mod.PromptExecutor = Boom
    monkeypatch.setitem(sys.modules, "execution", mod)
    sys.modules.pop("comfyui_guard_node", None)

    import comfyui_guard_node  # noqa: F401  (import performs the patch)

    instance = mod.PromptExecutor()
    result = instance.execute()  # must NOT raise -- this is the whole point of #16312
    assert result is None
    assert instance.success is False
