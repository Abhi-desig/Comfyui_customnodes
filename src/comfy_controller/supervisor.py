"""HealthSupervisor: owns ComfyUI process lifecycle (satisfies `SupervisorPort`).

Context (issue #16312): a framework-level exception can kill ComfyUI's
background `prompt_worker` thread while the HTTP server keeps answering 200
and `/prompt` keeps accepting jobs into the queue. Nothing ever executes
again. `/system_stats` and even `/queue` stay "fine" throughout, so neither
is a usable health signal on its own -- this module triangulates three
independent signals instead:

  1. WS `progress`/`executing` traffic  -> the worker is doing real work.
  2. GPU/CPU utilisation                -> disambiguates "silent because
     legitimately busy" from "silent because dead".
  3. A `front=True` canary prompt       -> the only signal that can't lie:
     if a trivial prompt that jumps the queue never reaches `executed`,
     the worker is not consuming the queue at all.

Restart discipline exists because restart loops make VRAM fragmentation
worse, not better: hysteresis (never act on one sample), capped exponential
backoff, and a hard intensity ceiling that hands the problem to a human.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

import structlog

from .models import HealthVerdict
from .ports import ComfyPort

logger = structlog.get_logger(__name__)

# A minimal API-format graph. Content is irrelevant -- the canary only exists
# to prove the queue is being consumed, so it must be as cheap as possible.
_CANARY_GRAPH: dict[str, Any] = {"1": {"class_type": "_supervisor_canary", "inputs": {}}}

# WS event types that count as "the worker is alive and doing something".
_PROGRESS_EVENTS = frozenset({"progress", "executing"})
# WS event types that resolve a canary (or an ordinary job) one way or another.
_TERMINAL_EVENTS = frozenset(
    {"executed", "execution_success", "execution_error", "execution_interrupted"}
)


def _nvidia_smi_gpu_util() -> float | None:
    """Default GPU reader. Degrades to None when nvidia-smi is absent (dev
    laptop, CPU-only box, etc.) -- callers must treat None as 'unknown',
    never as 'idle', since silently assuming idle would restart a busy GPU."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
        first_line = out.stdout.strip().splitlines()[0]
        return float(first_line)
    except Exception:
        return None


class HealthSupervisor:
    """Implements `SupervisorPort` against a single `ComfyPort` adapter.

    Only depends on the `ComfyPort` protocol -- never on the concrete HTTP
    adapter -- so it can be driven against `FakeComfy` or any other adapter.
    """

    def __init__(
        self,
        comfy: ComfyPort,
        *,
        stall_timeout: float = 600.0,
        canary_timeout: float = 30.0,
        hard_ceiling: float = 2700.0,
        gpu_idle_threshold: float = 10.0,
        cpu_idle_threshold: float = 5.0,
        gpu_util_fn: Callable[[], float | None] | None = _nvidia_smi_gpu_util,
        cpu_util_fn: Callable[[], float | None] | None = None,
        hysteresis_n: int = 2,
        backoff_base: float = 10.0,
        backoff_cap: float = 300.0,
        intensity_max_restarts: int = 5,
        intensity_window: float = 600.0,
        recycle_every_n_jobs: int = 40,
        launch_argv: Sequence[str] | None = None,
        cache_none: bool = True,
        disable_dynamic_vram: bool = True,
        extra_launch_args: Sequence[str] = (),
        marker_dir: str | Path = ".comfy_supervisor",
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.comfy = comfy

        self.stall_timeout = stall_timeout
        self.canary_timeout = canary_timeout
        self.hard_ceiling = hard_ceiling
        self.gpu_idle_threshold = gpu_idle_threshold
        self.cpu_idle_threshold = cpu_idle_threshold
        self.gpu_util_fn = gpu_util_fn
        self.cpu_util_fn = cpu_util_fn

        self.hysteresis_n = hysteresis_n
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.intensity_max_restarts = intensity_max_restarts
        self.intensity_window = intensity_window
        self.recycle_every_n_jobs = recycle_every_n_jobs

        self.launch_argv = list(launch_argv) if launch_argv is not None else None
        self.cache_none = cache_none
        self.disable_dynamic_vram = disable_dynamic_vram
        self.extra_launch_args = tuple(extra_launch_args)
        self.marker_dir = Path(marker_dir)

        self._clock = clock
        self._sleep = sleep

        self._last_progress_at = clock()
        self._pump_task: asyncio.Task[None] | None = None
        self._prompt_waiters: dict[str, asyncio.Event] = {}
        self._prompt_events: dict[str, list[str]] = {}

        self._bad_streak = 0
        self._restart_times: list[float] = []
        self._held = False
        self._restart_lock = asyncio.Lock()
        self._proc: asyncio.subprocess.Process | None = None

    # ---------------------------------------------------------- introspection

    @property
    def held(self) -> bool:
        """True once the intensity ceiling has been hit. Stays True until
        `clear_hold()` -- a human decided the underlying problem is fixed."""
        return self._held

    @property
    def restart_count(self) -> int:
        return len(self._restart_times)

    def clear_hold(self) -> None:
        self._held = False
        self._restart_times.clear()

    async def aclose(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            try:
                await self._pump_task
            except (asyncio.CancelledError, Exception):
                pass

    # -------------------------------------------------------------- assess()

    async def assess(self) -> HealthVerdict:
        self._ensure_pump_started()

        if not await self._safe_is_responding():
            return HealthVerdict.DEAD

        silence = self._clock() - self._last_progress_at
        if silence < self.stall_timeout:
            return HealthVerdict.HEALTHY

        gpu = await self._read_gpu_util()
        cpu = self._read_cpu_util()
        idle = self._looks_idle(gpu, cpu)

        if idle:
            # SUSPECT -> escalate immediately to the canary, the only signal
            # that can't be fooled by a wedged-but-200-OK server.
            return await self._canary_verdict()

        if silence >= self.hard_ceiling:
            # BUSY for too long even by GPU-pegged standards -- verify with
            # the canary rather than waiting on faith forever.
            return await self._canary_verdict()

        return HealthVerdict.BUSY

    def _looks_idle(self, gpu: float | None, cpu: float | None) -> bool:
        if gpu is not None:
            return gpu < self.gpu_idle_threshold
        if cpu is not None:
            return cpu < self.cpu_idle_threshold
        # No signal at all: never restart a job making progress, but here we
        # already know WS has been silent past stall_timeout, so treat
        # "unknown load" as idle-enough to warrant the (harmless) canary
        # check rather than waiting blindly up to hard_ceiling with no signal.
        return True

    async def _canary_verdict(self) -> HealthVerdict:
        reached = await self._run_canary()
        if reached:
            self._last_progress_at = self._clock()
            return HealthVerdict.HEALTHY
        return HealthVerdict.ZOMBIE

    # ----------------------------------------------------------------- tick()

    async def tick(self) -> HealthVerdict:
        """Convenience orchestration: assess, apply hysteresis, restart when
        warranted. This is the intended production entry point; `assess()`
        alone never restarts anything on its own."""
        verdict = await self.assess()
        if verdict in (HealthVerdict.ZOMBIE, HealthVerdict.DEAD):
            self._bad_streak += 1
        else:
            self._bad_streak = 0

        if self._bad_streak >= self.hysteresis_n:
            await self.restart(reason=verdict.value)
            self._bad_streak = 0
        return verdict

    # ---------------------------------------------------------------- canary

    async def _run_canary(self) -> bool:
        job_key = f"supervisor-canary-{uuid.uuid4().hex}"
        try:
            prompt_id = await self.comfy.submit(_CANARY_GRAPH, job_key, front=True)
        except Exception:
            logger.warning("canary_submit_failed", exc_info=True)
            return False

        waiter = asyncio.Event()
        self._prompt_waiters[prompt_id] = waiter
        try:
            await asyncio.wait_for(waiter.wait(), timeout=self.canary_timeout)
        except asyncio.TimeoutError:
            return False
        finally:
            self._prompt_waiters.pop(prompt_id, None)

        events = self._prompt_events.pop(prompt_id, [])
        return "executed" in events or "execution_success" in events

    # ------------------------------------------------------------- WS pump

    def _ensure_pump_started(self) -> None:
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(self._pump_events())

    async def _pump_events(self) -> None:
        try:
            async for evt in self.comfy.events():
                etype = evt.get("type")
                data = evt.get("data", {}) or {}
                if etype in _PROGRESS_EVENTS:
                    self._last_progress_at = self._clock()
                if etype in _TERMINAL_EVENTS:
                    if etype in ("executed", "execution_success"):
                        self._last_progress_at = self._clock()
                    pid = data.get("prompt_id")
                    waiter = self._prompt_waiters.get(pid) if pid else None
                    if waiter is not None:
                        self._prompt_events.setdefault(pid, []).append(etype)
                        waiter.set()
        except asyncio.CancelledError:
            raise
        except Exception:
            # A dropped WS connection is a DEAD/ZOMBIE symptom, not a crash --
            # assess()'s HTTP probe and canary will catch it on the next call.
            logger.warning("event_pump_stopped", exc_info=True)

    async def _safe_is_responding(self) -> bool:
        try:
            return await self.comfy.is_responding()
        except Exception:
            return False

    # ------------------------------------------------------------ GPU / CPU

    async def _read_gpu_util(self) -> float | None:
        if self.gpu_util_fn is None:
            return None
        try:
            return await asyncio.to_thread(self.gpu_util_fn)
        except Exception:
            return None

    def _read_cpu_util(self) -> float | None:
        if self.cpu_util_fn is None:
            return None
        try:
            return self.cpu_util_fn()
        except Exception:
            return None

    # -------------------------------------------------------------- restart

    async def restart(self, reason: str) -> bool:
        async with self._restart_lock:
            if self._held:
                return False

            now = self._clock()
            self._restart_times = [t for t in self._restart_times if now - t < self.intensity_window]

            if len(self._restart_times) >= self.intensity_max_restarts:
                self._held = True
                logger.critical(
                    "restart_intensity_ceiling_hit",
                    reason=reason,
                    restarts_in_window=len(self._restart_times),
                    window_s=self.intensity_window,
                )
                return False

            delay = min(self.backoff_base * (2 ** len(self._restart_times)), self.backoff_cap)
            logger.warning("restarting_comfyui", reason=reason, backoff_s=delay)
            await self._sleep(delay)

            await self._do_restart(reason)
            self._restart_times.append(self._clock())
            self._bad_streak = 0
            return True

    async def _do_restart(self, reason: str) -> None:
        self._write_marker(reason)
        await self._terminate_current_process()
        await self._launch_new_process()

    def _write_marker(self, reason: str) -> None:
        try:
            self.marker_dir.mkdir(parents=True, exist_ok=True)
            marker = self.marker_dir / f"restart-{int(time.time() * 1000)}.json"
            marker.write_text(json.dumps({"reason": reason, "at": time.time()}))
        except OSError:
            logger.warning("marker_write_failed", exc_info=True)

    async def _terminate_current_process(self) -> None:
        if self._proc is None or self._proc.returncode is not None:
            return
        self._proc.terminate()
        try:
            await asyncio.wait_for(self._proc.wait(), timeout=15)
        except asyncio.TimeoutError:
            self._proc.kill()
            await self._proc.wait()

    async def _launch_new_process(self) -> None:
        # Deliberately NOT os.execv(): that replaces this process image and
        # orphans the child from its supervisor (ComfyUI-Manager's legacy
        # restart path does exactly that). We relaunch as a genuine child so
        # this supervisor keeps owning its lifecycle across restarts.
        if self.launch_argv is None:
            # Dry-run / test mode: no real ComfyUI to spawn.
            self._proc = None
            return
        argv = self._build_launch_argv()
        self._proc = await asyncio.create_subprocess_exec(*argv)

    def _build_launch_argv(self) -> list[str]:
        argv = list(self.launch_argv or [])
        if self.cache_none:
            argv.append("--cache-none")
        if self.disable_dynamic_vram:
            argv.append("--disable-dynamic-vram")
        argv.extend(self.extra_launch_args)
        return argv

    # ---------------------------------------------------------------- recycle

    async def maybe_recycle(self, jobs_since_start: int) -> bool:
        """Proactive restart every N jobs -- insurance against slow leaks
        that no exception guard catches."""
        if self.recycle_every_n_jobs <= 0:
            return False
        if jobs_since_start <= 0 or jobs_since_start % self.recycle_every_n_jobs != 0:
            return False
        return await self.restart(reason=f"proactive_recycle@{jobs_since_start}_jobs")
