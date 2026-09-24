"""A fake ComfyUI server with failure injection.

The highest-leverage file in the project: it lets every other module be built and
tested in parallel, at full speed, on a laptop with no GPU.

Faithful to the real server where it matters:
  * `execution_error` is sent only to the submitting client_id (broadcast=False)
  * `/queue` returns arrays of *tuples*, not dicts
  * `/free` sets flags consumed between jobs; it is not synchronous
  * `front=True` jumps the queue
  * the ZOMBIE fault keeps HTTP answering 200 while nothing ever executes,
    exactly as issue #16312 behaves
"""

from __future__ import annotations

import asyncio
import base64
import enum
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from aiohttp import WSMsgType, web

# Smallest valid PNG: 1x1, fully transparent. Decodable by Pillow.
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


class Fault(str, enum.Enum):
    NONE = "none"
    ZOMBIE = "zombie"          # #16312: HTTP fine, worker thread dead, no WS events ever
    NODE_ERROR = "node_error"  # ordinary per-node failure -> execution_error
    OOM = "oom"                # caught per-node upstream: FAILURE, worker SURVIVES
    MISSING_FILE = "missing_file"   # permanent error
    SILENT_MIDWAY = "silent_midway"  # emits some progress then stops (hang, not death)
    SLOW = "slow"              # legitimately long job — the watchdog must NOT restart this


@dataclass
class _Job:
    prompt_id: str
    client_id: str
    graph: dict[str, Any]
    number: float


@dataclass
class FakeComfyState:
    """Test-facing control surface."""

    fault: Fault = Fault.NONE
    fault_after_n_jobs: int = 0
    slow_seconds: float = 30.0
    step_delay: float = 0.01
    # When set, actually write the PNG that `/history` claims exists. Real
    # ComfyUI does, and the controller verifies the file is readable before
    # judging -- a fake that only reports filenames lets that check pass
    # vacuously. Left None by default so existing callers are unaffected.
    output_dir: Path | None = None
    executed_count: int = 0
    submitted_count: int = 0
    free_calls: list[dict[str, bool]] = field(default_factory=list)
    interrupted: list[str] = field(default_factory=list)

    def fault_active(self) -> bool:
        return self.fault is not Fault.NONE and self.submitted_count > self.fault_after_n_jobs


class FakeComfy:
    def __init__(self, state: FakeComfyState | None = None) -> None:
        self.state = state or FakeComfyState()
        self._queue: list[_Job] = []
        self._running: _Job | None = None
        self._history: dict[str, dict[str, Any]] = {}
        self._sockets: dict[str, web.WebSocketResponse] = {}
        self._number = 0.0
        self._worker: asyncio.Task[None] | None = None
        self._runner: web.AppRunner | None = None
        self.port = 0

    # ------------------------------------------------------------------ lifecycle

    async def start(self, port: int = 0) -> str:
        app = web.Application()
        app.router.add_post("/prompt", self._post_prompt)
        app.router.add_get("/queue", self._get_queue)
        app.router.add_post("/queue", self._post_queue)
        app.router.add_get("/history", self._get_history)
        app.router.add_get("/history/{prompt_id}", self._get_history_one)
        app.router.add_post("/free", self._post_free)
        app.router.add_post("/interrupt", self._post_interrupt)
        app.router.add_get("/system_stats", self._system_stats)
        app.router.add_get("/ws", self._ws)

        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self._worker = asyncio.create_task(self._prompt_worker())
        return f"http://127.0.0.1:{self.port}"

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
        if self._runner:
            await self._runner.cleanup()

    # ------------------------------------------------------------------- ws plumbing

    async def _send(self, event: str, data: dict[str, Any], client_id: str | None) -> None:
        """client_id=None broadcasts. Real ComfyUI sends execution_error with
        broadcast=False, so it reaches only the submitting client."""
        msg = json.dumps({"type": event, "data": data})
        targets = (
            [self._sockets[client_id]] if client_id and client_id in self._sockets
            else (list(self._sockets.values()) if client_id is None else [])
        )
        for ws in targets:
            if not ws.closed:
                await ws.send_str(msg)

    async def _ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        sid = request.rel_url.query.get("clientId") or uuid.uuid4().hex
        self._sockets[sid] = ws
        await ws.send_str(
            json.dumps({"type": "status",
                        "data": {"status": {"exec_info": {"queue_remaining": len(self._queue)}},
                                 "sid": sid}})
        )
        async for msg in ws:
            if msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
        self._sockets.pop(sid, None)
        return ws

    # ---------------------------------------------------------------------- routes

    async def _post_prompt(self, request: web.Request) -> web.Response:
        body = await request.json()
        self._number += 1
        number = -self._number if body.get("front") else self._number
        job = _Job(uuid.uuid4().hex, body.get("client_id", ""), body.get("prompt", {}), number)
        self.state.submitted_count += 1
        # A zombie still accepts jobs into the queue and returns 200. That is the bug.
        self._queue.append(job)
        self._queue.sort(key=lambda j: j.number)
        return web.json_response({"prompt_id": job.prompt_id, "number": number, "node_errors": {}})

    async def _get_queue(self, request: web.Request) -> web.Response:
        def tup(j: _Job) -> list[Any]:
            return [j.number, j.prompt_id, j.graph, {}, []]
        running = [tup(self._running)] if self._running else []
        return web.json_response({"queue_running": running,
                                  "queue_pending": [tup(j) for j in self._queue]})

    async def _post_queue(self, request: web.Request) -> web.Response:
        body = await request.json()
        if body.get("clear"):
            self._queue.clear()
        for pid in body.get("delete", []):
            self._queue = [j for j in self._queue if j.prompt_id != pid]
        return web.Response(status=200)

    async def _get_history(self, request: web.Request) -> web.Response:
        return web.json_response(self._history)

    async def _get_history_one(self, request: web.Request) -> web.Response:
        pid = request.match_info["prompt_id"]
        return web.json_response({pid: self._history[pid]} if pid in self._history else {})

    async def _post_free(self, request: web.Request) -> web.Response:
        self.state.free_calls.append(await request.json())
        return web.Response(status=200)

    async def _post_interrupt(self, request: web.Request) -> web.Response:
        try:
            pid = (await request.json()).get("prompt_id")
        except Exception:
            pid = None
        self.state.interrupted.append(pid or "*")
        return web.Response(status=200)

    async def _system_stats(self, request: web.Request) -> web.Response:
        # Deliberately keeps answering 200 during a zombie — this is why it must
        # never be used as a health check.
        return web.json_response({"system": {"comfyui_version": "fake"}, "devices": []})

    # ----------------------------------------------------------------- the worker

    async def _prompt_worker(self) -> None:
        while True:
            if self.state.fault_active() and self.state.fault is Fault.ZOMBIE:
                # Thread is dead. Jobs pile up in queue_running forever, no events.
                if self._running is None and self._queue:
                    self._running = self._queue.pop(0)
                await asyncio.sleep(0.05)
                continue

            if not self._queue:
                await asyncio.sleep(0.01)
                continue

            self._running = self._queue.pop(0)
            try:
                await self._execute(self._running)
            finally:
                self._running = None

    async def _execute(self, job: _Job) -> None:
        cid, pid = job.client_id, job.prompt_id
        fault = self.state.fault if self.state.fault_active() else Fault.NONE

        await self._send("execution_start", {"prompt_id": pid}, cid)
        await self._send("executing", {"node": "1", "prompt_id": pid}, cid)

        if fault is Fault.SILENT_MIDWAY:
            await self._send("progress", {"node": "1", "prompt_id": pid, "value": 3, "max": 20}, cid)
            return  # never completes; job leaves queue_running but nothing follows

        if fault is Fault.SLOW:
            for i in range(20):
                await asyncio.sleep(self.state.slow_seconds / 20)
                await self._send("progress", {"node": "1", "prompt_id": pid,
                                              "value": i + 1, "max": 20}, cid)

        if fault in (Fault.NODE_ERROR, Fault.OOM, Fault.MISSING_FILE):
            err = {
                Fault.NODE_ERROR: ("RuntimeError", "node blew up"),
                # OOM is caught per-node upstream: the prompt fails, worker survives.
                Fault.OOM: ("torch.cuda.OutOfMemoryError", "CUDA out of memory"),
                Fault.MISSING_FILE: ("FileNotFoundError", "No such file or directory: input.png"),
            }[fault]
            payload = {"prompt_id": pid, "node_id": "1", "node_type": "KSampler",
                       "exception_type": err[0], "exception_message": err[1],
                       "traceback": [], "current_inputs": {}, "current_outputs": []}
            await self._send("execution_error", payload, cid)   # submitting client only
            self._history[pid] = {"prompt": [], "outputs": {},
                                  "status": {"status_str": "error", "completed": False,
                                             "messages": []}}
            return

        if fault is Fault.NONE:
            for i in range(4):
                await asyncio.sleep(self.state.step_delay)
                await self._send("progress", {"node": "1", "prompt_id": pid,
                                              "value": i + 1, "max": 4}, cid)

        prefix = self._filename_prefix(job.graph)
        filename = f"{prefix}_00001_.png"
        self._write_output(filename)
        outputs = {"9": {"images": [{"filename": filename,
                                     "subfolder": "", "type": "output"}]}}
        await self._send("executed", {"node": "9", "output": outputs["9"], "prompt_id": pid}, cid)
        await self._send("executing", {"node": None, "prompt_id": pid}, cid)
        await self._send("execution_success", {"prompt_id": pid}, cid)
        self._history[pid] = {"prompt": [], "outputs": outputs,
                              "status": {"status_str": "success", "completed": True,
                                         "messages": []}}
        self.state.executed_count += 1

    def _write_output(self, filename: str) -> None:
        """Write a real, decodable 1x1 PNG so the controller's readability
        check (and any prefilter that opens the image) exercises the true
        path rather than passing vacuously."""
        if self.state.output_dir is None:
            return
        out = self.state.output_dir / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(_PNG_1X1)

    @staticmethod
    def _filename_prefix(graph: dict[str, Any]) -> str:
        for node in graph.values():
            if isinstance(node, dict):
                p = node.get("inputs", {}).get("filename_prefix")
                if p:
                    return str(p)
        return "ComfyUI"
