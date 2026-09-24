"""ComfyPort implementation talking to a real (or FakeComfy) ComfyUI server.

Two things make ComfyUI's HTTP+WS API easy to get subtly wrong, and both are
load-bearing here:

  * `execution_error` is sent with broadcast=False — it only reaches the
    websocket connection whose `clientId` matches the `client_id` the prompt
    was submitted with. Get that mismatched (or reconnect without keeping the
    same id) and node failures silently vanish.
  * The websocket *will* drop. A batch runs for hours; a reconnect must not
    lose track of prompts already in flight, so on reconnect we reconcile
    every still-unresolved prompt_id against `/history/{prompt_id}` before
    resuming the normal event loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import hashlib
import json
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, NamedTuple

import aiohttp
import stamina

from ..models import ExecutionError, JobOutcome

# --------------------------------------------------------------------- errors


class ComfyValidationError(RuntimeError):
    """POST /prompt was accepted (HTTP 200) but ComfyUI rejected the graph
    itself — `node_errors` was non-empty. Not a transient failure; retrying
    the same graph will not help."""


# ---------------------------------------------------------------- pure helpers


def graph_hash(graph: dict[str, Any]) -> str:
    """Stable hash of an API-format graph's structure.

    Used by `Asset.job_key` (see models.py) so the same graph shape always
    contributes the same component to the job key, independent of dict
    ordering.
    """
    payload = json.dumps(graph, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def validate_filename_prefix(filename_prefix: str) -> str:
    """Reject anything that could make ComfyUI's `SaveImage` write outside its
    own output directory.

    `filename_prefix` is joined against ComfyUI's output dir server-side, so a
    prefix carrying a path separator or a `..` component is a path-traversal
    primitive. It is also what startup reconciliation globs for on disk, and a
    key with a separator in it silently lands in a subdirectory the glob never
    looks at. Both reasons say the same thing: keep it a flat, relative name.
    """
    if not filename_prefix:
        raise ValueError("filename_prefix must not be empty")
    if "/" in filename_prefix or "\\" in filename_prefix or "\x00" in filename_prefix:
        raise ValueError(f"filename_prefix must not contain a path separator: {filename_prefix!r}")
    if ".." in filename_prefix:
        raise ValueError(f"filename_prefix must not contain '..': {filename_prefix!r}")
    if Path(filename_prefix).is_absolute():
        raise ValueError(f"filename_prefix must be relative: {filename_prefix!r}")
    return filename_prefix


def patch_graph(
    graph: dict[str, Any],
    *,
    seed: int | None = None,
    prompt_text: str | None = None,
    filename_prefix: str | None = None,
    seed_node: str | None = None,
    prompt_node: str | None = None,
    filename_prefix_node: str | None = None,
) -> dict[str, Any]:
    """Return a patched deep copy of an API-format graph. Never mutates `graph`.

    Each of `seed` / `prompt_text` / `filename_prefix` is applied only when
    given (None = leave alone), targeting the first node matching the
    conventional `class_type` (KSampler / CLIPTextEncode) unless the matching
    `*_node` override names an explicit node id — needed the moment a graph
    has more than one node of that type, e.g. separate positive/negative
    CLIPTextEncode nodes.

    `filename_prefix` is the exception: it is applied to EVERY `SaveImage`
    node (unless `filename_prefix_node` names one explicitly). A graph that
    saves more than one image would otherwise leave the second `SaveImage`
    writing under whatever prefix the workflow author baked in, which both
    collides across assets and hides that output from reconciliation's
    job_key glob. It is validated by `validate_filename_prefix` first.
    """
    patched = copy.deepcopy(graph)

    def _target(class_type: str, override: str | None) -> dict[str, Any] | None:
        if override is not None:
            node = patched.get(override)
            return node if isinstance(node, dict) else None
        for node in patched.values():
            if isinstance(node, dict) and node.get("class_type") == class_type:
                return node
        return None

    def _targets(class_type: str, override: str | None) -> list[dict[str, Any]]:
        if override is not None:
            node = patched.get(override)
            return [node] if isinstance(node, dict) else []
        return [
            node
            for node in patched.values()
            if isinstance(node, dict) and node.get("class_type") == class_type
        ]

    if seed is not None:
        node = _target("KSampler", seed_node)
        if node is not None:
            node.setdefault("inputs", {})["seed"] = seed

    if prompt_text is not None:
        node = _target("CLIPTextEncode", prompt_node)
        if node is not None:
            node.setdefault("inputs", {})["text"] = prompt_text

    if filename_prefix is not None:
        validate_filename_prefix(filename_prefix)
        for node in _targets("SaveImage", filename_prefix_node):
            node.setdefault("inputs", {})["filename_prefix"] = filename_prefix

    return patched


def strip_preview_header(frame: bytes) -> tuple[int, bytes]:
    """ComfyUI preview frames are a 4-byte big-endian type header (1=JPEG,
    2=PNG) followed by the encoded image. We only need to not choke on them,
    but split it out as a pure function so the framing logic is testable
    without a live socket."""
    if len(frame) < 4:
        return (0, frame)
    frame_type = int.from_bytes(frame[:4], "big")
    return (frame_type, frame[4:])


class QueueEntry(NamedTuple):
    """ComfyUI's `/queue` returns arrays of *tuples*
    `(number, prompt_id, prompt, extra_data, outputs_to_execute)`, not dicts."""

    number: float
    prompt_id: str
    prompt: dict[str, Any]
    extra_data: dict[str, Any]
    outputs_to_execute: list[Any]


def _parse_queue_entries(raw: list[Any]) -> list[QueueEntry]:
    """Defensive: skip anything that isn't shaped like the tuple we expect
    rather than raising, since this feeds health/queue-depth reporting and a
    malformed entry shouldn't take the whole read down."""
    entries: list[QueueEntry] = []
    for item in raw:
        if not isinstance(item, (list, tuple)):
            continue
        entries.append(
            QueueEntry(
                number=item[0] if len(item) > 0 else 0.0,
                prompt_id=item[1] if len(item) > 1 else "",
                prompt=item[2] if len(item) > 2 else {},
                extra_data=item[3] if len(item) > 3 else {},
                outputs_to_execute=item[4] if len(item) > 4 else [],
            )
        )
    return entries


def _is_transient_http_error(exc: BaseException) -> bool:
    """Retry connection failures and 5xx; never retry a 4xx (that graph, or
    that request, will not start working the second time)."""
    if isinstance(exc, (aiohttp.ClientConnectionError, asyncio.TimeoutError, aiohttp.ServerTimeoutError)):
        return True
    if isinstance(exc, aiohttp.ClientResponseError):
        return exc.status >= 500
    return False


# ------------------------------------------------------------------ the port


class ComfyHTTP:
    """Talks to one ComfyUI server over HTTP + a single reconnecting websocket."""

    def __init__(
        self,
        base_url: str,
        *,
        session: aiohttp.ClientSession | None = None,
        client_id: str | None = None,
        ws_backoff_initial: float = 0.5,
        ws_backoff_max: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        # Generated once per controller instance and reused for both POST
        # /prompt and the ws connection — see module docstring.
        self.client_id = client_id or uuid.uuid4().hex

        self._session = session
        self._owns_session = session is None
        self._ws_backoff_initial = ws_backoff_initial
        self._ws_backoff_max = ws_backoff_max

        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ws_task: asyncio.Task[None] | None = None
        self._connected_once = asyncio.Event()
        self._closed = False

        # Event fan-out for events() consumers.
        self._subscribers: list[asyncio.Queue[dict[str, Any]]] = []

        # Outcome tracking, keyed by prompt_id, so await_outcome() and the
        # reconnect-reconciliation pass share one source of truth.
        self._in_flight: set[str] = set()
        self._results: dict[str, JobOutcome] = {}
        self._waiters: dict[str, asyncio.Event] = {}

    # ------------------------------------------------------------- lifecycle

    async def close(self) -> None:
        self._closed = True
        if self._ws_task is not None:
            self._ws_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._ws_task
        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
            self._owns_session = True
        return self._session

    def _ensure_reader_started(self) -> None:
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._ws_reader_loop())

    async def _wait_connected(self, timeout: float = 10.0) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._connected_once.wait(), timeout=timeout)

    # ------------------------------------------------------------- HTTP calls

    @stamina.retry(on=_is_transient_http_error, attempts=4, wait_initial=0.2, wait_max=2.0, timeout=10)
    async def _get_json(self, path: str) -> dict[str, Any]:
        session = await self._ensure_session()
        async with session.get(f"{self.base_url}{path}", timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            return await resp.json()

    @stamina.retry(on=_is_transient_http_error, attempts=4, wait_initial=0.2, wait_max=2.0, timeout=10)
    async def _post_json(self, path: str, payload: dict[str, Any] | None) -> dict[str, Any] | None:
        session = await self._ensure_session()
        async with session.post(
            f"{self.base_url}{path}", json=payload, timeout=aiohttp.ClientTimeout(total=10)
        ) as resp:
            resp.raise_for_status()
            if resp.content_type == "application/json":
                return await resp.json()
            return None

    # ------------------------------------------------------------- ComfyPort

    async def submit(self, graph: dict[str, Any], job_key: str, front: bool = False) -> str:
        # filename_prefix is forced from job_key (not left to whatever the
        # graph already had) so startup reconciliation can find this job's
        # outputs on disk even if the prompt_id itself was never recorded.
        patched = patch_graph(graph, filename_prefix=job_key)

        self._ensure_reader_started()
        await self._wait_connected()  # avoid racing our own execution_start

        payload = {"client_id": self.client_id, "prompt": patched, "front": front}
        data = await self._post_json("/prompt", payload)
        assert data is not None
        node_errors = data.get("node_errors") or {}
        if node_errors:
            raise ComfyValidationError(f"ComfyUI rejected the graph: {node_errors}")

        prompt_id = str(data["prompt_id"])
        self._in_flight.add(prompt_id)
        return prompt_id

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        self._ensure_reader_started()
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            with contextlib.suppress(ValueError):
                self._subscribers.remove(queue)

    async def await_outcome(self, prompt_id: str, timeout: float) -> JobOutcome:
        self._ensure_reader_started()

        cached = self._results.pop(prompt_id, None)
        if cached is not None:
            self._waiters.pop(prompt_id, None)
            return cached

        waiter = self._waiters.setdefault(prompt_id, asyncio.Event())
        # Every exit path — resolved, timed out, cancelled, or the defensive
        # RuntimeError below — must drop the waiter and stop treating the
        # prompt as in flight. Leaking either means an unbounded dict for a
        # batch that runs for hours, plus a /history round-trip for that dead
        # prompt on every single websocket reconnect from here on.
        try:
            try:
                await asyncio.wait_for(waiter.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                # One direct history check before giving up: covers an event
                # that was missed entirely (dropped mid-gap, before any
                # reconnect has had a chance to reconcile it).
                entry = await self.history(prompt_id)
                if entry is not None:
                    return self._outcome_from_history(prompt_id, entry)
                raise

            result = self._results.pop(prompt_id, None)
            if result is None:  # pragma: no cover - defensive; _resolve always sets one
                raise RuntimeError(f"waiter for {prompt_id} fired without a stored outcome")
            return result
        finally:
            self._waiters.pop(prompt_id, None)
            self._in_flight.discard(prompt_id)

    async def queue_depth(self) -> tuple[int, int]:
        data = await self._get_json("/queue")
        running = _parse_queue_entries(data.get("queue_running", []))
        pending = _parse_queue_entries(data.get("queue_pending", []))
        return (len(running), len(pending))

    async def history(self, prompt_id: str) -> dict[str, Any] | None:
        data = await self._get_json(f"/history/{prompt_id}")
        return data.get(prompt_id)

    async def is_responding(self) -> bool:
        """HTTP liveness only — NOT a worker-liveness signal. ComfyUI keeps
        answering /system_stats with 200 straight through a zombie worker
        (#16312); use HealthVerdict / the canary for real liveness."""
        try:
            await self._get_json("/system_stats")
            return True
        except (aiohttp.ClientError, asyncio.TimeoutError):
            return False

    async def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        await self._post_json("/free", {"unload_models": unload_models, "free_memory": free_memory})

    async def interrupt(self, prompt_id: str | None = None) -> None:
        payload = {"prompt_id": prompt_id} if prompt_id else None
        await self._post_json("/interrupt", payload)

    # --------------------------------------------------------------- ws loop

    async def _ws_reader_loop(self) -> None:
        ws_url = self._ws_url()
        backoff = self._ws_backoff_initial
        while not self._closed:
            try:
                session = await self._ensure_session()
                async with session.ws_connect(
                    ws_url, timeout=aiohttp.ClientWSTimeout(ws_close=10)
                ) as ws:
                    self._ws = ws
                    backoff = self._ws_backoff_initial
                    self._connected_once.set()
                    # A reconnect must not lose track of prompts already in
                    # flight: reconcile each against history before resuming
                    # the normal event loop, in case its outcome event fired
                    # while we were disconnected.
                    await self._reconcile_in_flight()

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                event = json.loads(msg.data)
                            except json.JSONDecodeError:
                                continue
                            await self._dispatch_event(event)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            # Preview frame — strip the header and move on;
                            # must never break the text-event loop.
                            strip_preview_header(msg.data)
                        elif msg.type in (
                            aiohttp.WSMsgType.ERROR,
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                        ):
                            break
            except asyncio.CancelledError:
                raise
            except Exception:
                pass  # dropped connection; fall through to backoff + retry
            finally:
                self._ws = None
                self._connected_once.clear()

            if self._closed:
                return
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, self._ws_backoff_max)

    def _ws_url(self) -> str:
        scheme = "wss" if self.base_url.startswith("https://") else "ws"
        host = self.base_url.split("://", 1)[-1]
        return f"{scheme}://{host}/ws?clientId={self.client_id}"

    async def _reconcile_in_flight(self) -> None:
        # A history entry only exists once execution has actually finished
        # (success or error) — FakeComfy even sets status.completed=False on
        # an error entry, so presence in history (not that flag) is what
        # means "terminal" here.
        for prompt_id in list(self._in_flight):
            entry = await self.history(prompt_id)
            if entry is not None:
                await self._resolve(prompt_id, self._outcome_from_history(prompt_id, entry))

    async def _dispatch_event(self, event: dict[str, Any]) -> None:
        for queue in list(self._subscribers):
            queue.put_nowait(event)

        etype = event.get("type")
        data = event.get("data") or {}
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            return

        if etype == "execution_error":
            error = ExecutionError(
                prompt_id=prompt_id,
                node_id=data.get("node_id"),
                node_type=data.get("node_type"),
                exception_message=data.get("exception_message", ""),
                exception_type=data.get("exception_type", ""),
                traceback=data.get("traceback", []) or [],
            )
            await self._resolve(prompt_id, JobOutcome(prompt_id=prompt_id, succeeded=False, error=error))
        elif etype == "execution_success":
            entry = await self.history(prompt_id)
            paths = self._extract_output_paths((entry or {}).get("outputs", {}))
            await self._resolve(prompt_id, JobOutcome(prompt_id=prompt_id, succeeded=True, output_paths=paths))
        elif etype == "execution_interrupted":
            await self._resolve(prompt_id, JobOutcome(prompt_id=prompt_id, succeeded=False))

    async def _resolve(self, prompt_id: str, outcome: JobOutcome) -> None:
        self._results[prompt_id] = outcome
        self._in_flight.discard(prompt_id)
        self._waiters.setdefault(prompt_id, asyncio.Event()).set()

    # ------------------------------------------------------- history parsing

    def _outcome_from_history(self, prompt_id: str, entry: dict[str, Any]) -> JobOutcome:
        status = entry.get("status") or {}
        outputs = entry.get("outputs") or {}
        output_paths = self._extract_output_paths(outputs)
        if status.get("status_str") == "success":
            return JobOutcome(prompt_id=prompt_id, succeeded=True, output_paths=output_paths)
        error = self._error_from_history_messages(prompt_id, status.get("messages"))
        return JobOutcome(prompt_id=prompt_id, succeeded=False, output_paths=output_paths, error=error)

    @staticmethod
    def _error_from_history_messages(prompt_id: str, messages: list[Any] | None) -> ExecutionError:
        for msg in messages or []:
            if isinstance(msg, (list, tuple)) and len(msg) == 2 and msg[0] == "execution_error":
                data = msg[1] if isinstance(msg[1], dict) else {}
                return ExecutionError(
                    prompt_id=prompt_id,
                    node_id=data.get("node_id"),
                    node_type=data.get("node_type"),
                    exception_message=data.get("exception_message", ""),
                    exception_type=data.get("exception_type", ""),
                    traceback=data.get("traceback", []) or [],
                )
        # No detailed message survived (e.g. FakeComfy's minimal history) —
        # surface that the job failed rather than silently reporting success.
        return ExecutionError(prompt_id=prompt_id, exception_message="execution failed; no detail in history")

    @staticmethod
    def _extract_output_paths(outputs: dict[str, Any]) -> list[str]:
        paths: list[str] = []
        for node_output in (outputs or {}).values():
            for image in node_output.get("images", []) or []:
                subfolder = image.get("subfolder") or ""
                filename = image.get("filename", "")
                paths.append(f"{subfolder}/{filename}" if subfolder else filename)
        return paths
