"""Tests for the ComfyHTTP adapter, run against FakeComfy (no GPU needed)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

from comfy_controller.adapters.comfy_http import (
    ComfyHTTP,
    QueueEntry,
    _parse_queue_entries,
    graph_hash,
    patch_graph,
    strip_preview_header,
)
from comfy_controller.testing.fake_comfy import FakeComfy, FakeComfyState, Fault

SAMPLE_GRAPH: dict[str, Any] = {
    "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a cat", "clip": ["1", 1]}},
    "3": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "model": ["1", 0]}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "ComfyUI", "images": ["3", 0]}},
}


# --------------------------------------------------------------------- fixtures


@pytest_asyncio.fixture
async def fake_comfy():
    fc = FakeComfy()
    base_url = await fc.start()
    try:
        yield fc, base_url
    finally:
        await fc.stop()


@pytest_asyncio.fixture
async def comfy(fake_comfy):
    _, base_url = fake_comfy
    client = ComfyHTTP(base_url)
    try:
        yield client
    finally:
        await client.close()


# ------------------------------------------------------------------ pure helpers


def test_patch_graph_returns_deep_copy_and_does_not_mutate_input():
    original_json = json.dumps(SAMPLE_GRAPH, sort_keys=True)

    patched = patch_graph(SAMPLE_GRAPH, seed=999, prompt_text="a dog", filename_prefix="job-abc")

    assert json.dumps(SAMPLE_GRAPH, sort_keys=True) == original_json  # untouched
    assert patched is not SAMPLE_GRAPH
    assert patched["3"] is not SAMPLE_GRAPH["3"]
    assert patched["3"]["inputs"]["seed"] == 999
    assert patched["2"]["inputs"]["text"] == "a dog"
    assert patched["9"]["inputs"]["filename_prefix"] == "job-abc"


def test_patch_graph_is_deterministic():
    a = patch_graph(SAMPLE_GRAPH, seed=5, prompt_text="x", filename_prefix="p")
    b = patch_graph(SAMPLE_GRAPH, seed=5, prompt_text="x", filename_prefix="p")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_patch_graph_leaves_unspecified_fields_alone():
    patched = patch_graph(SAMPLE_GRAPH, seed=42)
    assert patched["3"]["inputs"]["seed"] == 42
    assert patched["2"]["inputs"]["text"] == "a cat"  # untouched
    assert patched["9"]["inputs"]["filename_prefix"] == "ComfyUI"  # untouched


def test_patch_graph_explicit_node_id_override_for_ambiguous_class_type():
    graph = {
        "10": {"class_type": "CLIPTextEncode", "inputs": {"text": "positive"}},
        "11": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
    }
    # Without an override, the first CLIPTextEncode found (by dict order) wins.
    default_target = patch_graph(graph, prompt_text="new positive")
    assert default_target["10"]["inputs"]["text"] == "new positive"
    assert default_target["11"]["inputs"]["text"] == "negative"

    # An explicit node id override reaches the other node instead.
    overridden = patch_graph(graph, prompt_text="new negative", prompt_node="11")
    assert overridden["10"]["inputs"]["text"] == "positive"
    assert overridden["11"]["inputs"]["text"] == "new negative"


def test_graph_hash_stable_and_order_independent():
    reordered = {k: SAMPLE_GRAPH[k] for k in reversed(list(SAMPLE_GRAPH))}
    assert graph_hash(SAMPLE_GRAPH) == graph_hash(reordered)


def test_graph_hash_sensitive_to_structure():
    changed = patch_graph(SAMPLE_GRAPH, seed=2)
    assert graph_hash(SAMPLE_GRAPH) != graph_hash(changed)


def test_strip_preview_header_splits_type_and_payload():
    frame = (2).to_bytes(4, "big") + b"\x89PNG..."
    frame_type, payload = strip_preview_header(frame)
    assert frame_type == 2
    assert payload == b"\x89PNG..."


def test_parse_queue_entries_reads_tuples_not_dicts():
    raw = [[1.0, "pid-1", {"1": {"class_type": "KSampler"}}, {"extra": True}, ["9"]]]
    entries = _parse_queue_entries(raw)
    assert entries == [QueueEntry(1.0, "pid-1", {"1": {"class_type": "KSampler"}}, {"extra": True}, ["9"])]


def test_parse_queue_entries_skips_malformed_rows_defensively():
    raw = [
        [1.0, "pid-1", {}, {}, []],
        "not-a-tuple-at-all",
        [2.0],  # short, but still shaped like a sequence
    ]
    entries = _parse_queue_entries(raw)
    assert len(entries) == 2
    assert entries[0].prompt_id == "pid-1"
    assert entries[1] == QueueEntry(2.0, "", {}, {}, [])


# -------------------------------------------------------------------- happy path


async def test_submit_and_await_outcome_happy_path(comfy: ComfyHTTP):
    prompt_id = await comfy.submit(SAMPLE_GRAPH, job_key="job-key-1")
    outcome = await comfy.await_outcome(prompt_id, timeout=5)

    assert outcome.succeeded
    assert outcome.error is None
    assert outcome.output_paths == ["job-key-1_00001_.png"]  # filename_prefix came from job_key


async def test_submit_sends_stable_client_id_used_by_both_http_and_ws(fake_comfy):
    _, base_url = fake_comfy
    comfy = ComfyHTTP(base_url)
    try:
        first_id = comfy.client_id
        prompt_id = await comfy.submit(SAMPLE_GRAPH, job_key="job-key-2")
        # Same instance keeps the same client_id across calls.
        assert comfy.client_id == first_id
        # If client_id sent on /prompt didn't match the one used on the ws
        # connection, execution_error/execution_success would never reach us
        # (FakeComfy sends them broadcast=False) and this would time out.
        outcome = await comfy.await_outcome(prompt_id, timeout=5)
        assert outcome.succeeded
    finally:
        await comfy.close()


async def test_two_instances_get_different_client_ids(fake_comfy):
    _, base_url = fake_comfy
    a, b = ComfyHTTP(base_url), ComfyHTTP(base_url)
    try:
        assert a.client_id != b.client_id
    finally:
        await a.close()
        await b.close()


# ------------------------------------------------------------------- node error


async def test_node_error_populates_execution_error(fake_comfy):
    fc, base_url = fake_comfy
    fc.state.fault = Fault.NODE_ERROR
    fc.state.fault_after_n_jobs = 0

    comfy = ComfyHTTP(base_url)
    try:
        prompt_id = await comfy.submit(SAMPLE_GRAPH, job_key="job-key-err")
        outcome = await comfy.await_outcome(prompt_id, timeout=5)

        assert not outcome.succeeded
        assert outcome.error is not None
        assert outcome.error.prompt_id == prompt_id
        assert outcome.error.node_id == "1"
        assert outcome.error.node_type == "KSampler"
        assert outcome.error.exception_type == "RuntimeError"
        assert outcome.error.exception_message == "node blew up"
    finally:
        await comfy.close()


# ------------------------------------------------------------------- queue depth


async def test_queue_depth_counts_running_and_pending_from_tuples(fake_comfy):
    fc, base_url = fake_comfy
    fc.state.fault = Fault.SLOW
    fc.state.fault_after_n_jobs = 0
    fc.state.slow_seconds = 0.4

    comfy = ComfyHTTP(base_url)
    try:
        running_id = await comfy.submit(SAMPLE_GRAPH, job_key="job-running")
        await asyncio.sleep(0.05)  # let FakeComfy pick it up as "running"
        await comfy.submit(SAMPLE_GRAPH, job_key="job-pending-1")
        await comfy.submit(SAMPLE_GRAPH, job_key="job-pending-2")

        running, pending = await comfy.queue_depth()
        assert running == 1
        assert pending == 2

        await comfy.await_outcome(running_id, timeout=5)  # drain so fixture teardown is clean
    finally:
        await comfy.close()


# --------------------------------------------------------------- front ordering


async def test_submit_front_true_jumps_the_queue(fake_comfy):
    fc, base_url = fake_comfy
    fc.state.fault = Fault.SLOW
    fc.state.fault_after_n_jobs = 0
    fc.state.slow_seconds = 0.4

    comfy = ComfyHTTP(base_url)
    try:
        running_id = await comfy.submit(SAMPLE_GRAPH, job_key="job-running")
        await asyncio.sleep(0.05)  # let it start running so the next two queue up
        back_id = await comfy.submit(SAMPLE_GRAPH, job_key="job-back", front=False)
        front_id = await comfy.submit(SAMPLE_GRAPH, job_key="job-front", front=True)

        data = await comfy._get_json("/queue")
        pending = _parse_queue_entries(data["queue_pending"])
        pending_ids = [e.prompt_id for e in pending]

        assert pending_ids[0] == front_id  # jumped ahead despite being submitted last
        assert back_id in pending_ids
        # front jobs are given a negative sequence number by ComfyUI/FakeComfy
        front_entry = next(e for e in pending if e.prompt_id == front_id)
        assert front_entry.number < 0

        await comfy.await_outcome(running_id, timeout=5)
    finally:
        await comfy.close()


# ------------------------------------------------------------------ binary frame


async def _binary_then_text_ws_server() -> tuple[web.AppRunner, str]:
    """A tiny standalone ws server (not FakeComfy — Wave 0 is frozen) that
    sends one binary preview-shaped frame before a normal text event, to
    prove ComfyHTTP's event loop survives interleaved binary frames."""

    async def handler(request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        header = (1).to_bytes(4, "big")  # 1 = JPEG
        await ws.send_bytes(header + b"\xff\xd8\xff\xfake-jpeg-bytes")
        await ws.send_str(json.dumps({"type": "status", "data": {"sid": "x"}}))
        await asyncio.sleep(0.2)  # keep the socket open briefly for the client to read
        await ws.close()
        return ws

    app = web.Application()
    app.router.add_get("/ws", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
    return runner, f"http://127.0.0.1:{port}"


async def test_binary_frame_does_not_break_the_text_event_loop():
    runner, base_url = await _binary_then_text_ws_server()
    comfy = ComfyHTTP(base_url)
    try:
        events = comfy.events()
        first_event = await asyncio.wait_for(events.__anext__(), timeout=5)
        assert first_event == {"type": "status", "data": {"sid": "x"}}
    finally:
        await comfy.close()
        await runner.cleanup()


# ------------------------------------------------------------- reconnect/reconcile


async def test_reconciles_missed_outcome_via_history_after_reconnect(fake_comfy):
    """Simulates a dropped connection: the execution_error event is sent while
    nobody is listening on this client_id (identical to what a real ws drop
    looks like from the client's side), then verifies ComfyHTTP recovers the
    outcome from /history/{prompt_id} once it (re)connects and reconciles a
    prompt_id it already knows is in flight."""
    fc, base_url = fake_comfy
    fc.state.fault = Fault.NODE_ERROR
    fc.state.fault_after_n_jobs = 0

    comfy = ComfyHTTP(base_url)
    try:
        async with aiohttp.ClientSession() as raw:
            resp = await raw.post(
                f"{base_url}/prompt",
                json={"client_id": comfy.client_id, "prompt": SAMPLE_GRAPH},
            )
            body = await resp.json()
        prompt_id = body["prompt_id"]

        for _ in range(200):
            if prompt_id in fc._history:
                break
            await asyncio.sleep(0.01)
        assert prompt_id in fc._history  # confirm it really finished before we ever connected

        # This is what startup reconciliation (or a resumed reader after a
        # drop) would have recorded: "we submitted this, we don't know how it
        # ended yet".
        comfy._in_flight.add(prompt_id)

        outcome = await comfy.await_outcome(prompt_id, timeout=5)
        assert not outcome.succeeded
        assert outcome.error is not None
        assert outcome.error.prompt_id == prompt_id
    finally:
        await comfy.close()


# ------------------------------------------------------------------- misc port


async def test_is_responding_is_http_only(comfy: ComfyHTTP):
    assert await comfy.is_responding() is True


async def test_free_posts_flags(fake_comfy):
    fc, base_url = fake_comfy
    comfy = ComfyHTTP(base_url)
    try:
        await comfy.free(unload_models=True, free_memory=False)
        assert fc.state.free_calls[-1] == {"unload_models": True, "free_memory": False}
    finally:
        await comfy.close()


async def test_interrupt_posts_prompt_id(fake_comfy):
    fc, base_url = fake_comfy
    comfy = ComfyHTTP(base_url)
    try:
        await comfy.interrupt("some-prompt-id")
        assert fc.state.interrupted[-1] == "some-prompt-id"
        await comfy.interrupt()
        assert fc.state.interrupted[-1] == "*"
    finally:
        await comfy.close()


async def test_history_returns_none_for_unknown_prompt(comfy: ComfyHTTP):
    assert await comfy.history("does-not-exist") is None
