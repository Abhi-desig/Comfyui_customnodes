import pytest
from comfy_controller.models import Asset, AssetState, QCVerdict, CheckResult
from comfy_controller.testing.fake_comfy import FakeComfy, FakeComfyState, Fault


def test_terminal_states():
    assert AssetState.SAVED.is_terminal
    assert not AssetState.CLAIMED.is_terminal
    assert AssetState.SUBMITTED.needs_reconciliation


def test_job_key_deterministic():
    a = Asset(id="x", workflow="w.json", inputs={"p": "hi"}, seed=1)
    assert a.job_key("m", "g") == a.job_key("m", "g")
    assert a.job_key("m", "g") != a.job_key("m2", "g")


def test_verdict_all_checks_must_pass():
    v = QCVerdict(checks=[CheckResult(name="logo", evidence="e", passed=True),
                          CheckResult(name="colour", evidence="e", passed=False)])
    assert not v.passed and v.failed_checks == ["colour"]


async def test_fake_comfy_happy_path():
    fc = FakeComfy()
    base = await fc.start()
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.ws_connect(f"{base}/ws?clientId=c1") as ws:
                await s.post(f"{base}/prompt", json={"client_id": "c1", "prompt": {}})
                seen = []
                while "execution_success" not in seen:
                    msg = await ws.receive_json(timeout=5)
                    seen.append(msg["type"])
                assert "executed" in seen
    finally:
        await fc.stop()


async def test_fake_comfy_zombie_accepts_but_never_runs():
    fc = FakeComfy(FakeComfyState(fault=Fault.ZOMBIE))
    base = await fc.start()
    try:
        import aiohttp, asyncio
        async with aiohttp.ClientSession() as s:
            r = await s.post(f"{base}/prompt", json={"client_id": "c1", "prompt": {}})
            assert r.status == 200                      # HTTP still fine
            assert (await s.get(f"{base}/system_stats")).status == 200   # still 200!
            await asyncio.sleep(0.3)
            q = await (await s.get(f"{base}/queue")).json()
            assert q["queue_running"]                   # stuck forever
            assert fc.state.executed_count == 0
    finally:
        await fc.stop()
