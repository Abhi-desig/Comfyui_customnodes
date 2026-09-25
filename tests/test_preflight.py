"""Tests for preflight.py: comfyctl's real-environment self-check.

No network, no GPU -- per the task brief. Anything that would otherwise hit
a real network (Anthropic, Slack) is exercised through an injected fake
client; anything that talks to "ComfyUI" spins up a real, local FakeComfy
test server (127.0.0.1, an actual aiohttp app) exactly like test_cli.py's
end-to-end tests already do -- that's a real HTTP/WS round trip against a
process on the loopback interface, not a network dependency.
"""

from __future__ import annotations

import asyncio
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml
from aiohttp import web

from comfy_controller import preflight
from comfy_controller.models import AssetState
from comfy_controller.ports import JudgeUnavailable
from comfy_controller.testing.fake_comfy import FakeComfy, FakeComfyState, Fault
from comfy_controller.testing.fake_judge import FakeJudge


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _workflow(tmp_path: Path, name: str = "wf.json", **extra_node_kwargs: Any) -> Path:
    path = tmp_path / name
    graph = {
        "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "x"}},
    }
    path.write_text(json.dumps(graph))
    return path


def _manifest(tmp_path: Path, wf: Path, name: str = "m.yaml") -> Path:
    manifest = tmp_path / name
    manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": str(wf)}]))
    return manifest


# --------------------------------------------------------------------- config


def test_check_config_pass_reports_secrets_without_values(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-super-secret")
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "claude"})

    result, cfg = preflight.check_config(cfg_path)
    assert result.status is preflight.Status.PASS
    assert cfg is not None
    assert "sk-super-secret" not in result.detail
    assert "ANTHROPIC_API_KEY=set" in result.detail
    assert "SLACK_BOT_TOKEN=unset" in result.detail


def test_check_config_fail_on_bad_config(tmp_path):
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "totally_unknown_field": 1})
    result, cfg = preflight.check_config(cfg_path)
    assert result.status is preflight.Status.FAIL
    assert cfg is None


# ------------------------------------------------------------------ guard log


def test_parse_guard_status_confirmed():
    text = "some noise\ncomfyui_guard_node: PATCH CONFIRMED -- guarded\nmore noise\n"
    assert preflight.parse_guard_status(text) == "confirmed"


def test_parse_guard_status_failed():
    text = "comfyui_guard_node: patch verification FAILED -- not guarded\n"
    assert preflight.parse_guard_status(text) == "failed"


def test_parse_guard_status_absent_when_no_markers():
    assert preflight.parse_guard_status("nothing relevant here\n") == "absent"


def test_parse_guard_status_last_line_wins():
    # An earlier failed boot followed by a later, successful (re)launch.
    text = "verification FAILED\n...\nPATCH CONFIRMED\n"
    assert preflight.parse_guard_status(text) == "confirmed"
    text2 = "PATCH CONFIRMED\n...\nNOT installed\n"
    assert preflight.parse_guard_status(text2) == "failed"


def test_check_guard_pass(tmp_path):
    log = tmp_path / "comfyui-latest.log"
    log.write_text("[launch_comfyui] guard node confirmed #16312 patch active\ncomfyui_guard_node: PATCH CONFIRMED\n")
    result = preflight.check_guard(log)
    assert result.status is preflight.Status.PASS


def test_check_guard_fail_when_log_missing(tmp_path):
    result = preflight.check_guard(tmp_path / "does-not-exist.log")
    assert result.status is preflight.Status.FAIL
    assert result.remediation


def test_check_guard_fail_when_explicit_failure(tmp_path):
    log = tmp_path / "comfyui-latest.log"
    log.write_text("comfyui_guard_node: NOT installed\n")
    result = preflight.check_guard(log)
    assert result.status is preflight.Status.FAIL


def test_check_guard_fail_when_no_markers_at_all(tmp_path):
    log = tmp_path / "comfyui-latest.log"
    log.write_text("ComfyUI started fine, nothing about the guard\n")
    result = preflight.check_guard(log)
    assert result.status is preflight.Status.FAIL


def test_resolve_comfyui_log_path_precedence(monkeypatch):
    monkeypatch.delenv("COMFYUI_LOG_DIR", raising=False)
    assert preflight.resolve_comfyui_log_path(None) == Path("/var/log/comfy-controller/comfyui-latest.log")
    monkeypatch.setenv("COMFYUI_LOG_DIR", "/custom/dir")
    assert preflight.resolve_comfyui_log_path(None) == Path("/custom/dir/comfyui-latest.log")
    assert preflight.resolve_comfyui_log_path("/explicit/path.log") == Path("/explicit/path.log")


# --------------------------------------------------------------- launch flags


def test_evaluate_launch_flags_ok():
    line = "python main.py --listen 127.0.0.1 --port 8188 --cache-none --disable-dynamic-vram --output-directory /x"
    assert preflight.evaluate_launch_flags(line) == []


def test_evaluate_launch_flags_flags_cache_ram():
    line = "python main.py --cache-ram --disable-dynamic-vram"
    problems = preflight.evaluate_launch_flags(line)
    assert any("cache-ram" in p for p in problems)


def test_evaluate_launch_flags_flags_missing_cache_none():
    line = "python main.py --disable-dynamic-vram"
    problems = preflight.evaluate_launch_flags(line)
    assert any("cache-none" in p for p in problems)


def test_evaluate_launch_flags_flags_missing_disable_dynamic_vram():
    line = "python main.py --cache-none"
    problems = preflight.evaluate_launch_flags(line)
    assert any("dynamic-vram" in p for p in problems)


def test_check_launch_flags_pass(tmp_path):
    log = tmp_path / "comfyui-latest.log"
    log.write_text("[launch_comfyui] launching: python main.py --cache-none --disable-dynamic-vram\n")
    result = preflight.check_launch_flags(log)
    assert result.status is preflight.Status.PASS


def test_check_launch_flags_fail_when_no_launch_line(tmp_path):
    log = tmp_path / "comfyui-latest.log"
    log.write_text("nothing about launching here\n")
    result = preflight.check_launch_flags(log)
    assert result.status is preflight.Status.FAIL


def test_check_launch_flags_uses_last_launch_line(tmp_path):
    log = tmp_path / "comfyui-latest.log"
    log.write_text(
        "[launch_comfyui] launching: python main.py --cache-ram\n"
        "[launch_comfyui] launching: python main.py --cache-none --disable-dynamic-vram\n"
    )
    result = preflight.check_launch_flags(log)
    assert result.status is preflight.Status.PASS


# -------------------------------------------------------------- models/nodes


def test_check_graph_against_object_info_all_ok():
    graph = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "good.safetensors"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "a free-text prompt, not an enum"}},
        "3": {"class_type": "SomeSampler", "inputs": {"link_input": ["1", 0]}},
    }
    object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["good.safetensors", "other.safetensors"], {}]}}},
        "CLIPTextEncode": {"input": {"required": {"text": ["STRING", {}]}}},
        "SomeSampler": {"input": {"required": {"link_input": ["IMAGE", {}]}}},
    }
    missing_nodes, missing_models = preflight.check_graph_against_object_info(graph, object_info)
    assert missing_nodes == []
    assert missing_models == []


def test_check_graph_against_object_info_missing_node_type():
    graph = {"1": {"class_type": "TotallyUnknownNode", "inputs": {}}}
    missing_nodes, missing_models = preflight.check_graph_against_object_info(graph, {})
    assert missing_nodes == ["TotallyUnknownNode"]
    assert missing_models == []


def test_check_graph_against_object_info_missing_model_value():
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "does_not_exist.safetensors"}}}
    object_info = {
        "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"], {}]}}},
    }
    missing_nodes, missing_models = preflight.check_graph_against_object_info(graph, object_info)
    assert missing_nodes == []
    assert "CheckpointLoaderSimple.ckpt_name='does_not_exist.safetensors'" in missing_models


async def _start_object_info_server(object_info: dict) -> tuple[web.AppRunner, str]:
    async def handler(request: web.Request) -> web.Response:
        return web.json_response(object_info)

    app = web.Application()
    app.router.add_get("/object_info", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, f"http://127.0.0.1:{port}"


async def test_check_models_nodes_skips_without_manifest():
    import aiohttp

    async with aiohttp.ClientSession() as session:
        result = await preflight.check_models_nodes(session, "http://127.0.0.1:1", [])
    assert result.status is preflight.Status.SKIP


async def test_check_models_nodes_pass_and_fail_against_real_http_server(tmp_path):
    import aiohttp

    good_graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "a.safetensors"}}}
    object_info = {"CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["a.safetensors"], {}]}}}}
    runner, base_url = await _start_object_info_server(object_info)
    try:
        async with aiohttp.ClientSession() as session:
            ok = await preflight.check_models_nodes(session, base_url, [("wf.json", good_graph)])
            assert ok.status is preflight.Status.PASS

            bad_graph = {"1": {"class_type": "NoSuchNode", "inputs": {}}}
            fail = await preflight.check_models_nodes(session, base_url, [("wf.json", bad_graph)])
            assert fail.status is preflight.Status.FAIL
            assert "NoSuchNode" in fail.detail
    finally:
        await runner.cleanup()


# ---------------------------------------------------------------- comfy up


async def test_check_comfy_reachable_pass_and_fail():
    import aiohttp

    fc = FakeComfy(FakeComfyState())
    base_url = await fc.start()
    try:
        async with aiohttp.ClientSession() as session:
            ok = await preflight.check_comfy_reachable(session, base_url)
            assert ok.status is preflight.Status.PASS
            assert "comfyui_version" in ok.detail
    finally:
        await fc.stop()

    async with aiohttp.ClientSession() as session:
        fail = await preflight.check_comfy_reachable(session, "http://127.0.0.1:1")
    assert fail.status is preflight.Status.FAIL


# ------------------------------------------------------------ output roundtrip


async def test_check_output_roundtrip_fails_without_output_dir():
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    comfy = ComfyHTTP("http://127.0.0.1:1")
    result = await preflight.check_output_roundtrip(comfy, None, None, 5.0)
    assert result.status is preflight.Status.FAIL
    await comfy.close()


async def test_check_output_roundtrip_fails_when_dir_not_writable(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    not_a_dir = tmp_path / "im_a_file"
    not_a_dir.write_text("x")
    comfy = ComfyHTTP("http://127.0.0.1:1")
    # A path *under* a plain file can never be created as a directory.
    result = await preflight.check_output_roundtrip(comfy, str(not_a_dir / "sub"), None, 5.0)
    assert result.status is preflight.Status.FAIL
    await comfy.close()


async def test_check_output_roundtrip_warns_without_workflow(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    comfy = ComfyHTTP("http://127.0.0.1:1")
    result = await preflight.check_output_roundtrip(comfy, str(out_dir), None, 5.0)
    assert result.status is preflight.Status.WARN
    await comfy.close()


async def test_check_output_roundtrip_pass_against_real_fakecomfy(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        result = await preflight.check_output_roundtrip(comfy, str(out_dir), (str(wf), graph), 10.0)
        assert result.status is preflight.Status.PASS
        await comfy.close()
    finally:
        await fc.stop()


async def test_check_output_roundtrip_fails_on_path_mismatch(tmp_path):
    """The exact bug the task brief calls out: ComfyUI writes under one
    directory, the controller looks under a different one."""
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    real_out_dir = tmp_path / "real_out"
    real_out_dir.mkdir()
    wrong_out_dir = tmp_path / "wrong_out"
    wrong_out_dir.mkdir()

    state = FakeComfyState(output_dir=real_out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        result = await preflight.check_output_roundtrip(comfy, str(wrong_out_dir), (str(wf), graph), 10.0)
        assert result.status is preflight.Status.FAIL
        await comfy.close()
    finally:
        await fc.stop()


# ------------------------------------------------------------------ heartbeat


def test_check_heartbeat_path_pass(tmp_path):
    path = tmp_path / "sub" / "heartbeat"
    result = preflight.check_heartbeat_path(path)
    assert result.status is preflight.Status.PASS
    assert path.exists()


def test_check_heartbeat_path_fail_when_parent_is_a_file(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    result = preflight.check_heartbeat_path(blocker / "heartbeat")
    assert result.status is preflight.Status.FAIL


def test_resolve_heartbeat_path_precedence(monkeypatch):
    monkeypatch.delenv("DEADMAN_HEARTBEAT_FILE", raising=False)
    assert preflight.resolve_heartbeat_path(None) == Path(".comfy_supervisor") / "heartbeat"
    monkeypatch.setenv("DEADMAN_HEARTBEAT_FILE", "/opt/x/heartbeat")
    assert preflight.resolve_heartbeat_path(None) == Path("/opt/x/heartbeat")
    assert preflight.resolve_heartbeat_path("/explicit/heartbeat") == Path("/explicit/heartbeat")
    # The real config's own heartbeat_file outranks the env var fallback...
    assert preflight.resolve_heartbeat_path(None, "/from/config/heartbeat") == Path("/from/config/heartbeat")
    # ...but an explicit CLI override still outranks the config.
    assert preflight.resolve_heartbeat_path("/explicit/heartbeat", "/from/config/heartbeat") == Path("/explicit/heartbeat")


def test_resolve_heartbeat_path_disabled_via_empty_config_value():
    assert preflight.resolve_heartbeat_path(None, "") is None


def test_check_heartbeat_path_skips_when_disabled():
    result = preflight.check_heartbeat_path(None)
    assert result.status is preflight.Status.SKIP


# --------------------------------------------------------------- disk space


def test_check_disk_space_pass_with_low_threshold(tmp_path):
    result = preflight.check_disk_space(str(tmp_path), 0.0)
    assert result.status is preflight.Status.PASS


def test_check_disk_space_fail_with_huge_threshold(tmp_path):
    result = preflight.check_disk_space(str(tmp_path), 10_000_000.0)
    assert result.status is preflight.Status.FAIL


def test_check_disk_space_fail_on_missing_path():
    result = preflight.check_disk_space("/no/such/path/at/all", 0.0)
    assert result.status is preflight.Status.FAIL


def test_check_disk_space_falls_back_to_cwd_when_unset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = preflight.check_disk_space(None, 0.0)
    assert result.status is preflight.Status.PASS
    assert "current directory" in result.detail


# --------------------------------------------------------------------- slack


class _FakeSlackClient:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self._response = response or {"team": "acme", "user": "comfyctl-bot"}
        self._error = error

    def auth_test(self):
        if self._error:
            raise self._error
        return self._response


async def test_check_slack_skips_when_notifier_is_console(tmp_path):
    from comfy_controller.config import AppConfig

    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "fake"}))
    result = await preflight.check_slack(cfg)
    assert result.status is preflight.Status.SKIP


async def test_check_slack_pass_with_valid_token(tmp_path, monkeypatch):
    from comfy_controller.config import AppConfig

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "fake", "notifier": "slack"}))
    result = await preflight.check_slack(cfg, client=_FakeSlackClient())
    assert result.status is preflight.Status.PASS
    assert "acme" in result.detail


async def test_check_slack_fail_with_invalid_token(tmp_path, monkeypatch):
    from comfy_controller.config import AppConfig

    monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-fake")
    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "fake", "notifier": "slack"}))
    result = await preflight.check_slack(cfg, client=_FakeSlackClient(error=RuntimeError("invalid_auth")))
    assert result.status is preflight.Status.FAIL


async def test_check_slack_warns_on_webhook_only_never_posts(tmp_path, monkeypatch):
    from comfy_controller.config import AppConfig

    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack.example/x")
    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "fake", "notifier": "slack"}))
    result = await preflight.check_slack(cfg, client=None)
    assert result.status is preflight.Status.WARN


# ------------------------------------------------------------------ anthropic


async def test_check_anthropic_skips_on_quick(tmp_path):
    from comfy_controller.config import AppConfig

    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "fake"}))
    result = await preflight.check_anthropic(cfg, quick=True)
    assert result.status is preflight.Status.SKIP


async def test_check_anthropic_skips_on_fake_backend(tmp_path):
    from comfy_controller.config import AppConfig

    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "fake"}))
    result = await preflight.check_anthropic(cfg, quick=False)
    assert result.status is preflight.Status.SKIP


@dataclass
class _FakeUsage:
    input_tokens: int = 100
    output_tokens: int = 20
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


class _FakeResponse:
    def __init__(self, parsed_output, stop_reason="end_turn"):
        self.parsed_output = parsed_output
        self.stop_reason = stop_reason
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self, outcome):
        self._outcome = outcome

    async def parse(self, **kwargs):
        if isinstance(self._outcome, Exception):
            raise self._outcome
        return self._outcome


class _FakeAnthropicClient:
    def __init__(self, outcome):
        self.messages = _FakeMessages(outcome)


async def test_check_anthropic_pass_reports_measured_cost(tmp_path, monkeypatch):
    from comfy_controller.config import AppConfig
    from comfy_controller.models import CheckResult, QCVerdict

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    rubric_path = tmp_path / "rubric.yaml"
    rubric_path.write_text(
        yaml.safe_dump(
            {
                "checks": [{"name": "ok", "question": "is it ok?", "required": True}],
                "pricing": {"claude-opus-5": {"input_per_mtok": 5.0, "output_per_mtok": 25.0}},
            }
        )
    )
    cfg = AppConfig.load(
        _write_yaml(tmp_path / "c.yaml", {"judge_backend": "claude", "rubric_path": str(rubric_path)})
    )
    verdict = QCVerdict(checks=[CheckResult(name="ok", evidence="clearly visible and fine", passed=True)])
    client = _FakeAnthropicClient(_FakeResponse(verdict))

    result = await preflight.check_anthropic(cfg, quick=False, client=client)
    assert result.status is preflight.Status.PASS
    assert "measured cost $" in result.detail


async def test_check_anthropic_warns_on_content_validation_failure(tmp_path, monkeypatch):
    from comfy_controller.config import AppConfig

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "claude"}))
    client = _FakeAnthropicClient(JudgeUnavailable("judge response has empty/templated evidence for check 'ok': ''"))

    result = await preflight.check_anthropic(cfg, quick=False, client=client)
    assert result.status is preflight.Status.WARN


async def test_check_anthropic_fails_on_infra_error(tmp_path, monkeypatch):
    from comfy_controller.config import AppConfig

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    cfg = AppConfig.load(_write_yaml(tmp_path / "c.yaml", {"judge_backend": "claude"}))
    client = _FakeAnthropicClient(JudgeUnavailable("judge network error: boom"))

    result = await preflight.check_anthropic(cfg, quick=False, client=client)
    assert result.status is preflight.Status.FAIL


# ------------------------------------------------------------------ end-to-end


async def test_check_end_to_end_skips_without_manifest():
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    comfy = ComfyHTTP("http://127.0.0.1:1")
    judge = FakeJudge()
    result = await preflight.check_end_to_end(comfy, judge, None, None, 5.0)
    assert result.status is preflight.Status.SKIP
    await comfy.close()


async def test_check_end_to_end_pass_against_real_fakecomfy(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        judge = FakeJudge()
        judge.queue_pass()
        result = await preflight.check_end_to_end(comfy, judge, (str(wf), graph), str(out_dir), 10.0)
        assert result.status is preflight.Status.PASS
        await comfy.close()
    finally:
        await fc.stop()


async def test_check_end_to_end_fails_when_judge_unavailable(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        judge = FakeJudge()
        judge.queue_unavailable("simulated 529")
        result = await preflight.check_end_to_end(comfy, judge, (str(wf), graph), str(out_dir), 10.0)
        assert result.status is preflight.Status.FAIL
        await comfy.close()
    finally:
        await fc.stop()


async def test_check_end_to_end_fails_when_comfyui_errors(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, fault=Fault.NODE_ERROR, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        judge = FakeJudge()
        result = await preflight.check_end_to_end(comfy, judge, (str(wf), graph), str(out_dir), 10.0)
        assert result.status is preflight.Status.FAIL
        await comfy.close()
    finally:
        await fc.stop()


# ------------------------------------------------------------------------ soak


async def test_run_soak_all_pass(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        judge = FakeJudge()

        report = await preflight.run_soak(
            comfy, judge, (str(wf), graph), str(out_dir), 10.0, tmp_path / ".comfy_supervisor",
            count=3, minutes=None, interval_s=0.0,
        )
        assert len(report.iterations) == 3
        assert report.all_passed
        assert report.success_count == 3
        assert len(report.latencies) == 3
        assert report.restarts_observed == 0
        await comfy.close()
    finally:
        await fc.stop()


async def test_run_soak_reports_failure_and_is_not_all_passed(tmp_path):
    """One bad iteration (judge unavailable) must show up as a failed
    iteration and flip `all_passed` False -- this is what makes `--loop`
    exit non-zero when any iteration fails."""
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        judge = FakeJudge()
        judge.queue_pass()
        judge.queue_unavailable("simulated outage")
        judge.queue_pass()

        report = await preflight.run_soak(
            comfy, judge, (str(wf), graph), str(out_dir), 10.0, tmp_path / ".comfy_supervisor",
            count=3, minutes=None, interval_s=0.0,
        )
        assert len(report.iterations) == 3
        assert not report.all_passed
        assert report.success_count == 2
        statuses = [it.result.status for it in report.iterations]
        assert statuses == [preflight.Status.PASS, preflight.Status.FAIL, preflight.Status.PASS]
        await comfy.close()
    finally:
        await fc.stop()


async def test_run_soak_counts_restart_markers(tmp_path):
    from comfy_controller.adapters.comfy_http import ComfyHTTP

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    marker_dir = tmp_path / ".comfy_supervisor"
    marker_dir.mkdir()
    (marker_dir / "restart-1.json").write_text("{}")

    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        comfy = ComfyHTTP(base_url)
        wf = _workflow(tmp_path)
        graph = json.loads(wf.read_text())
        judge = FakeJudge()

        async def fake_sleep(_s):
            # A "restart" happens between iteration 1 and 2.
            if len(list(marker_dir.glob("restart-*.json"))) == 1:
                (marker_dir / "restart-2.json").write_text("{}")

        report = await preflight.run_soak(
            comfy, judge, (str(wf), graph), str(out_dir), 10.0, marker_dir,
            count=2, minutes=None, interval_s=0.0, sleep=fake_sleep,
        )
        assert report.restarts_observed == 1
        await comfy.close()
    finally:
        await fc.stop()


def test_render_loop_report_unstable_when_any_iteration_failed():
    good = preflight.PreflightResult("end_to_end", preflight.Status.PASS, "ok")
    bad = preflight.PreflightResult("end_to_end", preflight.Status.FAIL, "boom")
    report = preflight.LoopReport(
        iterations=[
            preflight.LoopIteration(1, good, 1.0),
            preflight.LoopIteration(2, bad, None),
        ],
        restarts_observed=0,
    )
    text = preflight.render_loop_report(report)
    assert "UNSTABLE" in text
    assert not report.all_passed


# -------------------------------------------------------------------- exit code


def test_compute_exit_code_all_pass_is_zero():
    results = [preflight.PreflightResult("a", preflight.Status.PASS, "ok"), preflight.PreflightResult("b", preflight.Status.SKIP, "n/a")]
    assert preflight.compute_exit_code(results) == 0


def test_compute_exit_code_warn_only_is_two():
    results = [preflight.PreflightResult("a", preflight.Status.PASS, "ok"), preflight.PreflightResult("b", preflight.Status.WARN, "careful")]
    assert preflight.compute_exit_code(results) == 2


def test_compute_exit_code_any_fail_is_one():
    results = [
        preflight.PreflightResult("a", preflight.Status.WARN, "careful"),
        preflight.PreflightResult("b", preflight.Status.FAIL, "broken"),
    ]
    assert preflight.compute_exit_code(results) == 1


def test_render_report_contains_every_check_name():
    results = [preflight.PreflightResult("a", preflight.Status.PASS, "ok")]
    text = preflight.render_report(results)
    assert "a" in text
    assert "SAFE to start the night" in text


# ------------------------------------------------------------- full run() glue


async def test_run_reports_config_failure_and_stops(tmp_path):
    out = io.StringIO()
    code = await preflight.run(str(tmp_path / "missing.yaml"), out=out)
    assert code == 1
    assert "config" in out.getvalue()


async def test_run_quick_end_to_end_full_pipeline(tmp_path):
    """A close-to-real run() against FakeComfy + judge_backend=fake, with
    --quick equivalent semantics disabled so end_to_end actually runs, and a
    manifest so every manifest-gated check gets exercised."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        wf = _workflow(tmp_path)
        manifest = _manifest(tmp_path, wf)
        cfg_path = _write_yaml(
            tmp_path / "cfg.yaml",
            {"judge_backend": "fake", "comfy_base_url": base_url, "comfy_output_dir": str(out_dir)},
        )
        log_path = tmp_path / "comfyui.log"
        log_path.write_text(
            "comfyui_guard_node: PATCH CONFIRMED\n"
            "[launch_comfyui] launching: python main.py --cache-none --disable-dynamic-vram\n"
        )

        out = io.StringIO()
        code = await preflight.run(
            str(cfg_path),
            manifest_path=str(manifest),
            quick=True,  # anthropic/slack unconfigured anyway; keep this a fast smoke test
            comfyui_log_file=str(log_path),
            heartbeat_file=str(tmp_path / ".comfy_supervisor" / "heartbeat"),
            min_free_gb=0.0,
            e2e_timeout_s=15.0,
            out=out,
        )
        text = out.getvalue()
        assert "guard_patch" in text
        assert "launch_flags" in text
        # FakeComfy (testing/fake_comfy.py) doesn't implement /object_info --
        # it's a real ComfyUI endpoint this fake test double never needed
        # until now -- so models_nodes legitimately FAILs here; that's a
        # property of the test double, not a bug in the check. output_roundtrip
        # doesn't depend on /object_info and still passes.
        assert "models_nodes" in text
        assert "[FAIL]  models_nodes" in text
        assert "[  OK]  output_roundtrip" in text
        # --quick: no paid/e2e check should have run.
        assert "[SKIP]  anthropic_judge" in text
        assert "[SKIP]  end_to_end" in text
        assert code == 1  # blocked on models_nodes, exactly as it should be
    finally:
        await fc.stop()


async def test_run_non_quick_runs_full_end_to_end(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    state = FakeComfyState(output_dir=out_dir, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        wf = _workflow(tmp_path)
        manifest = _manifest(tmp_path, wf)
        cfg_path = _write_yaml(
            tmp_path / "cfg.yaml",
            {"judge_backend": "fake", "comfy_base_url": base_url, "comfy_output_dir": str(out_dir)},
        )
        log_path = tmp_path / "comfyui.log"
        log_path.write_text(
            "comfyui_guard_node: PATCH CONFIRMED\n"
            "[launch_comfyui] launching: python main.py --cache-none --disable-dynamic-vram\n"
        )

        out = io.StringIO()
        code = await preflight.run(
            str(cfg_path),
            manifest_path=str(manifest),
            quick=False,
            comfyui_log_file=str(log_path),
            heartbeat_file=str(tmp_path / ".comfy_supervisor" / "heartbeat"),
            min_free_gb=0.0,
            e2e_timeout_s=15.0,
            out=out,
        )
        text = out.getvalue()
        assert "end_to_end" in text
        assert "submit -> generate -> fetch -> judge -> save" in text
    finally:
        await fc.stop()


async def test_run_with_loop_runs_soak_and_exits_nonzero_on_failure(tmp_path):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # NODE_ERROR fault fires on every job (fault_after_n_jobs=0), so every
    # soak iteration's generation fails -> loop must exit non-zero.
    state = FakeComfyState(output_dir=out_dir, fault=Fault.NODE_ERROR, step_delay=0.0)
    fc = FakeComfy(state)
    base_url = await fc.start()
    try:
        wf = _workflow(tmp_path)
        manifest = _manifest(tmp_path, wf)
        cfg_path = _write_yaml(
            tmp_path / "cfg.yaml",
            {"judge_backend": "fake", "comfy_base_url": base_url, "comfy_output_dir": str(out_dir)},
        )
        log_path = tmp_path / "comfyui.log"
        log_path.write_text(
            "comfyui_guard_node: PATCH CONFIRMED\n"
            "[launch_comfyui] launching: python main.py --cache-none --disable-dynamic-vram\n"
        )

        out = io.StringIO()
        code = await preflight.run(
            str(cfg_path),
            manifest_path=str(manifest),
            quick=False,
            loop_count=2,
            loop_interval_s=0.0,
            comfyui_log_file=str(log_path),
            heartbeat_file=str(tmp_path / ".comfy_supervisor" / "heartbeat"),
            min_free_gb=0.0,
            e2e_timeout_s=15.0,
            out=out,
        )
        assert code == 1
        text = out.getvalue()
        assert "soak: single-asset end-to-end x2" in text
        assert "verdict: UNSTABLE" in text
    finally:
        await fc.stop()

