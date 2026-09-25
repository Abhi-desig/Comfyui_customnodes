"""Tests for config.py, manifest.py, and cli.py.

The `run`/`resume` end-to-end tests spin up a real FakeComfy test server and
point a YAML config at it with `judge_backend: fake`, so the whole
config -> manifest -> BatchRunner -> SQLiteStore chain is exercised through
the actual CLI entry point (`comfy_controller.cli.main`), not just its pieces.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from comfy_controller import cli as cli_module
from comfy_controller.adapters.sqlite_store import SQLiteStore
from comfy_controller.cli import build_parser, main
from comfy_controller.config import AppConfig, ConfigError
from comfy_controller.manifest import ManifestError, load_manifest
from comfy_controller.models import Asset, AssetState, CheckResult, QCVerdict
from comfy_controller.runner import read_all_records
from comfy_controller.testing.fake_comfy import FakeComfy, FakeComfyState


# ---------------------------------------------------------------------- config


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_config_loads_defaults_and_judge_backend_fake(tmp_path):
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "db_path": str(tmp_path / "x.db")})
    cfg = AppConfig.load(cfg_path)
    assert cfg.judge_backend == "fake"
    assert cfg.concurrency == 4
    assert cfg.anthropic_api_key is None


def test_config_rejects_secret_keys_in_yaml(tmp_path):
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "anthropic_api_key": "sk-leaked"})
    with pytest.raises(ConfigError, match="anthropic_api_key"):
        AppConfig.load(cfg_path)


def test_config_rejects_nested_secret_keys_in_yaml(tmp_path):
    cfg_path = _write_yaml(
        tmp_path / "cfg.yaml", {"judge_backend": "fake", "supervisor": {"nested": {"slack_bot_token": "xoxb-x"}}}
    )
    with pytest.raises(ConfigError, match="slack_bot_token"):
        AppConfig.load(cfg_path)


def test_config_requires_anthropic_key_env_for_claude_backend(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "claude"})
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        AppConfig.load(cfg_path)


def test_config_picks_up_secret_from_env_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "claude"})
    cfg = AppConfig.load(cfg_path)
    assert cfg.anthropic_api_key == "sk-from-env"
    # Never serialized back out (e.g. into logs) via normal model dumping.
    assert "sk-from-env" not in cfg.model_dump_json()


def test_config_env_override_non_secret_field(tmp_path, monkeypatch):
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake"})
    monkeypatch.setenv("COMFYCTL_CONCURRENCY", "9")
    cfg = AppConfig.load(cfg_path)
    assert cfg.concurrency == 9


def test_config_env_cannot_smuggle_a_secret(tmp_path, monkeypatch):
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake"})
    monkeypatch.setenv("COMFYCTL_ANTHROPIC_API_KEY", "sk-should-be-ignored")
    cfg = AppConfig.load(cfg_path)
    assert cfg.anthropic_api_key is None  # the real env var (no prefix) wasn't set


def test_config_slack_notifier_requires_token(tmp_path, monkeypatch):
    monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
    monkeypatch.delenv("SLACK_WEBHOOK_URL", raising=False)
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "notifier": "slack"})
    with pytest.raises(ConfigError, match="SLACK"):
        AppConfig.load(cfg_path)


def test_config_rejects_unknown_top_level_field(tmp_path):
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "totally_unknown_field": 1})
    with pytest.raises(ConfigError):
        AppConfig.load(cfg_path)


# -------------------------------------------------------------------- manifest


def _workflow(tmp_path: Path, name: str = "wf.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}))
    return path


def test_manifest_loads_yaml(tmp_path):
    wf = _workflow(tmp_path)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(
        yaml.safe_dump(
            [
                {"id": "a0", "workflow": str(wf), "seed": 1, "inputs": {"prompt": "hi"}},
                {"id": "a1", "workflow": str(wf)},
            ]
        )
    )
    assets = load_manifest(manifest)
    assert [a.id for a in assets] == ["a0", "a1"]
    assert assets[0].seed == 1
    assert assets[0].inputs == {"prompt": "hi"}


def test_manifest_loads_csv_with_json_inputs_column(tmp_path):
    wf = _workflow(tmp_path)
    manifest = tmp_path / "m.csv"
    manifest.write_text(
        "id,workflow,seed,inputs\n"
        f"a0,{wf},7,\"{{\"\"prompt\"\": \"\"hello\"\"}}\"\n"
    )
    assets = load_manifest(manifest)
    assert assets[0].id == "a0"
    assert assets[0].seed == 7
    assert assets[0].inputs == {"prompt": "hello"}


def test_manifest_resolves_relative_workflow_against_workflow_dir(tmp_path):
    wf_dir = tmp_path / "workflows"
    wf_dir.mkdir()
    _workflow(wf_dir, "wf.json")
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": "wf.json"}]))
    assets = load_manifest(manifest, workflow_dir=wf_dir)
    assert assets[0].workflow == str(wf_dir / "wf.json")


def test_manifest_unknown_workflow_path_raises(tmp_path):
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": "does_not_exist.json"}]))
    with pytest.raises(ManifestError, match="not found"):
        load_manifest(manifest)


def test_manifest_bad_kind_raises(tmp_path):
    wf = _workflow(tmp_path)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": str(wf), "kind": "not_a_kind"}]))
    with pytest.raises(ManifestError, match="unknown kind"):
        load_manifest(manifest)


def test_manifest_sequence_without_frame_count_raises(tmp_path):
    wf = _workflow(tmp_path)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": str(wf), "kind": "sequence"}]))
    with pytest.raises(ManifestError, match="frame_count"):
        load_manifest(manifest)


def test_manifest_sequence_with_frame_count_ok(tmp_path):
    wf = _workflow(tmp_path)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": str(wf), "kind": "sequence", "frame_count": 5}]))
    assets = load_manifest(manifest)
    assert assets[0].kind.value == "sequence"
    assert assets[0].frame_count == 5


def test_manifest_duplicate_id_raises(tmp_path):
    wf = _workflow(tmp_path)
    manifest = tmp_path / "m.yaml"
    manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": str(wf)}, {"id": "a0", "workflow": str(wf)}]))
    with pytest.raises(ManifestError, match="duplicate"):
        load_manifest(manifest)


def test_manifest_missing_file_raises():
    with pytest.raises(ManifestError, match="not found"):
        load_manifest("/no/such/manifest.yaml")


def test_manifest_unsupported_extension_raises(tmp_path):
    bad = tmp_path / "m.txt"
    bad.write_text("nope")
    with pytest.raises(ManifestError, match="unsupported"):
        load_manifest(bad)


# --------------------------------------------------------------------- CLI: admin


def _seeded_store(db_path: Path) -> None:
    import asyncio

    async def _seed():
        store = SQLiteStore(db_path)
        await store.load_batch([Asset(id="a0", workflow="w.json"), Asset(id="a1", workflow="w.json")])
        rec = await store.claim_next()
        await store.record_submission(rec.asset.id, "p1")
        await store.transition(
            rec.asset.id,
            AssetState.PARKED_APPROVAL,
            verdict=QCVerdict(checks=[CheckResult(name="c", evidence="e", passed=False)]),
            output_paths=["out.png"],
        )
        await store.close()

    asyncio.run(_seed())


def test_cli_status_reports_counts(tmp_path, capsys):
    db_path = tmp_path / "s.db"
    _seeded_store(db_path)
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "db_path": str(db_path)})

    rc = main(["status", "--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "parked_approval" in out
    assert "TOTAL" in out


def test_cli_approve_requeues_for_regeneration_not_saved(tmp_path, capsys):
    """`approve` must not silently rubber-stamp the rejected output as SAVED --
    the design intent is "generate this again", and `comfyctl approve X &&
    comfyctl resume` has to be the thing that actually regenerates it."""
    db_path = tmp_path / "s.db"
    _seeded_store(db_path)
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "db_path": str(db_path)})

    rc = main(["approve", "--config", str(cfg_path), "a0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Approved a0" in out
    assert "regenerat" in out.lower()

    store = SQLiteStore(db_path)
    import asyncio

    summary = asyncio.run(store.summary())
    records = {r.asset.id: r for r in read_all_records(store)}
    asyncio.run(store.close())

    # Requeued, not fast-forwarded to SAVED. (`_seeded_store` also loads a1,
    # untouched and already PENDING, so the total PENDING count is 2 -- a0's
    # own state is the thing that matters here.)
    assert summary.get(AssetState.SAVED, 0) == 0

    rec = records["a0"]
    assert rec.state is AssetState.PENDING
    # Fresh chances: attempt/crash_count reset, and the rejected run's own
    # verdict/output/error don't linger on a row that's about to be
    # regenerated from scratch.
    assert rec.attempt == 0
    assert rec.crash_count == 0
    assert rec.last_verdict is None
    assert rec.output_paths == []


async def test_cli_approve_then_resume_actually_regenerates(tmp_path):
    """The end-to-end contract: `approve` followed by `resume` must produce a
    brand new SAVED asset, not just flip a status flag."""
    fc = FakeComfy(FakeComfyState())
    base = await fc.start()
    try:
        wf = _workflow(tmp_path)
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": str(wf)}]))
        db_path = tmp_path / "run.db"
        cfg_path = _write_yaml(
            tmp_path / "cfg.yaml",
            {
                "judge_backend": "fake",
                "db_path": str(db_path),
                "comfy_base_url": base,
                "concurrency": 1,
                "tick_interval_s": 3600.0,
            },
        )

        # Seed a0 straight into PARKED_APPROVAL, as if a prior run's judge
        # rejected it -- never actually generated via this manifest/run.
        store = SQLiteStore(db_path)
        await store.load_batch([Asset(id="a0", workflow=str(wf))])
        rec = await store.claim_next()
        await store.record_submission(rec.asset.id, "stale-prompt-id")
        await store.transition(
            rec.asset.id,
            AssetState.PARKED_APPROVAL,
            verdict=QCVerdict(checks=[CheckResult(name="c", evidence="e", passed=False)]),
            output_paths=["stale_out.png"],
        )
        await store.close()

        # NOT `main(["approve", ...])` here: `cmd_approve` drives its own
        # top-level `asyncio.run()` (see `cmd_approve`'s docstring / the CLI
        # sync test above, which covers that path), which cannot nest inside
        # the event loop this async test itself needs to keep FakeComfy's
        # server alive -- same reasoning as `test_cli_run_end_to_end_then_
        # resume`'s own docstring for calling `_run_batch` directly.
        approve_store = SQLiteStore(db_path)
        await approve_store.requeue_for_regeneration("a0")
        await approve_store.close()

        await cli_module._run_batch(str(cfg_path), None)

        store2 = SQLiteStore(db_path)
        summary = await store2.summary()
        records = {r.asset.id: r for r in read_all_records(store2)}
        await store2.close()

        assert summary.get(AssetState.SAVED) == 1
        assert records["a0"].comfy_prompt_id != "stale-prompt-id"
        assert records["a0"].output_paths != ["stale_out.png"]
    finally:
        await fc.stop()


def test_cli_approve_unknown_asset_id_fails(tmp_path, capsys):
    db_path = tmp_path / "s.db"
    _seeded_store(db_path)
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "db_path": str(db_path)})

    rc = main(["approve", "--config", str(cfg_path), "does-not-exist"])
    assert rc == 1


def test_cli_report_prints_morning_report(tmp_path, capsys):
    db_path = tmp_path / "s.db"
    _seeded_store(db_path)
    cfg_path = _write_yaml(tmp_path / "cfg.yaml", {"judge_backend": "fake", "db_path": str(db_path)})

    rc = main(["report", "--config", str(cfg_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Morning report" in out
    assert "To-do" in out


def test_cli_argparse_requires_a_command():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_argparse_dispatches_each_subcommand():
    parser = build_parser()
    assert parser.parse_args(["run", "--config", "c.yaml", "m.yaml"]).func is cli_module.cmd_run
    assert parser.parse_args(["resume", "--config", "c.yaml"]).func is cli_module.cmd_resume
    assert parser.parse_args(["status", "--config", "c.yaml"]).func is cli_module.cmd_status
    assert parser.parse_args(["approve", "--config", "c.yaml", "a0"]).func is cli_module.cmd_approve
    assert parser.parse_args(["report", "--config", "c.yaml"]).func is cli_module.cmd_report


def test_cli_bad_config_path_reports_error_not_traceback(tmp_path, capsys):
    rc = main(["status", "--config", str(tmp_path / "missing.yaml")])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


# ---------------------------------------------------------------- CLI: run/resume


async def test_cli_run_end_to_end_then_resume(tmp_path):
    """Exercises the CLI's own wiring (`cli._run_batch`, which `cmd_run` and
    `cmd_resume` both call) against a real FakeComfy server. This calls the
    async entry point directly rather than going through `main()`'s argv
    parsing: `main()` drives its own top-level `asyncio.run()`, which cannot
    be nested inside the event loop this test itself needs to keep FakeComfy's
    aiohttp server alive across two runs. `test_cli_argparse_dispatches_each_
    subcommand` covers that `main()` reaches these same functions."""
    fc = FakeComfy(FakeComfyState())
    base = await fc.start()
    try:
        wf = _workflow(tmp_path)
        manifest = tmp_path / "m.yaml"
        manifest.write_text(yaml.safe_dump([{"id": "a0", "workflow": str(wf)}, {"id": "a1", "workflow": str(wf)}]))
        db_path = tmp_path / "run.db"
        cfg_path = _write_yaml(
            tmp_path / "cfg.yaml",
            {
                "judge_backend": "fake",
                "db_path": str(db_path),
                "comfy_base_url": base,
                "concurrency": 2,
                "tick_interval_s": 3600.0,
            },
        )

        await cli_module._run_batch(str(cfg_path), str(manifest))

        store = SQLiteStore(db_path)
        summary = await store.summary()
        await store.close()
        assert summary.get(AssetState.SAVED) == 2

        # `resume` with no manifest must be a no-op re-run against the
        # same (already-complete) DB -- it should finish immediately.
        await cli_module._run_batch(str(cfg_path), None)
        store2 = SQLiteStore(db_path)
        summary2 = await store2.summary()
        await store2.close()
        assert summary2.get(AssetState.SAVED) == 2
    finally:
        await fc.stop()
