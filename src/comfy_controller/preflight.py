"""`comfyctl preflight`: a self-check of the REAL environment before trusting
an unattended overnight run against ~500 assets.

Every check below answers one question a human would otherwise have to
answer by hand at 11pm before walking away from the box: "is this actually
going to hold up all night?" Each check is independent, is defensive about
its own failure modes (a check crashing must never take down the rest of the
report -- see `_safe_async`/`_safe_sync`), and reports exactly one of
PASS / WARN / FAIL / SKIP with a concrete remediation line. The point is a
human deciding "safe to start the night?" at a glance, not a stack trace.

Design choices worth calling out:

* **The guard-node check is the single most important one here.** If
  `comfyui_guard_node/` silently failed to patch, issue #16312's
  zombie-`prompt_worker` behaviour is back, and nothing else in this
  codebase can detect it from the outside -- `/system_stats` keeps
  answering 200 straight through it (see `adapters/comfy_http.py`'s
  `is_responding` docstring). The guard node contributes no HTTP-visible
  node of its own (`NODE_CLASS_MAPPINGS = {}}` at the bottom of
  `comfyui_guard_node/__init__.py`), so the only place its verdict is
  observable at all is the ComfyUI log -- the exact "PATCH CONFIRMED" /
  "NOT installed" / "verification FAILED" lines `ops/launch_comfyui.sh`
  itself blocks startup on. This check greps the same log for the same
  markers.
* **Only two checks cost real money or a real generation against a
  possibly-loaded model**: the Anthropic judge call, and the full
  end-to-end pipeline. `quick=True` skips exactly those two and nothing
  else.
* **Nothing here is guessed when it can't be checked honestly.** Node/model
  existence, the output round-trip, and the end-to-end check all need a
  real workflow to test against; without `--manifest` they report SKIP
  with a remediation, never a vacuous PASS.
* **Nothing here shares state with a real batch run.** The end-to-end and
  soak checks use their own throwaway SQLite path, never `cfg.db_path` --
  a preflight run must never corrupt the ledger a real `run`/`resume`
  depends on.
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import enum
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, TextIO

import aiohttp
import structlog

from .adapters.comfy_http import ComfyHTTP, ComfyValidationError
from .adapters.sqlite_store import SQLiteStore
from .config import AppConfig, ConfigError
from .manifest import ManifestError, load_manifest
from .models import Asset, AssetState
from .ports import JudgeUnavailable

log = structlog.get_logger(__name__)


# --------------------------------------------------------------------- result


class Status(str, enum.Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


_STATUS_LABEL = {Status.PASS: "OK", Status.WARN: "WARN", Status.FAIL: "FAIL", Status.SKIP: "SKIP"}


@dataclass(frozen=True)
class PreflightResult:
    name: str
    status: Status
    detail: str
    remediation: str | None = None


def compute_exit_code(results: list[PreflightResult]) -> int:
    """0 all good, 1 a blocking failure, 2 warnings only. SKIP never blocks --
    it means "not applicable without --manifest", not "something is wrong"."""
    if any(r.status is Status.FAIL for r in results):
        return 1
    if any(r.status is Status.WARN for r in results):
        return 2
    return 0


def render_result(r: PreflightResult) -> str:
    lines = [f"[{_STATUS_LABEL[r.status]:>4}]  {r.name:<20}  {r.detail}"]
    if r.remediation and r.status in (Status.FAIL, Status.WARN):
        lines.append(f"           -> {r.remediation}")
    return "\n".join(lines)


# --------------------------------------------------------------- safety nets


async def _safe_async(name: str, coro: Awaitable[PreflightResult]) -> PreflightResult:
    """A check crashing on something unexpected must never take the rest of
    the report down with it -- this is the outer safety net; each check also
    handles its OWN expected failure modes (network errors, bad files, ...)
    with a tailored remediation before this ever has to fire."""
    try:
        return await coro
    except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
        log.exception("preflight_check_crashed", check=name)
        return PreflightResult(
            name, Status.FAIL, f"check crashed: {type(exc).__name__}: {exc}",
            "this looks like a bug in the check itself rather than the environment -- see the log above.",
        )


def _safe_sync(name: str, fn, *a: Any, **kw: Any) -> PreflightResult:
    try:
        return fn(*a, **kw)
    except Exception as exc:  # noqa: BLE001
        log.exception("preflight_check_crashed", check=name)
        return PreflightResult(
            name, Status.FAIL, f"check crashed: {type(exc).__name__}: {exc}",
            "this looks like a bug in the check itself rather than the environment -- see the log above.",
        )


# ------------------------------------------------------------------- helpers


async def _get_json(session: aiohttp.ClientSession, base_url: str, path: str) -> dict[str, Any]:
    async with session.get(
        f"{base_url.rstrip('/')}{path}", timeout=aiohttp.ClientTimeout(total=10)
    ) as resp:
        resp.raise_for_status()
        return await resp.json()


def _read_text_best_effort(path: Path, max_bytes: int = 2_000_000) -> str | None:
    """The tail of `path`, decoded leniently, or None if it can't be read at
    all. Only the tail matters -- a night-long log can be huge, and the
    markers we grep for are only ever written once near the very top of a
    (re)launch, well within this window for any log that hasn't been
    rotated mid-read."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) > max_bytes:
        data = data[-max_bytes:]
    return data.decode("utf-8", errors="replace")


def _image_problem(path: Path) -> str | None:
    """None when `path` is a readable, parseable image; a reason otherwise.
    Mirrors runner.py's `_image_integrity_problem` -- duplicated rather than
    imported, since that's a private helper in a file this task does not own."""
    try:
        if not path.is_file():
            return "not a regular file"
        if path.stat().st_size == 0:
            return "0-byte file"
    except OSError as exc:
        return f"{type(exc).__name__}: {exc}"
    try:
        from PIL import Image

        with Image.open(path) as img:
            img.verify()
    except Exception as exc:  # noqa: BLE001 - any parse failure means "don't trust it"
        return f"unparseable image: {type(exc).__name__}: {exc}"
    return None


def _resolve_output_path(output_path: str, base: Path | None) -> Path:
    path = Path(output_path)
    if base is not None and not path.is_absolute():
        path = base / path
    return path


# ------------------------------------------------------------------- config


def check_config(config_path: str | Path) -> tuple[PreflightResult, AppConfig | None]:
    try:
        cfg = AppConfig.load(config_path)
    except ConfigError as exc:
        return (
            PreflightResult("config", Status.FAIL, str(exc), "fix the config file/environment and re-run."),
            None,
        )
    # Which secrets are present, never their values.
    secrets = {
        "ANTHROPIC_API_KEY": cfg.anthropic_api_key is not None,
        "SLACK_BOT_TOKEN": cfg.slack_bot_token is not None,
        "SLACK_WEBHOOK_URL": cfg.slack_webhook_url is not None,
    }
    secret_summary = ", ".join(f"{k}={'set' if v else 'unset'}" for k, v in secrets.items())
    return PreflightResult("config", Status.PASS, f"config loaded from {config_path}; {secret_summary}"), cfg


# ------------------------------------------------------------------ comfy up


async def check_comfy_reachable(session: aiohttp.ClientSession, base_url: str) -> PreflightResult:
    try:
        data = await _get_json(session, base_url, "/system_stats")
    except Exception as exc:  # noqa: BLE001 - any of these means "not reachable"
        return PreflightResult(
            "comfy_reachable", Status.FAIL, f"{type(exc).__name__}: {exc}",
            f"start ComfyUI (see ops/launch_comfyui.sh) or fix comfy_base_url ({base_url!r}) in the config.",
        )
    version = (data.get("system") or {}).get("comfyui_version", "unknown")
    return PreflightResult("comfy_reachable", Status.PASS, f"reachable; comfyui_version={version}")


# ------------------------------------------------------------------ guard node


_GUARD_CONFIRM_RE = re.compile(r"PATCH CONFIRMED")
_GUARD_FAIL_RE = re.compile(r"NOT installed|verification FAILED")


def resolve_comfyui_log_path(override: str | None) -> Path:
    if override:
        return Path(override)
    # Matches ops/launch_comfyui.sh's own COMFYUI_LOG_DIR default and its
    # comfyui-latest.log symlink -- the same file that script blocks
    # startup on grepping.
    log_dir = os.environ.get("COMFYUI_LOG_DIR", "/var/log/comfy-controller")
    return Path(log_dir) / "comfyui-latest.log"


def parse_guard_status(log_text: str) -> str:
    """'confirmed' | 'failed' | 'absent'. Scans the WHOLE text and lets the
    LAST relevant line win, so a stale failure line from an earlier boot
    doesn't shadow a later, successful (re)launch."""
    status = "absent"
    for line in log_text.splitlines():
        if _GUARD_CONFIRM_RE.search(line):
            status = "confirmed"
        elif _GUARD_FAIL_RE.search(line):
            status = "failed"
    return status


def check_guard(log_path: Path) -> PreflightResult:
    text = _read_text_best_effort(log_path)
    if text is None:
        return PreflightResult(
            "guard_patch", Status.FAIL, f"could not read ComfyUI's log at {log_path}",
            "confirm ComfyUI was started via ops/launch_comfyui.sh (which writes this log), or pass "
            "--comfyui-log-file / set COMFYUI_LOG_DIR to the right path, then re-run. This is the single "
            "most important check here -- do not start an overnight run without it passing.",
        )
    status = parse_guard_status(text)
    if status == "confirmed":
        return PreflightResult(
            "guard_patch", Status.PASS, f"guard node confirmed the #16312 patch is active (see {log_path})"
        )
    if status == "failed":
        return PreflightResult(
            "guard_patch", Status.FAIL,
            f"the guard node explicitly reported it did NOT install/confirm the patch (see {log_path})",
            "read comfyui_guard_node/__init__.py's own warning line in the log for why (import error, or "
            "execution.PromptExecutor renamed upstream) before trusting an unattended run.",
        )
    return PreflightResult(
        "guard_patch", Status.FAIL,
        f"no guard-node confirmation OR failure line found in {log_path} -- cannot confirm the #16312 "
        "mitigation is active",
        "the guard node may never have loaded, or its logging changed; re-run ops/launch_comfyui.sh and "
        "confirm it reaches its own 'guard node confirmed' log line before trusting an unattended run.",
    )


# ------------------------------------------------------------------ launch flags


_LAUNCH_LINE_RE = re.compile(r"launching:\s*(.*)$")


def parse_last_launch_command(log_text: str) -> str | None:
    last = None
    for line in log_text.splitlines():
        m = _LAUNCH_LINE_RE.search(line)
        if m:
            last = m.group(1)
    return last


def evaluate_launch_flags(cmd_line: str) -> list[str]:
    problems = []
    if "--cache-ram" in cmd_line:
        problems.append("--cache-ram is explicitly set -- this is the exact #16312 trigger (RAMPressureCache)")
    elif "--cache-none" not in cmd_line:
        problems.append("--cache-none is not set -- ComfyUI defaults to --cache-ram, the #16312 trigger")
    if "--disable-dynamic-vram" not in cmd_line:
        problems.append("--disable-dynamic-vram is not set -- DynamicVRAM is ComfyUI's default and has an open OOM regression")
    return problems


def check_launch_flags(log_path: Path) -> PreflightResult:
    text = _read_text_best_effort(log_path)
    if text is None:
        return PreflightResult(
            "launch_flags", Status.FAIL, f"could not read ComfyUI's log at {log_path}",
            "same file as guard_patch -- fix log discovery first (--comfyui-log-file / COMFYUI_LOG_DIR).",
        )
    cmd_line = parse_last_launch_command(text)
    if cmd_line is None:
        return PreflightResult(
            "launch_flags", Status.FAIL, f"no 'launching: ...' line found in {log_path}",
            "confirm ComfyUI is started via ops/launch_comfyui.sh, which logs its exact argv before exec.",
        )
    problems = evaluate_launch_flags(cmd_line)
    if problems:
        return PreflightResult(
            "launch_flags", Status.FAIL, "; ".join(problems),
            "relaunch via ops/launch_comfyui.sh unmodified -- it always passes both flags (see its own "
            "comment on why each one is deliberate).",
        )
    return PreflightResult("launch_flags", Status.PASS, f"--cache-none and --disable-dynamic-vram both present: {cmd_line.strip()}")


# ------------------------------------------------------------------ models/nodes


def check_graph_against_object_info(
    graph: dict[str, Any], object_info: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Returns (missing_node_types, missing_model_or_option_values).

    A node's `class_type` must be a key in `/object_info`. Beyond that, only
    WIDGET values that ComfyUI's own schema declares as a fixed
    enum/combo (`[[choice, ...], {...}]` -- e.g. `ckpt_name`) are checked
    against the live choice list; a link reference (`[node_id, output_idx]`)
    is skipped as not a widget value, and a freeform STRING input (e.g. a
    CLIPTextEncode prompt) has no enum to check against and is skipped too.
    """
    missing_nodes: set[str] = set()
    missing_models: list[str] = []
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        class_type = node.get("class_type")
        if not class_type:
            continue
        info = object_info.get(class_type)
        if info is None:
            missing_nodes.add(class_type)
            continue
        schema: dict[str, Any] = {}
        input_spec = info.get("input", {}) if isinstance(info, dict) else {}
        for group in ("required", "optional"):
            schema.update(input_spec.get(group) or {})
        for key, value in (node.get("inputs") or {}).items():
            if isinstance(value, list):
                continue  # a link [node_id, output_index], not a widget value
            spec = schema.get(key)
            if not (isinstance(spec, list) and spec and isinstance(spec[0], list)):
                continue  # not an enum/combo input (e.g. free text) -- nothing to check
            if value not in spec[0]:
                missing_models.append(f"{class_type}.{key}={value!r}")
    return sorted(missing_nodes), missing_models


async def check_models_nodes(
    session: aiohttp.ClientSession, base_url: str, workflows: list[tuple[str, dict[str, Any]]]
) -> PreflightResult:
    if not workflows:
        return PreflightResult(
            "models_nodes", Status.SKIP, "no --manifest given",
            "pass --manifest to verify the configured workflow(s)' node types and model files exist on this ComfyUI.",
        )
    try:
        object_info = await _get_json(session, base_url, "/object_info")
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(
            "models_nodes", Status.FAIL, f"could not fetch /object_info: {type(exc).__name__}: {exc}",
            "confirm ComfyUI is reachable (see comfy_reachable) before trusting this run.",
        )

    all_missing_nodes: set[str] = set()
    all_missing_models: list[str] = []
    for label, graph in workflows:
        nodes, models = check_graph_against_object_info(graph, object_info)
        all_missing_nodes.update(nodes)
        all_missing_models.extend(f"{label}: {m}" for m in models)

    if all_missing_nodes or all_missing_models:
        parts = []
        if all_missing_nodes:
            parts.append(f"missing node type(s): {', '.join(sorted(all_missing_nodes))}")
        if all_missing_models:
            parts.append(f"missing model/option value(s): {'; '.join(all_missing_models)}")
        return PreflightResult(
            "models_nodes", Status.FAIL, "; ".join(parts),
            "install the missing custom node(s) and/or copy the missing model file(s) onto this box before "
            "trusting the run -- a mid-batch 'unknown node' rejection wastes GPU hours for nothing.",
        )
    return PreflightResult(
        "models_nodes", Status.PASS,
        f"every node type and model reference in {len(workflows)} configured workflow(s) exists on this ComfyUI",
    )


# ---------------------------------------------------------------- generation


async def _generate_once(
    comfy: ComfyHTTP, graph: dict[str, Any], job_key: str, timeout: float
) -> tuple[Any, str | None]:
    """Submit `graph` and wait for its outcome. Returns (JobOutcome, None) on
    success, or (None, reason) on any failure -- never raises, so callers can
    turn a failure straight into a FAIL result without their own try/except."""
    try:
        prompt_id = await comfy.submit(graph, job_key)
    except ComfyValidationError as exc:
        return None, f"ComfyUI rejected the graph: {exc}"
    except Exception as exc:  # noqa: BLE001
        return None, f"submit failed: {type(exc).__name__}: {exc}"

    try:
        outcome = await comfy.await_outcome(prompt_id, timeout=timeout)
    except asyncio.TimeoutError:
        return None, f"timed out after {timeout:.0f}s waiting for prompt {prompt_id}"
    except Exception as exc:  # noqa: BLE001
        return None, f"await_outcome failed: {type(exc).__name__}: {exc}"

    if not outcome.succeeded:
        err = outcome.error
        msg = err.exception_message if err else "unknown failure"
        return None, f"ComfyUI reported a failed job: {msg}"
    return outcome, None


# ------------------------------------------------------------- output round trip


async def check_output_roundtrip(
    comfy: ComfyHTTP,
    output_dir: str | None,
    workflow: tuple[str, dict[str, Any]] | None,
    timeout: float,
) -> PreflightResult:
    if not output_dir:
        return PreflightResult(
            "output_roundtrip", Status.FAIL, "comfy_output_dir is not set in the config",
            "set comfy_output_dir to the SAME path ComfyUI's --output-directory points at (see "
            "ops/launch_comfyui.sh's COMFY_OUTPUT_DIR comment) -- without it the controller cannot "
            "resolve ComfyUI's reported output paths at all.",
        )
    out_dir = Path(output_dir)
    marker = out_dir / f".preflight-write-test-{uuid.uuid4().hex}"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text("preflight")
        marker.unlink()
    except OSError as exc:
        return PreflightResult(
            "output_roundtrip", Status.FAIL, f"{output_dir} is not writable: {type(exc).__name__}: {exc}",
            f"mkdir -p {output_dir} and confirm the controller's user can write there.",
        )

    if workflow is None:
        return PreflightResult(
            "output_roundtrip", Status.WARN,
            f"{output_dir} exists and is writable, but no --manifest was given so the full round trip "
            "(a file ComfyUI itself writes) was not verified",
            "pass --manifest to verify the controller can read back a real ComfyUI-written file -- this "
            "exact output-path-mismatch bug has shipped once already.",
        )

    label, graph = workflow
    job_key = f"preflight-roundtrip-{uuid.uuid4().hex[:8]}"
    outcome, err = await _generate_once(comfy, graph, job_key, timeout)
    if outcome is None:
        return PreflightResult(
            "output_roundtrip", Status.FAIL, err or "generation failed",
            "a real generation against this workflow failed -- see comfy_reachable/models_nodes first.",
        )
    if not outcome.output_paths:
        return PreflightResult(
            "output_roundtrip", Status.FAIL, "ComfyUI reported success but produced no output files",
            f"confirm the workflow {label!r} has a SaveImage (or equivalent) node.",
        )

    problems = []
    for p in outcome.output_paths:
        resolved = _resolve_output_path(p, out_dir)
        problem = _image_problem(resolved)
        if problem:
            problems.append(f"{resolved}: {problem}")
    if problems:
        return PreflightResult(
            "output_roundtrip", Status.FAIL, "; ".join(problems),
            "confirm comfy_output_dir matches ComfyUI's own --output-directory EXACTLY (see "
            "ops/launch_comfyui.sh's COMFY_OUTPUT_DIR / COMFYCTL_COMFY_OUTPUT_DIR comment) -- this is the "
            "exact output-path-mismatch bug that has shipped before.",
        )
    return PreflightResult(
        "output_roundtrip", Status.PASS,
        f"controller read back {len(outcome.output_paths)} file(s) ComfyUI just wrote under {out_dir}",
    )


# ------------------------------------------------------------------- heartbeat


def resolve_heartbeat_path(override: str | None, cfg_heartbeat_file: str | None = None) -> Path | None:
    """None means heartbeat writing is intentionally disabled (an empty
    `heartbeat_file` in the config -- see config.py's own comment on that
    field) -- distinct from "not configured", which falls through to the
    env var / default below.

    Precedence: an explicit `--heartbeat-file` override, then the real
    config's own `heartbeat_file` (the path `RunnerConfig`/`HealthSupervisor`
    will ACTUALLY use -- see runner.py's `_maybe_touch_heartbeat`), then
    `$DEADMAN_HEARTBEAT_FILE` (for a config that predates that field), then
    the same hardcoded default `HealthSupervisor`'s own `marker_dir` uses.
    """
    if override:
        return Path(override)
    if cfg_heartbeat_file is not None:
        return Path(cfg_heartbeat_file) if cfg_heartbeat_file else None
    env = os.environ.get("DEADMAN_HEARTBEAT_FILE")
    if env:
        return Path(env)
    return Path(".comfy_supervisor") / "heartbeat"


def check_heartbeat_path(path: Path | None) -> PreflightResult:
    if path is None:
        return PreflightResult(
            "heartbeat_path", Status.SKIP, "heartbeat writing is disabled (config's heartbeat_file is empty)"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"preflight_check": True, "ts": time.time()}))
    except OSError as exc:
        return PreflightResult(
            "heartbeat_path", Status.FAIL, f"{path} is not writable: {type(exc).__name__}: {exc}",
            f"mkdir -p {path.parent} and confirm the controller's user can write there -- layer 2 of the "
            "dead-man's switch (ops/deadman.sh) fails *safe* on a missing/stale heartbeat, which means "
            "every run terminates early if this path is wrong.",
        )
    return PreflightResult("heartbeat_path", Status.PASS, f"{path} is writable")


# ---------------------------------------------------------------- disk space


def check_disk_space(output_dir: str | None, min_free_gb: float) -> PreflightResult:
    path = output_dir or "."
    note = "" if output_dir else " (comfy_output_dir not set; checked the current directory instead)"
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return PreflightResult(
            "disk_space", Status.FAIL, f"could not stat {path}: {type(exc).__name__}: {exc}",
            "confirm the output directory exists (see output_roundtrip) before trusting this run.",
        )
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        return PreflightResult(
            "disk_space", Status.FAIL,
            f"only {free_gb:.1f} GiB free at {path}{note} (threshold {min_free_gb:.1f} GiB)",
            "free up space or point comfy_output_dir at a bigger volume before an overnight run.",
        )
    return PreflightResult("disk_space", Status.PASS, f"{free_gb:.1f} GiB free at {path}{note}")


# ------------------------------------------------------------------- anthropic


# Smallest valid PNG: 1x1, fully transparent -- decodable by Pillow. Same
# bytes testing/fake_comfy.py uses as its bundled output stand-in.
_PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)

# Judge response messages that mean "the API call itself worked, but its
# content failed our own post-hoc validation" -- see
# adapters/claude_judge.py's `_validate_content`. Very plausibly caused by
# the placeholder image having nothing real to describe, not a broken API.
_CONTENT_VALIDATION_MARKERS = ("templated evidence", "missing required checks", "structured-output validation")


def _write_placeholder_image(path: Path) -> None:
    path.write_bytes(_PLACEHOLDER_PNG)


async def check_anthropic(
    cfg: AppConfig, *, quick: bool, client: Any = None, test_image_path: str | None = None
) -> PreflightResult:
    if quick:
        return PreflightResult("anthropic_judge", Status.SKIP, "--quick")
    if cfg.judge_backend != "claude":
        return PreflightResult(
            "anthropic_judge", Status.SKIP, f"judge_backend={cfg.judge_backend!r}; no real Anthropic call configured"
        )
    if not cfg.anthropic_api_key:
        return PreflightResult(
            "anthropic_judge", Status.FAIL, "ANTHROPIC_API_KEY is not set", "set ANTHROPIC_API_KEY in the environment."
        )

    from .adapters.claude_judge import ClaudeJudge, RubricConfig
    from .adapters.prefilter import PrefilterConfig

    try:
        rubric = RubricConfig.from_yaml(cfg.rubric_path) if Path(cfg.rubric_path).exists() else RubricConfig()
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(
            "anthropic_judge", Status.FAIL, f"could not load {cfg.rubric_path}: {exc}",
            "fix config/rubric.yaml before trusting the real judge.",
        )
    # One cheap call, not the real self-consistency N -- this check only
    # wants to know "is the API reachable and roughly what does it cost",
    # not "would the judge approve this image". The prefilter is disabled
    # too: it's a free, deterministic pre-check (adapters/prefilter.py) that
    # would otherwise reject the bundled placeholder image (a 1x1 pixel has
    # zero sharpness/brightness) before the network is ever touched --
    # correct behaviour for a real asset, but it would silently defeat the
    # one thing THIS check exists to prove: that the paid API is reachable.
    rubric = dataclasses.replace(
        rubric,
        self_consistency_n=1,
        prefilter=PrefilterConfig(
            min_variance_of_laplacian=0.0,
            brightness_min=0.0,
            brightness_max=255.0,
            aspect_ratio_min=0.0,
            aspect_ratio_max=1_000.0,
            max_palette_distance=0.0,
            edge_density_threshold=1.1,
        ),
    )

    if client is None:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)

    judge = ClaudeJudge(client, rubric)

    with tempfile.TemporaryDirectory(prefix="comfyctl-preflight-") as tmp:
        if test_image_path:
            image_path = Path(test_image_path)
        else:
            image_path = Path(tmp) / "preflight-test.png"
            _write_placeholder_image(image_path)

        asset = Asset(id="preflight-check", workflow="")
        try:
            await judge.judge(asset, [image_path])
        except JudgeUnavailable as exc:
            message = str(exc)
            if any(marker in message for marker in _CONTENT_VALIDATION_MARKERS):
                return PreflightResult(
                    "anthropic_judge", Status.WARN,
                    f"Anthropic API responded, but the judge rejected its own response's content: {message}",
                    "very likely caused by the bundled placeholder image having nothing real to describe, "
                    "not an API problem -- pass --test-image with a real sample asset for a cleaner signal.",
                )
            return PreflightResult(
                "anthropic_judge", Status.FAIL, message,
                "check ANTHROPIC_API_KEY, network egress to api.anthropic.com, and the model name in "
                "config/rubric.yaml.",
            )
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                "anthropic_judge", Status.FAIL, f"{type(exc).__name__}: {exc}",
                "check ANTHROPIC_API_KEY and network egress to api.anthropic.com.",
            )

    cost = judge.cost.total_usd
    return PreflightResult("anthropic_judge", Status.PASS, f"judge call succeeded; measured cost ${cost:.4f}")


# ---------------------------------------------------------------------- slack


async def check_slack(cfg: AppConfig, *, client: Any = None) -> PreflightResult:
    if cfg.notifier != "slack":
        return PreflightResult("slack", Status.SKIP, "notifier=console; Slack not configured")

    if client is None and cfg.slack_bot_token:
        from slack_sdk import WebClient

        client = WebClient(token=cfg.slack_bot_token)

    if client is not None:
        try:
            resp = await asyncio.to_thread(client.auth_test)
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(
                "slack", Status.FAIL, f"auth.test failed: {type(exc).__name__}: {exc}",
                "check SLACK_BOT_TOKEN is valid and has not been revoked/rotated.",
            )
        team = resp.get("team") if hasattr(resp, "get") else getattr(resp, "team", None)
        user = resp.get("user") if hasattr(resp, "get") else getattr(resp, "user", None)
        return PreflightResult("slack", Status.PASS, f"auth.test OK -- team={team!r} user={user!r} (no message posted)")

    if cfg.slack_webhook_url:
        return PreflightResult(
            "slack", Status.WARN,
            "only SLACK_WEBHOOK_URL is set -- an incoming webhook has no auth.test equivalent, and posting "
            "through it would spam the channel just to validate it",
            "set SLACK_BOT_TOKEN as well so this check can validate credentials without posting, or accept "
            "the risk and let the first real alert be the first test.",
        )
    return PreflightResult(
        "slack", Status.FAIL, "notifier=slack but neither SLACK_BOT_TOKEN nor SLACK_WEBHOOK_URL is set",
        "set one of them in the environment.",
    )


# ------------------------------------------------------------------ end-to-end


def _build_preflight_judge(cfg: AppConfig) -> Any:
    """Mirrors cli.py's own `_build_judge` -- duplicated rather than imported
    since cli.py is the wiring layer and importing business logic FROM it
    would run the dependency the wrong way round; this is ~6 lines."""
    if cfg.judge_backend == "fake":
        from .testing.fake_judge import FakeJudge

        return FakeJudge()
    import anthropic

    from .adapters.claude_judge import ClaudeJudge, RubricConfig

    rubric = RubricConfig.from_yaml(cfg.rubric_path)
    client = anthropic.AsyncAnthropic(api_key=cfg.anthropic_api_key)
    return ClaudeJudge(client, rubric)


async def check_end_to_end(
    comfy: ComfyHTTP,
    judge: Any,
    workflow: tuple[str, dict[str, Any]] | None,
    output_dir: str | None,
    timeout: float,
) -> PreflightResult:
    """submit -> generate -> fetch output -> judge -> save, against the real
    ComfyUI `comfy` is pointed at. "save" is a throwaway SQLiteStore, never
    `cfg.db_path` -- see module docstring."""
    if workflow is None:
        return PreflightResult(
            "end_to_end", Status.SKIP, "no --manifest given",
            "pass --manifest to run a full submit -> generate -> fetch -> judge -> save check against a real asset.",
        )
    label, graph = workflow
    t0 = time.monotonic()
    job_key = f"preflight-e2e-{uuid.uuid4().hex[:8]}"
    outcome, err = await _generate_once(comfy, graph, job_key, timeout)
    if outcome is None:
        return PreflightResult(
            "end_to_end", Status.FAIL, err or "generation failed",
            "a real generation against this workflow failed -- see comfy_reachable/models_nodes first.",
        )

    resolved = [_resolve_output_path(p, Path(output_dir) if output_dir else None) for p in outcome.output_paths]
    for p in resolved:
        problem = _image_problem(p)
        if problem:
            return PreflightResult(
                "end_to_end", Status.FAIL, f"output unreadable: {p}: {problem}",
                "see output_roundtrip for the same failure in isolation.",
            )

    asset = Asset(id=f"preflight-{job_key}", workflow=label)
    try:
        verdict = await judge.judge(asset, resolved)
    except JudgeUnavailable as exc:
        return PreflightResult(
            "end_to_end", Status.FAIL, f"judge unavailable: {exc}",
            "see anthropic_judge for the same failure in isolation.",
        )

    try:
        with tempfile.TemporaryDirectory(prefix="comfyctl-preflight-db-") as tmp:
            store = SQLiteStore(Path(tmp) / "preflight.db")
            try:
                await store.load_batch([asset])
                rec = await store.claim_next()
                if rec is None:
                    raise RuntimeError("claim_next returned no record right after load_batch")
                await store.record_submission(asset.id, outcome.prompt_id)
                await store.transition(asset.id, AssetState.SAVED, verdict=verdict, output_paths=outcome.output_paths)
                summary = await store.summary()
            finally:
                await store.close()
    except Exception as exc:  # noqa: BLE001
        return PreflightResult(
            "end_to_end", Status.FAIL, f"store/save step failed: {type(exc).__name__}: {exc}",
            "this points at a StorePort/SQLite problem, not a ComfyUI/judge one.",
        )

    if summary.get(AssetState.SAVED) != 1:
        return PreflightResult(
            "end_to_end", Status.FAIL, "store did not record the asset as SAVED after the pipeline completed",
            "this points at a StorePort/SQLite problem, not a ComfyUI/judge one.",
        )

    elapsed = time.monotonic() - t0
    verdict_note = (
        "passed rubric" if verdict.passed else f"failed rubric ({', '.join(verdict.failed_checks)}) -- still a valid pipeline exercise"
    )
    return PreflightResult(
        "end_to_end", Status.PASS,
        f"submit -> generate -> fetch -> judge -> save all succeeded in {elapsed:.1f}s ({verdict_note})",
    )


# ------------------------------------------------------------------------ soak


def _count_restart_markers(marker_dir: Path) -> int:
    """HealthSupervisor._write_marker (supervisor.py) drops one
    restart-<ms>.json per real restart into its marker_dir -- the heartbeat
    file's own parent directory (see resolve_heartbeat_path). Counting these
    across a soak is a genuine, non-heuristic "did ComfyUI actually restart"
    signal, not a guess."""
    if not marker_dir.exists():
        return 0
    return len(list(marker_dir.glob("restart-*.json")))


@dataclass
class LoopIteration:
    index: int
    result: PreflightResult
    latency_s: float | None


@dataclass
class LoopReport:
    iterations: list[LoopIteration] = field(default_factory=list)
    restarts_observed: int = 0

    @property
    def success_count(self) -> int:
        return sum(1 for it in self.iterations if it.result.status is Status.PASS)

    @property
    def all_passed(self) -> bool:
        return bool(self.iterations) and self.success_count == len(self.iterations)

    @property
    def latencies(self) -> list[float]:
        return [it.latency_s for it in self.iterations if it.latency_s is not None]


async def run_soak(
    comfy: ComfyHTTP,
    judge: Any,
    workflow: tuple[str, dict[str, Any]] | None,
    output_dir: str | None,
    timeout: float,
    marker_dir: Path,
    *,
    count: int | None,
    minutes: float | None,
    interval_s: float,
    sleep=asyncio.sleep,
    clock=time.monotonic,
) -> LoopReport:
    """Repeat `check_end_to_end` on a cadence and report stability: success
    rate, latency spread, and any ComfyUI restart observed along the way.
    Always runs at least one iteration; stops on whichever of `count` /
    `minutes` is reached first (or after exactly one iteration if neither is
    given)."""
    start_restarts = _count_restart_markers(marker_dir)
    deadline = clock() + minutes * 60.0 if minutes is not None else None

    iterations: list[LoopIteration] = []
    i = 0
    while True:
        i += 1
        t0 = clock()
        result = await check_end_to_end(comfy, judge, workflow, output_dir, timeout)
        elapsed = clock() - t0
        latency = elapsed if result.status is Status.PASS else None
        iterations.append(LoopIteration(i, result, latency))

        reached_count = count is not None and i >= count
        reached_time = deadline is not None and clock() >= deadline
        no_budget_given = count is None and deadline is None
        if reached_count or reached_time or no_budget_given:
            break
        await sleep(interval_s)

    restarts_observed = _count_restart_markers(marker_dir) - start_restarts
    return LoopReport(iterations=iterations, restarts_observed=restarts_observed)


def render_loop_report(report: LoopReport) -> str:
    n = len(report.iterations)
    lines = [f"soak: single-asset end-to-end x{n}", "-" * 78]
    for it in report.iterations:
        lat = f"{it.latency_s:.2f}s" if it.latency_s is not None else "--"
        lines.append(f"  iter {it.index}/{n}  [{_STATUS_LABEL[it.result.status]:>4}]  latency={lat:<8}  {it.result.detail}")

    latencies = report.latencies
    if latencies:
        spread = f"min={min(latencies):.2f}s max={max(latencies):.2f}s mean={statistics.mean(latencies):.2f}s"
        if len(latencies) > 1:
            spread += f" stdev={statistics.pstdev(latencies):.2f}s"
    else:
        spread = "n/a (no successful iteration)"

    lines.append("-" * 78)
    lines.append(
        f"success rate: {report.success_count}/{n}  latency: {spread}  restarts observed: {report.restarts_observed}"
    )
    lines.append("verdict: STABLE" if report.all_passed else "verdict: UNSTABLE -- at least one iteration failed")
    return "\n".join(lines)


# -------------------------------------------------------------------- report


def render_report(results: list[PreflightResult]) -> str:
    lines = ["comfyctl preflight", "=" * 78]
    lines.extend(render_result(r) for r in results)
    lines.append("-" * 78)

    counts = Counter(r.status for r in results)
    summary = ", ".join(f"{counts.get(s, 0)} {s.value}" for s in Status)
    code = compute_exit_code(results)
    if code == 0:
        verdict = "SAFE to start the night"
    elif code == 1:
        verdict = "NOT SAFE -- blocking failure(s) above"
    else:
        verdict = "safe with warnings -- review above"
    lines.append(f"{summary} -- {verdict}")
    return "\n".join(lines)


# ----------------------------------------------------------------- entrypoint


async def run(
    config_path: str,
    *,
    manifest_path: str | None = None,
    quick: bool = False,
    loop_count: int | None = None,
    loop_minutes: float | None = None,
    loop_interval_s: float = 30.0,
    comfyui_log_file: str | None = None,
    heartbeat_file: str | None = None,
    min_free_gb: float = 10.0,
    e2e_timeout_s: float = 300.0,
    test_image_path: str | None = None,
    out: TextIO = sys.stdout,
) -> int:
    results: list[PreflightResult] = []

    config_result, cfg = check_config(config_path)
    results.append(config_result)
    if cfg is None:
        print(render_report(results), file=out)
        return compute_exit_code(results)

    workflows: list[tuple[str, dict[str, Any]]] = []
    if manifest_path:
        try:
            assets = load_manifest(manifest_path, workflow_dir=cfg.workflow_dir, max_frame_count=cfg.max_frame_count)
        except ManifestError as exc:
            results.append(
                PreflightResult(
                    "manifest", Status.FAIL, str(exc), "fix --manifest before trusting checks that depend on it."
                )
            )
            assets = []
        seen: set[str] = set()
        for asset in assets:
            if asset.workflow in seen:
                continue
            seen.add(asset.workflow)
            try:
                graph = json.loads(Path(asset.workflow).read_text())
            except (OSError, ValueError) as exc:
                results.append(
                    PreflightResult(
                        "manifest_workflow", Status.FAIL, f"could not load workflow {asset.workflow}: {exc}",
                        "fix or remove this row from the manifest before trusting the run.",
                    )
                )
                continue
            if isinstance(graph, dict):
                workflows.append((asset.workflow, graph))

    first_workflow = workflows[0] if workflows else None
    loop_report: LoopReport | None = None

    async with aiohttp.ClientSession() as session:
        results.append(await _safe_async("comfy_reachable", check_comfy_reachable(session, cfg.comfy_base_url)))

        log_path = resolve_comfyui_log_path(comfyui_log_file)
        results.append(_safe_sync("guard_patch", check_guard, log_path))
        results.append(_safe_sync("launch_flags", check_launch_flags, log_path))
        results.append(await _safe_async("models_nodes", check_models_nodes(session, cfg.comfy_base_url, workflows)))

        comfy = ComfyHTTP(cfg.comfy_base_url, session=session)
        try:
            results.append(
                await _safe_async(
                    "output_roundtrip",
                    check_output_roundtrip(comfy, cfg.comfy_output_dir, first_workflow, e2e_timeout_s),
                )
            )

            heartbeat_path = resolve_heartbeat_path(heartbeat_file, getattr(cfg, "heartbeat_file", None))
            results.append(_safe_sync("heartbeat_path", check_heartbeat_path, heartbeat_path))
            # The restart-marker directory HealthSupervisor writes to is a
            # separate concern from the heartbeat FILE above (disabling the
            # heartbeat via an empty `heartbeat_file` doesn't disable restart
            # markers) -- fall back to its own hardcoded default when the
            # heartbeat path couldn't be resolved to a concrete location.
            marker_dir = heartbeat_path.parent if heartbeat_path is not None else Path(".comfy_supervisor")
            results.append(_safe_sync("disk_space", check_disk_space, cfg.comfy_output_dir, min_free_gb))
            results.append(
                await _safe_async("anthropic_judge", check_anthropic(cfg, quick=quick, test_image_path=test_image_path))
            )
            results.append(await _safe_async("slack", check_slack(cfg)))

            if quick:
                results.append(PreflightResult("end_to_end", Status.SKIP, "--quick"))
            elif loop_count is not None or loop_minutes is not None:
                judge = _build_preflight_judge(cfg)
                loop_report = await run_soak(
                    comfy, judge, first_workflow, cfg.comfy_output_dir, e2e_timeout_s, marker_dir,
                    count=loop_count, minutes=loop_minutes, interval_s=loop_interval_s,
                )
            else:
                judge = _build_preflight_judge(cfg)
                results.append(
                    await _safe_async(
                        "end_to_end", check_end_to_end(comfy, judge, first_workflow, cfg.comfy_output_dir, e2e_timeout_s)
                    )
                )
        finally:
            await comfy.close()

    print(render_report(results), file=out)

    exit_code = compute_exit_code(results)
    if loop_report is not None:
        print(file=out)
        print(render_loop_report(loop_report), file=out)
        if not loop_report.all_passed:
            exit_code = max(exit_code, 1)
    return exit_code
