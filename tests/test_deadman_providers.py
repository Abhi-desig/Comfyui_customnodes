"""Tests for ops/deadman.sh's `runpod_terminate()` / `lambda_terminate()`
provider calls, run as a real subprocess against the actual bash script.

Both `curl` and `shutdown` are stubbed on PATH so no real network call or
shutdown ever happens:
  - the `shutdown` stub just records that it ran (see test_deadman_ops.py's
    `_stub_shutdown_bin` for why: `terminate_instance()`'s guaranteed local
    fallback always runs after the provider branch, succeeded or not).
  - the `curl` stub records its exact argv (so a test can assert on method,
    URL, and headers/body -- i.e. that the request this script builds
    actually matches the provider's documented API) and returns a
    caller-controlled HTTP status/body/exit code via env vars, so both the
    success and failure paths are exercised without hitting the network.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

DEADMAN_SH = Path(__file__).resolve().parent.parent / "ops" / "deadman.sh"

_CURL_STUB = """#!/bin/sh
log="${CURL_STUB_LOG:-}"
if [ -n "$log" ]; then
    out="["
    first=1
    for a in "$@"; do
        esc=$(printf '%s' "$a" | sed 's/\\\\/\\\\\\\\/g; s/"/\\\\"/g')
        if [ "$first" = 1 ]; then
            out="${out}\\"${esc}\\""
            first=0
        else
            out="${out},\\"${esc}\\""
        fi
    done
    out="${out}]"
    printf '%s\\n' "$out" >> "$log"
fi

out_file=""
prev=""
for a in "$@"; do
    if [ "$prev" = "-o" ]; then
        out_file="$a"
    fi
    prev="$a"
done

body="${CURL_STUB_BODY:-}"
if [ -n "$out_file" ]; then
    printf '%s' "$body" > "$out_file"
fi

printf '%s' "${CURL_STUB_HTTP_CODE:-200}"
exit "${CURL_STUB_EXIT:-0}"
"""

_SHUTDOWN_STUB = """#!/bin/sh
echo "called: $*" >> "{marker}"
exit 0
"""


def _stub_bin_dir(tmp_path: Path, *, shutdown_marker: Path) -> Path:
    bin_dir = tmp_path / "stub_bin"
    bin_dir.mkdir(exist_ok=True)

    curl_stub = bin_dir / "curl"
    curl_stub.write_text(_CURL_STUB)
    curl_stub.chmod(0o755)

    shutdown_stub = bin_dir / "shutdown"
    shutdown_stub.write_text(_SHUTDOWN_STUB.format(marker=shutdown_marker))
    shutdown_stub.chmod(0o755)

    return bin_dir


def _run_terminate(tmp_path: Path, *, extra_env: dict[str, str]) -> tuple[subprocess.CompletedProcess, list[list[str]]]:
    shutdown_marker = tmp_path / "shutdown_called.log"
    curl_log = tmp_path / "curl_calls.log"
    bin_dir = _stub_bin_dir(tmp_path, shutdown_marker=shutdown_marker)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["DEADMAN_LOG_FILE"] = str(tmp_path / "deadman.log")
    env["CURL_STUB_LOG"] = str(curl_log)
    env.update(extra_env)

    proc = subprocess.run(
        ["bash", str(DEADMAN_SH), "terminate", "test reason"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    calls: list[list[str]] = []
    if curl_log.exists():
        for line in curl_log.read_text().splitlines():
            if line.strip():
                calls.append(json.loads(line))
    proc.shutdown_marker = shutdown_marker  # type: ignore[attr-defined]
    return proc, calls


def _shutdown_was_called(proc: subprocess.CompletedProcess) -> bool:
    return proc.shutdown_marker.exists()  # type: ignore[attr-defined]


# ------------------------------------------------------------------- RunPod


def test_runpod_terminate_success_calls_documented_endpoint(tmp_path):
    proc, calls = _run_terminate(
        tmp_path,
        extra_env={
            "DEADMAN_PROVIDER": "runpod",
            "RUNPOD_API_KEY": "rp-key-123",
            "RUNPOD_POD_ID": "pod-abc",
            "CURL_STUB_HTTP_CODE": "204",
        },
    )
    assert len(calls) == 1, calls
    argv = calls[0]
    assert "-X" in argv and argv[argv.index("-X") + 1] == "DELETE"
    assert argv[-1] == "https://rest.runpod.io/v1/pods/pod-abc"
    assert "Authorization: Bearer rp-key-123" in argv

    assert "pod-abc deleted (HTTP 204)" in proc.stderr
    # The guaranteed local fallback always runs too, success or not.
    assert _shutdown_was_called(proc)


def test_runpod_terminate_failure_falls_through_to_local_shutdown(tmp_path):
    proc, calls = _run_terminate(
        tmp_path,
        extra_env={
            "DEADMAN_PROVIDER": "runpod",
            "RUNPOD_API_KEY": "rp-key-123",
            "RUNPOD_POD_ID": "pod-abc",
            "CURL_STUB_HTTP_CODE": "401",
            "CURL_STUB_BODY": '{"error": "unauthorized"}',
        },
    )
    assert len(calls) == 1
    assert "RunPod API terminate FAILED: HTTP 401" in proc.stderr
    assert "falling back to local shutdown" in proc.stderr
    assert _shutdown_was_called(proc)


def test_runpod_terminate_requires_credentials(tmp_path):
    proc, calls = _run_terminate(
        tmp_path,
        extra_env={"DEADMAN_PROVIDER": "runpod"},  # no RUNPOD_API_KEY / RUNPOD_POD_ID
    )
    assert calls == []  # never even attempted the HTTP call
    assert "RUNPOD_API_KEY / RUNPOD_POD_ID not set" in proc.stderr
    assert _shutdown_was_called(proc)


# ------------------------------------------------------------------- Lambda


def test_lambda_terminate_success_calls_documented_endpoint(tmp_path):
    proc, calls = _run_terminate(
        tmp_path,
        extra_env={
            "DEADMAN_PROVIDER": "lambda",
            "LAMBDA_API_KEY": "lambda-key-123",
            "LAMBDA_INSTANCE_ID": "i-xyz",
            "CURL_STUB_HTTP_CODE": "200",
            "CURL_STUB_BODY": '{"data": {"terminated_instances": ["i-xyz"]}}',
        },
    )
    assert len(calls) == 1, calls
    argv = calls[0]
    assert "-X" in argv and argv[argv.index("-X") + 1] == "POST"
    assert argv[-1] == "https://cloud.lambda.ai/api/v1/instance-operations/terminate"
    assert "Authorization: Bearer lambda-key-123" in argv
    assert "Content-Type: application/json" in argv
    assert "-d" in argv
    body_arg = argv[argv.index("-d") + 1]
    assert json.loads(body_arg) == {"instance_ids": ["i-xyz"]}

    assert "terminate requested (HTTP 200)" in proc.stderr
    assert _shutdown_was_called(proc)


def test_lambda_terminate_2xx_with_error_body_is_treated_as_failure(tmp_path):
    """Lambda's own documented envelope puts errors in the body even on some
    2xx responses -- a bare status-code check would wrongly call this a
    success."""
    proc, calls = _run_terminate(
        tmp_path,
        extra_env={
            "DEADMAN_PROVIDER": "lambda",
            "LAMBDA_API_KEY": "lambda-key-123",
            "LAMBDA_INSTANCE_ID": "i-xyz",
            "CURL_STUB_HTTP_CODE": "200",
            "CURL_STUB_BODY": '{"error": {"code": "invalid_instance", "message": "no such instance"}}',
        },
    )
    assert len(calls) == 1
    assert "Lambda API terminate FAILED" in proc.stderr
    assert "body contains an error" in proc.stderr
    assert _shutdown_was_called(proc)


def test_lambda_terminate_requires_credentials(tmp_path):
    proc, calls = _run_terminate(
        tmp_path,
        extra_env={"DEADMAN_PROVIDER": "lambda"},
    )
    assert calls == []
    assert "LAMBDA_API_KEY / LAMBDA_INSTANCE_ID not set" in proc.stderr
    assert _shutdown_was_called(proc)


# ---------------------------------------------------------------------- none


def test_provider_none_never_calls_curl_but_still_shuts_down(tmp_path):
    proc, calls = _run_terminate(tmp_path, extra_env={"DEADMAN_PROVIDER": "none"})
    assert calls == []
    assert _shutdown_was_called(proc)
