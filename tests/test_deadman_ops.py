"""Tests for ops/deadman.sh's layer-2 heartbeat check, run as a real
subprocess against the actual bash script (not a reimplementation of its
logic in Python) -- this is the only thing that verifies the shell script
itself, since it isn't imported by anything under src/.

The real `terminate_instance()` calls `shutdown -h now`, which we must never
actually invoke from a test. Instead a stub `shutdown` executable is placed
first on PATH; it just records that it was called (and exits 0, so the
script never falls through to `escalate_shutdown`'s systemctl/sysrq
attempts, which would otherwise print scary-looking failures on a dev
machine with no real shutdown(8)).
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

DEADMAN_SH = Path(__file__).resolve().parent.parent / "ops" / "deadman.sh"


def _stub_shutdown_bin(tmp_path: Path, marker_file: Path) -> Path:
    bin_dir = tmp_path / "stub_bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "shutdown"
    stub.write_text(
        "#!/bin/sh\n"
        f'echo "called: $*" >> "{marker_file}"\n'
        "exit 0\n"
    )
    stub.chmod(0o755)
    return bin_dir


def _run_check_heartbeat(tmp_path: Path, *, extra_env: dict[str, str]) -> subprocess.CompletedProcess:
    shutdown_marker = tmp_path / "shutdown_called.log"
    bin_dir = _stub_shutdown_bin(tmp_path, shutdown_marker)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env.setdefault("DEADMAN_PROVIDER", "none")
    env["DEADMAN_LOG_FILE"] = str(tmp_path / "deadman.log")
    env.update(extra_env)

    proc = subprocess.run(
        ["bash", str(DEADMAN_SH), "check-heartbeat"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    proc.shutdown_marker = shutdown_marker  # type: ignore[attr-defined]
    return proc


def _shutdown_was_called(proc: subprocess.CompletedProcess) -> bool:
    return proc.shutdown_marker.exists()  # type: ignore[attr-defined]


# ------------------------------------------------------------- fresh start


def test_missing_heartbeat_within_grace_period_does_not_terminate(tmp_path):
    """A controller that hasn't written its first heartbeat yet (still
    starting up) must not get its box pulled out from under it."""
    heartbeat_file = tmp_path / "hb" / "heartbeat"  # deliberately never created
    boot_marker = tmp_path / "boot-marker"  # deliberately doesn't exist yet either

    proc = _run_check_heartbeat(
        tmp_path,
        extra_env={
            "DEADMAN_HEARTBEAT_FILE": str(heartbeat_file),
            "DEADMAN_BOOT_MARKER_FILE": str(boot_marker),
            "DEADMAN_STARTUP_GRACE_MIN": "15",
        },
    )

    assert proc.returncode == 0, proc.stderr
    assert not _shutdown_was_called(proc)
    assert "startup grace period" in proc.stderr
    assert boot_marker.is_file()  # the marker itself gets created on first sight


def test_missing_heartbeat_past_grace_period_terminates(tmp_path):
    """Once the grace period has genuinely elapsed, a still-missing
    heartbeat is indistinguishable from a dead controller -- fail safe."""
    heartbeat_file = tmp_path / "hb" / "heartbeat"
    boot_marker = tmp_path / "boot-marker"
    boot_marker.parent.mkdir(parents=True, exist_ok=True)
    boot_marker.write_text(str(int(time.time()) - 120))  # "first seen missing" 2 min ago

    proc = _run_check_heartbeat(
        tmp_path,
        extra_env={
            "DEADMAN_HEARTBEAT_FILE": str(heartbeat_file),
            "DEADMAN_BOOT_MARKER_FILE": str(boot_marker),
            "DEADMAN_STARTUP_GRACE_MIN": "1",  # 60s -- well under the 120s elapsed above
        },
    )

    assert proc.returncode == 0, proc.stderr  # the stub "shutdown" succeeds
    assert _shutdown_was_called(proc)
    assert "past" in proc.stderr and "startup grace" in proc.stderr
    assert "CRITICAL: terminating instance" in proc.stderr


# ------------------------------------------------------------ ordinary cases


def test_fresh_heartbeat_is_healthy_and_never_terminates(tmp_path):
    heartbeat_file = tmp_path / "heartbeat"
    heartbeat_file.write_text('{"ts": 0, "iteration": 1}')

    proc = _run_check_heartbeat(
        tmp_path,
        extra_env={
            "DEADMAN_HEARTBEAT_FILE": str(heartbeat_file),
            "DEADMAN_HEARTBEAT_STALE_MIN": "20",
        },
    )

    assert proc.returncode == 0, proc.stderr
    assert not _shutdown_was_called(proc)
    assert "heartbeat OK" in proc.stderr


def test_stale_heartbeat_terminates_regardless_of_startup_grace(tmp_path):
    """A heartbeat that EXISTS but has gone stale is the core failure mode
    (controller dead or wedged mid-run) -- must terminate even though the
    file's mere existence proves startup finished long ago."""
    heartbeat_file = tmp_path / "heartbeat"
    heartbeat_file.write_text('{"ts": 0, "iteration": 1}')
    old = time.time() - 3600  # 1h old
    os.utime(heartbeat_file, (old, old))

    proc = _run_check_heartbeat(
        tmp_path,
        extra_env={
            "DEADMAN_HEARTBEAT_FILE": str(heartbeat_file),
            "DEADMAN_HEARTBEAT_STALE_MIN": "20",  # stale after 20min; this one is 60min old
        },
    )

    assert proc.returncode == 0, proc.stderr  # the stub "shutdown" succeeds
    assert _shutdown_was_called(proc)
    assert "heartbeat stale" in proc.stderr
    assert "CRITICAL: terminating instance" in proc.stderr


@pytest.mark.parametrize("bad_state", ["never_created", "stale"])
def test_status_subcommand_runs_without_provider_configured(tmp_path, bad_state):
    """Smoke test for `ops/deadman.sh status` (used by the runbook's manual
    verification step) -- it must be a pure read, never touching shutdown."""
    heartbeat_file = tmp_path / "heartbeat"
    if bad_state == "stale":
        heartbeat_file.write_text("x")
        old = time.time() - 3600
        os.utime(heartbeat_file, (old, old))

    env = dict(os.environ)
    env["DEADMAN_LOG_FILE"] = str(tmp_path / "deadman.log")
    env["DEADMAN_HEARTBEAT_FILE"] = str(heartbeat_file)
    env["DEADMAN_PROVIDER"] = "none"

    proc = subprocess.run(
        ["bash", str(DEADMAN_SH), "status"],
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, proc.stderr
    assert "dead-man's switch status" in proc.stdout
