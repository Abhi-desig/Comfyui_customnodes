#!/usr/bin/env bash
# ops/on_controller_stop.sh
#
# ExecStopPost= hook for deploy/comfy-controller.service. Closes the gap the
# audit found: previously there was NO path from "batch finished" to
# "instance stopped" -- a run finishing at 03:00 sat idle (and billing)
# until layer 1's `shutdown -h +660` fired around 09:00, roughly 6 idle
# H100-hours per successful run.
#
# systemd sets $SERVICE_RESULT (and $EXIT_CODE / $EXIT_STATUS) for
# ExecStopPost= commands (systemd >= 232 -- see systemd.exec(5) and
# systemd.service(5) "Command lines" section). We only act on a genuinely
# CLEAN stop:
#
#   - SERVICE_RESULT=success means the ExecStart process exited 0 and
#     systemd is NOT going to restart it (Restart=on-failure only restarts
#     on a *failure* exit) -- i.e. the batch actually finished.
#   - Anything else (a crash, a signal, a failure that Restart=on-failure
#     may bring back up, or the StartLimitBurst ceiling being hit) is
#     deliberately left alone here. Terminating the instance from this hook
#     on every failed attempt would fight systemd's own restart discipline
#     and could stop the box mid-restart-loop instead of letting it
#     recover; that path is what Restart=on-failure, layer 2 (stale
#     heartbeat, once wired -- see ops/deadman.sh's heartbeat-contract
#     comment) and layer 1 (fixed boot ceiling) exist to catch.
#
# Safe to re-run / idempotent: a final sync is always harmless, and
# ops/deadman.sh terminate (-> terminate_instance()) is itself safe to call
# more than once.
#
# NOTE on visibility: this script does NOT swallow ops/deadman.sh's exit
# status. If termination genuinely fails (see terminate_instance()'s
# escalation ladder), this script -- and therefore ExecStopPost, and
# therefore comfy-controller.service itself -- exits non-zero, so the unit
# shows `failed` even though the batch it ran succeeded. That's
# intentional: a stopped box vs. a weekend of billing is exactly the
# distinction docs/runbook.md sect 7 asks you to verify, and a silently
# "successful" unit would hide the one case that most needs a human.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { printf '[on_controller_stop] %s\n' "$*" >&2; }

log "controller stopped: SERVICE_RESULT=${SERVICE_RESULT:-unknown} EXIT_CODE=${EXIT_CODE:-unknown} EXIT_STATUS=${EXIT_STATUS:-unknown}"

if [ "${SERVICE_RESULT:-}" != "success" ]; then
    log "not a clean stop -- leaving the instance running for Restart=on-failure and the dead-man's" \
        "switch (layers 1/2) to handle. (SERVICE_RESULT is unset/unknown on very old systemd or a" \
        "manual, non-systemd invocation of this script -- same conservative behaviour applies: do nothing.)"
    exit 0
fi

log "clean finish detected -- flushing outputs, then terminating the instance"

if ! "${SCRIPT_DIR}/sync_outputs.sh" --once; then
    log "final 'sync_outputs.sh --once' failed (best-effort) -- continuing to terminate the instance anyway;" \
        "some in-flight output may not have made it off-box."
fi

exec "${SCRIPT_DIR}/deadman.sh" terminate "comfy-controller exited cleanly (batch finished)"
