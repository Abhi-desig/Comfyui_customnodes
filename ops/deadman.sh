#!/usr/bin/env bash
# ops/deadman.sh
#
# The dead-man's switch for an unattended overnight run on a rented GPU pod
# (RunPod Secure Cloud / Lambda on-demand / Vast). Highest-value file in
# ops/: everything else here just makes the run go better; this is what
# caps the damage when it doesn't.
#
# Why three independent layers, and why the outer one must not depend on
# the controller being alive: an idle H100 left running over a weekend is
# roughly $95, and none of RunPod / Vast / Lambda auto-stops an idle *Pod*
# (RunPod's idle-timeout auto-stop is a Serverless-only feature -- it does
# not apply to a Secure Cloud Pod). So "the controller will handle it" is
# not a safe assumption; the controller is exactly the thing that can fail
# to even start.
#
#   Layer 1 -- arm-boot-shutdown (this script, run once at boot, BEFORE the
#              controller starts): `shutdown -h +N`. Holds even if the
#              controller, or this very script's layer 2, never runs at
#              all. This is the backstop for "nothing else on this box
#              works." It is intentionally NOT re-armed by anything --
#              re-arming it would make it depend on something being alive,
#              which defeats the point.
#
#   Layer 2 -- check-heartbeat (this script, run every few minutes by
#              deploy/comfy-deadman.timer): if the controller's heartbeat
#              file has gone stale, terminate the instance via the cloud
#              provider's API (pluggable -- see terminate_instance()
#              below), falling back to a local `shutdown -h now` (and, if
#              that itself fails, an escalation ladder -- see
#              escalate_shutdown()) if the provider call fails or isn't
#              configured. This is the layer that actually saves most of
#              the money in the common failure mode (the controller process
#              dies or wedges mid-run); layer 1 is its backstop, not a
#              replacement for it.
#
#              A missing heartbeat file is treated as fail-safe (terminate)
#              EXCEPT during DEADMAN_STARTUP_GRACE_MIN after boot/restart
#              (see check_heartbeat()/`_seconds_since_boot_marker()` below):
#              the controller (and, before it, ComfyUI's own model load) can
#              legitimately take a few minutes to come up and write its
#              first heartbeat, and without a grace period layer 2 would
#              terminate a perfectly healthy box that just hasn't finished
#              starting yet.
#
#              The SAME terminate_instance() is also invoked proactively,
#              via `ops/deadman.sh terminate <reason>`, from
#              deploy/comfy-controller.service's ExecStopPost= the instant
#              the controller exits cleanly after a finished batch -- so a
#              successful overnight run doesn't have to wait for the
#              heartbeat to go stale (or for layer 1's hard ceiling) before
#              the box stops billing.
#
#   Layer 3 -- in-process budget counters inside the controller itself
#              (e.g. HealthSupervisor's restart-intensity ceiling, and any
#              spend/job budget the runner tracks). These are DEFENSE IN
#              DEPTH ONLY. They cannot substitute for layers 1/2: a dead
#              controller counts nothing, checks nothing, and enforces
#              nothing. Every budget counter in this codebase lives inside
#              the process that layer 1 exists specifically to not trust.
#
# Usage:
#   ops/deadman.sh arm-boot-shutdown   # layer 1 -- see deploy/comfy-deadman-boot.service
#   ops/deadman.sh check-heartbeat     # layer 2 -- see deploy/comfy-deadman.service + .timer
#   ops/deadman.sh terminate <reason>  # unconditional terminate -- see
#                                       # deploy/comfy-controller.service's ExecStopPost
#                                       # (fires proactively on a CLEAN controller exit,
#                                       # instead of waiting for the heartbeat to go stale)
#   ops/deadman.sh status              # human-readable report of both layers
#
# Safe to re-run: arm-boot-shutdown simply reschedules the same fixed
# ceiling (systemd's `shutdown` overwrites any previously scheduled
# shutdown rather than stacking), and check-heartbeat is a pure read unless
# it actually decides to terminate. `terminate` is intentionally NOT
# idempotent-safe to call twice in the sense of "harmless no-op" -- calling
# it again just re-attempts termination, which is fine (terminating an
# already-terminating/terminated instance is not harmful).

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration (override via environment; see ops/env.example)

# Layer 1: fixed ceiling from boot. Generous enough to cover a legitimate
# long overnight batch with margin, short enough that "the controller never
# started" doesn't run the meter all weekend.
DEADMAN_HARD_CEILING_MIN="${DEADMAN_HARD_CEILING_MIN:-660}"   # 11h

# Layer 2: heartbeat contract -- implemented by
# `BatchRunner._maybe_touch_heartbeat` in src/comfy_controller/runner.py
# (this file only reads the mtime side of the contract). See also
# docs/runbook.md's "Heartbeat contract" section.
#   - Path: DEADMAN_HEARTBEAT_FILE below (default matches HealthSupervisor's
#     existing marker_dir, default .comfy_supervisor/, so operators have one
#     state directory to know about instead of two).
#   - The controller touches (updates the mtime of) this file from its own
#     worker loop -- not a detached timer -- at least once per
#     `RunnerConfig.heartbeat_interval_s` (default 30s, comfortably more
#     often than DEADMAN_HEARTBEAT_STALE_MIN below).
#   - Only mtime is read (_heartbeat_age_s() below); file CONTENT is
#     ignored, so a bare `touch` satisfies the contract. The controller
#     writes a small JSON blob (e.g. {"ts": ..., "iteration": N}), which is a
#     nice-to-have for a human tailing the file by hand, never a requirement.
#   - The parent directory must exist and be writable by whatever user
#     comfy-controller.service runs as (`User=` in that unit); the controller
#     creates it with `mkdir -p` on its first write rather than assuming it's
#     there.
#   - Before the controller's first write (still starting up, e.g. ComfyUI's
#     own model load), the file does not exist yet. check_heartbeat() below
#     tolerates that for DEADMAN_STARTUP_GRACE_MIN -- past that, a still-
#     missing file is treated the same as a dead controller (fail *safe*,
#     not a false alarm).
DEADMAN_HEARTBEAT_FILE="${DEADMAN_HEARTBEAT_FILE:-/opt/comfy-controller/.comfy_supervisor/heartbeat}"
DEADMAN_HEARTBEAT_STALE_MIN="${DEADMAN_HEARTBEAT_STALE_MIN:-20}"

# How long a MISSING heartbeat file is tolerated as "still starting up"
# rather than "dead". Anchored to DEADMAN_BOOT_MARKER_FILE (below), a
# tmpfs-backed marker this script itself creates the first time it notices
# the heartbeat file is absent -- which, because comfy-deadman.timer's
# OnBootSec=2min fires shortly after boot, approximates "time since boot"
# (or since the controller's last restart, if the heartbeat file went
# missing again later -- see check_heartbeat()'s own comment on why that's
# also the right behaviour, not just an artifact of this implementation).
DEADMAN_STARTUP_GRACE_MIN="${DEADMAN_STARTUP_GRACE_MIN:-15}"
# /run is tmpfs: cleared on every reboot, which is exactly the "since boot"
# semantics this marker needs. Override for tests/non-systemd hosts.
DEADMAN_BOOT_MARKER_FILE="${DEADMAN_BOOT_MARKER_FILE:-/run/comfy-controller/deadman-boot-seen}"

# Cloud provider for layer 2's API-based termination. One of:
# runpod | lambda | vast | none
DEADMAN_PROVIDER="${DEADMAN_PROVIDER:-none}"

DEADMAN_LOG_FILE="${DEADMAN_LOG_FILE:-/var/log/comfy-controller/deadman.log}"

# log() must NEVER be able to fail the script. Under `set -euo pipefail`, an
# unwritable DEADMAN_LOG_FILE (read-only/full /var/log, or the unit running
# as a user without permission to it) makes `tee` return non-zero, and
# pipefail propagates that as the whole pipeline's exit status -- which,
# unguarded, kills the script via `set -e`. That is exactly the auditor's
# reproduction: terminate_instance() used to call log() as its first
# statement, so a merely-unwritable log path meant `shutdown` never ran at
# all. The `|| true` on both lines below is the fix; see also
# terminate_instance()'s own comment for the second, structural half of it.
log() {
    mkdir -p "$(dirname "${DEADMAN_LOG_FILE}")" 2>/dev/null || true
    printf '[deadman] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${DEADMAN_LOG_FILE}" >&2 || true
}

# --------------------------------------------------------------------------
# Layer 1

arm_boot_shutdown() {
    log "arming layer-1 hard ceiling: shutdown -h +${DEADMAN_HARD_CEILING_MIN} (fixed; not re-armed by anything)"

    # `shutdown -h +N` overwrites any previously scheduled shutdown rather
    # than stacking, which is what makes re-running this safe. But
    # shutdown(8)'s `+N` form talks to systemd-logind over D-Bus, and this
    # unit intentionally runs at `After=local-fs.target` -- before
    # dbus.service/systemd-logind.service are necessarily up (see
    # deploy/comfy-deadman-boot.service's own comment). If that call fails,
    # do NOT just log a warning and move on: fall back to a mechanism that
    # does not depend on logind at all, because this is the one layer that
    # is supposed to hold even when everything else on the box is broken.
    if shutdown -h "+${DEADMAN_HARD_CEILING_MIN}" \
        "comfy-controller dead-man's switch layer 1: unconditional shutdown ${DEADMAN_HARD_CEILING_MIN}min after boot"
    then
        log "layer-1 armed via 'shutdown -h +N'. This will fire even if the controller and layer 2 never run."
        return 0
    fi

    log "WARNING: 'shutdown -h +${DEADMAN_HARD_CEILING_MIN}' failed -- likely systemd-logind/D-Bus isn't up yet" \
        "(early boot) or shutdown(8) isn't wired to it on this host (common in a bare container). Falling" \
        "back to a mechanism that does not depend on logind."

    if arm_boot_shutdown_via_at; then
        return 0
    fi

    if arm_boot_shutdown_via_background_sleep; then
        return 0
    fi

    log "CRITICAL: every layer-1 arming mechanism failed -- this instance has NO boot-ceiling shutdown armed." \
        "This is the single highest-severity state this script can be in; see docs/runbook.md sect 6."
    return 1
}

# Fallback A: an `at` job. Still goes through userspace (atd), but does not
# require D-Bus/logind the way `shutdown -h +N` does.
arm_boot_shutdown_via_at() {
    command -v at >/dev/null 2>&1 || { log "'at' is not installed -- skipping the at(1) fallback"; return 1; }

    if command -v systemctl >/dev/null 2>&1; then
        if ! systemctl is-active --quiet atd 2>/dev/null && ! pgrep -x atd >/dev/null 2>&1; then
            log "'at' is installed but atd does not appear to be running -- an 'at' job would never fire; skipping"
            return 1
        fi
    fi

    if printf '%s\n' "systemctl poweroff --force --force 2>/dev/null || echo o > /proc/sysrq-trigger 2>/dev/null || true" \
        | at "now + ${DEADMAN_HARD_CEILING_MIN} minutes" >/dev/null 2>&1
    then
        log "layer-1 armed via an 'at' job (fires in ${DEADMAN_HARD_CEILING_MIN}min, independent of logind)."
        return 0
    fi

    log "'at' job submission failed"
    return 1
}

# Fallback B: a fully detached background process. Last resort -- bypasses
# shutdown(8)/logind/atd entirely: it just sleeps for the ceiling in its own
# session (so it survives this script, and this unit, exiting) and then
# powers off directly. Deliberately minimal (no logging inside the child)
# so this fallback can't itself fail for a reason unrelated to actually
# powering the box off.
arm_boot_shutdown_via_background_sleep() {
    command -v setsid >/dev/null 2>&1 || {
        log "'setsid' is not available -- cannot fully detach the background-sleep fallback; skipping"
        return 1
    }

    ceiling_s=$(( DEADMAN_HARD_CEILING_MIN * 60 ))

    setsid sh -c "sleep ${ceiling_s} && { systemctl poweroff --force --force 2>/dev/null || echo o > /proc/sysrq-trigger 2>/dev/null || true; }" \
        </dev/null >/dev/null 2>&1 &
    disown

    log "layer-1 armed via a detached background sleep+poweroff (fires in ${DEADMAN_HARD_CEILING_MIN}min /" \
        "${ceiling_s}s, independent of logind/atd)."
    return 0
}

# --------------------------------------------------------------------------
# Layer 2

_heartbeat_age_s() {
    if [ ! -f "${DEADMAN_HEARTBEAT_FILE}" ]; then
        echo "-1"
        return
    fi
    now="$(date +%s)"
    mtime="$(stat -c %Y "${DEADMAN_HEARTBEAT_FILE}" 2>/dev/null || stat -f %m "${DEADMAN_HEARTBEAT_FILE}")"
    echo "$(( now - mtime ))"
}

# Seconds since this script first noticed the heartbeat file was missing, in
# THIS boot (or since the controller's most recent restart -- see below).
# Creates DEADMAN_BOOT_MARKER_FILE on first call and returns 0; every
# subsequent call returns the elapsed time since then.
_seconds_since_startup_marker() {
    if [ ! -f "${DEADMAN_BOOT_MARKER_FILE}" ]; then
        mkdir -p "$(dirname "${DEADMAN_BOOT_MARKER_FILE}")" 2>/dev/null || true
        date +%s > "${DEADMAN_BOOT_MARKER_FILE}" 2>/dev/null || true
        echo "0"
        return
    fi
    now="$(date +%s)"
    marker_ts="$(cat "${DEADMAN_BOOT_MARKER_FILE}" 2>/dev/null || echo "${now}")"
    echo "$(( now - marker_ts ))"
}

check_heartbeat() {
    age_s="$(_heartbeat_age_s)"
    stale_after_s=$(( DEADMAN_HEARTBEAT_STALE_MIN * 60 ))

    if [ "${age_s}" -lt 0 ]; then
        grace_s=$(( DEADMAN_STARTUP_GRACE_MIN * 60 ))
        since_startup_s="$(_seconds_since_startup_marker)"

        if [ "${since_startup_s}" -lt "${grace_s}" ]; then
            log "heartbeat file ${DEADMAN_HEARTBEAT_FILE} does not exist yet, but we're still within the" \
                "${DEADMAN_STARTUP_GRACE_MIN}min startup grace period (${since_startup_s}s since first seen" \
                "missing) -- the controller (or ComfyUI's own model load) may still be starting. Not terminating."
            return
        fi

        log "WARNING: heartbeat file ${DEADMAN_HEARTBEAT_FILE} still does not exist after the" \
            "${DEADMAN_STARTUP_GRACE_MIN}min startup grace period (${since_startup_s}s) -- treating this the" \
            "same as a dead controller: fail safe, not a false alarm."
        terminate_instance "heartbeat file never created (past ${DEADMAN_STARTUP_GRACE_MIN}min startup grace)"
        return
    fi

    # The heartbeat file exists now, so startup is clearly done. Clear the
    # marker so that if the file goes missing again LATER (e.g. the
    # controller restarts and hasn't written its first heartbeat yet), that
    # later gap gets its own fresh grace period rather than inheriting
    # however little was left of this one -- a restarting controller needs
    # the same startup time a first boot does.
    rm -f "${DEADMAN_BOOT_MARKER_FILE}" 2>/dev/null || true

    if [ "${age_s}" -ge "${stale_after_s}" ]; then
        log "heartbeat is ${age_s}s old (stale after ${stale_after_s}s) -- controller looks dead"
        terminate_instance "heartbeat stale (${age_s}s)"
        return
    fi

    log "heartbeat OK (${age_s}s old, threshold ${stale_after_s}s)"
}

# --------------------------------------------------------------------------
# Provider-pluggable termination
#
# Every branch here falls through to a local `shutdown -h now` if the
# provider call fails, isn't configured, or the provider is "none". Local
# shutdown is the guaranteed action this script can always take without any
# cloud credentials -- but it is a WEAKER guarantee than the provider API
# call: shutting down the guest OS halts the OS, it does not necessarily
# release/deallocate the rented resource or stop billing on every provider.
# The provider API call is the one that reliably does that. Configure
# DEADMAN_PROVIDER for your actual host; do not rely on local shutdown alone
# for a real overnight run.

terminate_instance() {
    reason="$1"
    provider_terminated=0

    log "CRITICAL: terminating instance (reason: ${reason}); provider=${DEADMAN_PROVIDER}"

    case "${DEADMAN_PROVIDER}" in
        runpod)
            if runpod_terminate; then provider_terminated=1
            else log "runpod_terminate failed or not configured -- falling back to local shutdown"; fi
            ;;
        lambda)
            if lambda_terminate; then provider_terminated=1
            else log "lambda_terminate failed or not configured -- falling back to local shutdown"; fi
            ;;
        vast)
            if vast_terminate; then provider_terminated=1
            else log "vast_terminate failed or not configured -- falling back to local shutdown"; fi
            ;;
        none)   log "DEADMAN_PROVIDER=none -- no provider API configured, going straight to local shutdown" ;;
        *)      log "unknown DEADMAN_PROVIDER=${DEADMAN_PROVIDER} -- falling back to local shutdown" ;;
    esac

    # ------------------------------------------------------------------
    # Guaranteed local fallback. This is the part the auditor's repro broke:
    # terminate_instance() used to call log() as its very first statement,
    # so an unwritable log file + `set -e`/pipefail meant the `shutdown`
    # line below never ran at all. log() is now non-fatal on its own (see
    # its definition), but this is also structured so the shutdown attempt
    # itself is not gated behind, or preceded by, any logging call --
    # belt and suspenders.
    #
    # Also unlike before: a failing `shutdown -h now` is no longer silently
    # swallowed with `|| true`. RunPod Pods (and similar rented containers)
    # frequently have no systemd/shutdown(8) at all, so treating that
    # failure as success would let the unit report a clean run while the
    # box, and the bill, keep going. On failure we escalate, and if every
    # escalation path is exhausted we exit non-zero so the systemd unit
    # shows `failed` -- visible, not silent.
    shutdown_rc=0
    shutdown -h now "comfy-controller dead-man's switch layer 2: ${reason}" || shutdown_rc=$?

    if [ "${shutdown_rc}" -eq 0 ]; then
        log "issued local 'shutdown -h now' successfully (provider_terminated=${provider_terminated}, reason: ${reason})"
        return 0
    fi

    log "CRITICAL: 'shutdown -h now' failed or is absent on this host (rc=${shutdown_rc}) -- escalating" \
        "(provider_terminated=${provider_terminated})"

    if escalate_shutdown; then
        # Normally we won't reach this line at all -- the box powers off
        # first. If we do, the escalation call itself reported success.
        return 0
    fi

    log "CRITICAL: every local shutdown/escalation path failed -- instance is NOT confirmed stopped." \
        "provider=${DEADMAN_PROVIDER} provider_terminated=${provider_terminated} reason=${reason}." \
        "Go to the provider's console NOW -- see docs/runbook.md sect 7."
    return 1
}

# Escalation ladder for when `shutdown -h now` itself fails or is absent
# (common on a container-based rented GPU pod with no systemd/init as PID 1).
# Each step is progressively less graceful and less dependent on userspace
# machinery actually working.
escalate_shutdown() {
    if command -v systemctl >/dev/null 2>&1; then
        log "escalation 1/2: systemctl poweroff --force --force"
        if systemctl poweroff --force --force 2>/dev/null; then
            return 0
        fi
        log "systemctl poweroff --force --force did not succeed (or there's no working systemd on this host)"
    else
        log "systemctl not present on this host -- skipping that escalation step"
    fi

    if [ -w /proc/sysrq-trigger ]; then
        log "escalation 2/2: echo o > /proc/sysrq-trigger (immediate, uncontrolled power-off -- last resort)"
        echo o > /proc/sysrq-trigger 2>/dev/null || true
        # A working sysrq trigger powers the box off essentially
        # immediately; reaching this line at all means it didn't.
        return 1
    fi

    log "/proc/sysrq-trigger is not writable on this host (typical for an unprivileged container) -- no" \
        "further local escalation is possible"
    return 1
}

runpod_terminate() {
    # RunPod REST API, confirmed directly against RunPod's own API reference
    # (https://docs.runpod.io/api-reference/pods/DELETE/pods/podId --
    # "Delete a Pod"):
    #   DELETE https://rest.runpod.io/v1/pods/{podId}
    #   Authorization: Bearer <RUNPOD_API_KEY>
    #   204 = deleted; 400 = invalid pod id; 401 = unauthorized.
    # (RunPod also has an older GraphQL API and a v2 REST surface; this is
    # the current, documented v1 REST endpoint for exactly this operation.)
    if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
        log "RUNPOD_API_KEY / RUNPOD_POD_ID not set -- required for DEADMAN_PROVIDER=runpod"
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        log "'curl' is not installed -- cannot call the RunPod API"
        return 1
    fi

    resp_file="$(mktemp "${TMPDIR:-/tmp}/deadman-runpod-XXXXXX")" || {
        log "could not create a temp file for the RunPod API response"
        return 1
    }
    # `if ! VAR=$(...); then` (not `VAR=$(...); rc=$?`) is deliberate: under
    # `set -e`, a plain assignment statement that fails aborts the script
    # immediately, same footgun as log()'s own comment describes for `tee`.
    # Putting it in the `if` condition is what makes the failure something
    # this function can actually handle.
    if ! http_code="$(curl -sS --max-time 20 --connect-timeout 10 \
        -o "${resp_file}" -w '%{http_code}' \
        -X DELETE \
        -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
        "https://rest.runpod.io/v1/pods/${RUNPOD_POD_ID}" 2>>"${DEADMAN_LOG_FILE}")"
    then
        log "RunPod API call failed (curl network/TLS error) -- see ${DEADMAN_LOG_FILE}"
        rm -f "${resp_file}"
        return 1
    fi
    body="$(cat "${resp_file}" 2>/dev/null || true)"
    rm -f "${resp_file}"

    if [ "${http_code}" = "204" ]; then
        log "RunPod API: pod ${RUNPOD_POD_ID} deleted (HTTP 204)"
        return 0
    fi

    log "RunPod API terminate FAILED: HTTP ${http_code}${body:+ body=${body}}"
    return 1
}

lambda_terminate() {
    # Lambda Cloud public API, confirmed directly against Lambda's own API
    # reference (https://docs.lambda.ai/api/cloud, "Terminate instances"):
    #   POST https://cloud.lambda.ai/api/v1/instance-operations/terminate
    #   Authorization: Bearer <LAMBDA_API_KEY>
    #   Content-Type: application/json
    #   Body:    {"instance_ids": ["<LAMBDA_INSTANCE_ID>"]}
    #   Success: {"data": ...}
    #   Error:   {"error": {"code": "...", "message": "..."}}
    # The docs describing this endpoint didn't have the exact success status
    # code visible (Lambda's other endpoints return 200), so this checks for
    # a 2xx status AND the absence of an "error" key in the body, rather
    # than hard-coding one specific 2xx code that might be wrong.
    if [ -z "${LAMBDA_API_KEY:-}" ] || [ -z "${LAMBDA_INSTANCE_ID:-}" ]; then
        log "LAMBDA_API_KEY / LAMBDA_INSTANCE_ID not set -- required for DEADMAN_PROVIDER=lambda"
        return 1
    fi
    if ! command -v curl >/dev/null 2>&1; then
        log "'curl' is not installed -- cannot call the Lambda Cloud API"
        return 1
    fi

    resp_file="$(mktemp "${TMPDIR:-/tmp}/deadman-lambda-XXXXXX")" || {
        log "could not create a temp file for the Lambda API response"
        return 1
    }
    request_body="{\"instance_ids\": [\"${LAMBDA_INSTANCE_ID}\"]}"
    if ! http_code="$(curl -sS --max-time 20 --connect-timeout 10 \
        -o "${resp_file}" -w '%{http_code}' \
        -X POST \
        -H "Authorization: Bearer ${LAMBDA_API_KEY}" \
        -H "Content-Type: application/json" \
        -d "${request_body}" \
        "https://cloud.lambda.ai/api/v1/instance-operations/terminate" 2>>"${DEADMAN_LOG_FILE}")"
    then
        log "Lambda API call failed (curl network/TLS error) -- see ${DEADMAN_LOG_FILE}"
        rm -f "${resp_file}"
        return 1
    fi
    body="$(cat "${resp_file}" 2>/dev/null || true)"
    rm -f "${resp_file}"

    case "${http_code}" in
        2??)
            if printf '%s' "${body}" | grep -q '"error"'; then
                log "Lambda API terminate FAILED: HTTP ${http_code} but body contains an error: ${body}"
                return 1
            fi
            log "Lambda API: instance ${LAMBDA_INSTANCE_ID} terminate requested (HTTP ${http_code})"
            return 0
            ;;
        *)
            log "Lambda API terminate FAILED: HTTP ${http_code}${body:+ body=${body}}"
            return 1
            ;;
    esac
}

vast_terminate() {
    # Best-effort skeleton -- verify against Vast.ai's *current* API docs
    # before relying on this in production.
    #
    # TODO(operator): fill in / verify:
    #   1. VAST_API_KEY     (env var, read below)
    #   2. VAST_INSTANCE_ID (env var, read below)
    #   3. The exact current endpoint + method for "destroy this instance"
    #      from Vast.ai's own API reference.
    if [ -z "${VAST_API_KEY:-}" ] || [ -z "${VAST_INSTANCE_ID:-}" ]; then
        log "VAST_API_KEY / VAST_INSTANCE_ID not set -- required for DEADMAN_PROVIDER=vast"
        return 1
    fi

    log "TODO: Vast.ai terminate API call is not filled in -- see vast_terminate() in this file."
    log "      Instance ${VAST_INSTANCE_ID} was NOT terminated via the Vast API."
    return 1
}

# --------------------------------------------------------------------------
# status: human-readable report for the runbook's "verify the dead-man's
# switch actually armed" step.

status() {
    echo "== dead-man's switch status =="
    echo "-- layer 1 (fixed boot ceiling) --"
    if command -v shutdown >/dev/null 2>&1 && [ -r /run/systemd/shutdown/scheduled ]; then
        cat /run/systemd/shutdown/scheduled
    else
        echo "no scheduled-shutdown file found at /run/systemd/shutdown/scheduled"
        echo "(also check: shutdown -c prints nothing to cancel if none is armed)"
        echo
        echo "-- layer 1 fallback mechanisms (only relevant if 'shutdown -h +N' itself failed to arm) --"
        if command -v atq >/dev/null 2>&1; then
            echo "at queue:"
            atq 2>/dev/null | sed 's/^/  /' || echo "  (could not read the at queue)"
        else
            echo "'at' not installed -- the at(1) fallback could not have armed"
        fi
        if pgrep -f 'sleep [0-9]+ && { systemctl poweroff' >/dev/null 2>&1; then
            echo "a detached background sleep+poweroff fallback process IS running"
        else
            echo "no detached background sleep+poweroff fallback process found"
        fi
        echo "if none of the above show anything armed, layer 1 is genuinely NOT armed -- see docs/runbook.md sect 6"
    fi
    echo
    echo "-- layer 2 (heartbeat watchdog) --"
    echo "heartbeat file: ${DEADMAN_HEARTBEAT_FILE}"
    age_s="$(_heartbeat_age_s)"
    if [ "${age_s}" -lt 0 ]; then
        echo "heartbeat file does not exist"
        if [ -f "${DEADMAN_BOOT_MARKER_FILE}" ]; then
            now="$(date +%s)"
            marker_ts="$(cat "${DEADMAN_BOOT_MARKER_FILE}" 2>/dev/null || echo "${now}")"
            echo "startup grace: $(( now - marker_ts ))s elapsed (grace period ${DEADMAN_STARTUP_GRACE_MIN}min)"
        else
            echo "startup grace: not started yet (will begin on the next check-heartbeat run)"
        fi
    else
        echo "heartbeat age: ${age_s}s (stale after $(( DEADMAN_HEARTBEAT_STALE_MIN * 60 ))s)"
    fi
    echo "provider: ${DEADMAN_PROVIDER}"
    echo
    echo "-- systemd timer --"
    if command -v systemctl >/dev/null 2>&1; then
        systemctl list-timers comfy-deadman.timer --no-pager 2>&1 || true
    else
        echo "systemctl not available on this host"
    fi
}

# --------------------------------------------------------------------------

case "${1:-}" in
    arm-boot-shutdown) arm_boot_shutdown ;;
    check-heartbeat)   check_heartbeat ;;
    terminate)         terminate_instance "${2:?usage: $0 terminate <reason>}" ;;
    status)            status ;;
    *)
        echo "usage: $0 {arm-boot-shutdown|check-heartbeat|terminate <reason>|status}" >&2
        exit 2
        ;;
esac
