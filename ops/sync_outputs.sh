#!/usr/bin/env bash
# ops/sync_outputs.sh
#
# Streams generated outputs off the box AS THEY ARE PRODUCED, not at the end
# of the run. Without a network volume, a RunPod pod's local disk is gone
# the instant the pod is preempted or stopped -- there is no "it's already
# rendered, just grab it later." This script is what turns "preemption
# loses the whole night" into "preemption loses at most one in-flight
# asset."
#
# Backends: rclone (default; works with R2, S3, and dozens of others via an
# rclone remote) or the `aws` CLI's `s3 sync` directly. Pick whichever is
# already configured on the box.
#
# Safe to re-run: sync operations are idempotent (both backends only
# transfer new/changed files), and a lock file prevents two loops from
# hammering the same destination concurrently.
#
# Usage:
#   ops/sync_outputs.sh            # watch loop (default)
#   ops/sync_outputs.sh --once     # single sync pass then exit (e.g. final
#                                   # flush before a planned shutdown)

set -euo pipefail

# --------------------------------------------------------------------------
# Configuration (override via environment; see ops/env.example)

SYNC_BACKEND="${SYNC_BACKEND:-rclone}"           # rclone | s3
# Defaults to COMFY_OUTPUT_DIR -- THE single authoritative output-path
# variable for this deployment (see ops/env.example's "shared: output path"
# section and ops/launch_comfyui.sh, which passes the same value to ComfyUI
# via --output-directory). Do not override SYNC_SRC and ComfyUI's output
# directory independently: defaulting them separately (this script defaulted
# to /opt/comfy-controller/output while ComfyUI defaulted to
# <COMFYUI_DIR>/output = /opt/ComfyUI/output) was exactly the mismatch that
# left outputs unsynced.
SYNC_SRC="${SYNC_SRC:-${COMFY_OUTPUT_DIR:-/opt/comfy-controller/output}}"
# rclone: "remote:bucket/prefix"   e.g. "r2:my-bucket/overnight-run"
# s3:     "s3://bucket/prefix"
SYNC_DEST="${SYNC_DEST:?set SYNC_DEST -- e.g. r2:my-bucket/prefix (rclone) or s3://my-bucket/prefix (s3)}"
SYNC_INTERVAL_S="${SYNC_INTERVAL_S:-30}"
# NOT /tmp: world-writable, so any local user could hold this lock and stall
# off-box syncing all night. Lives under the same state directory the
# controller/deadman scripts already use (/opt/comfy-controller), owned by
# the service user (`User=` in deploy/comfy-outputs-sync.service).
SYNC_LOCK_FILE="${SYNC_LOCK_FILE:-/opt/comfy-controller/.comfy_sync/sync-outputs.lock}"
SYNC_LOG_FILE="${SYNC_LOG_FILE:-/var/log/comfy-controller/sync-outputs.log}"
SYNC_RCLONE_EXTRA_ARGS="${SYNC_RCLONE_EXTRA_ARGS:-}"   # e.g. "--transfers=4"
SYNC_S3_EXTRA_ARGS="${SYNC_S3_EXTRA_ARGS:-}"

log() {
    mkdir -p "$(dirname "${SYNC_LOG_FILE}")" 2>/dev/null || true
    printf '[sync_outputs] %s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${SYNC_LOG_FILE}" >&2
}
die() { log "FATAL: $*"; exit 1; }

[ -d "${SYNC_SRC}" ] || die "SYNC_SRC does not exist: ${SYNC_SRC}"

case "${SYNC_BACKEND}" in
    rclone) command -v rclone >/dev/null 2>&1 || die "SYNC_BACKEND=rclone but rclone is not installed" ;;
    s3)     command -v aws >/dev/null 2>&1 || die "SYNC_BACKEND=s3 but the aws CLI is not installed" ;;
    *)      die "unknown SYNC_BACKEND=${SYNC_BACKEND} (want rclone or s3)" ;;
esac

# --------------------------------------------------------------------------

sync_once() {
    case "${SYNC_BACKEND}" in
        rclone)
            # shellcheck disable=SC2086 # intentional word-splitting of extra args
            rclone copy "${SYNC_SRC}" "${SYNC_DEST}" \
                --create-empty-src-dirs=false \
                ${SYNC_RCLONE_EXTRA_ARGS}
            ;;
        s3)
            # shellcheck disable=SC2086
            aws s3 sync "${SYNC_SRC}" "${SYNC_DEST}" \
                ${SYNC_S3_EXTRA_ARGS}
            ;;
    esac
}

sync_once_locked() {
    if command -v flock >/dev/null 2>&1; then
        mkdir -p "$(dirname "${SYNC_LOCK_FILE}")"
        # Fixed FD (not the bash-4.1+ `{fd}` auto-allocation form) so this
        # also works under older bash on a minimal container image.
        exec 200>"${SYNC_LOCK_FILE}"
        if ! flock -n 200; then
            log "another sync is already in flight (lock: ${SYNC_LOCK_FILE}), skipping this pass"
            return 0
        fi
        sync_once
        flock -u 200
    else
        # flock not available (e.g. minimal container image): fall back to
        # running unlocked. rclone/aws are safe to overlap; this just risks
        # doing some redundant work, never corruption.
        sync_once
    fi
}

# --------------------------------------------------------------------------

log "starting: backend=${SYNC_BACKEND} src=${SYNC_SRC} dest=${SYNC_DEST}"

if [ "${1:-}" = "--once" ]; then
    sync_once_locked
    log "single pass complete"
    exit 0
fi

trap 'log "stopping watch loop"; exit 0' TERM INT

while true; do
    sync_once_locked || log "sync pass failed (backend=${SYNC_BACKEND}); will retry next interval"
    sleep "${SYNC_INTERVAL_S}"
done
