#!/usr/bin/env bash
# ops/launch_comfyui.sh
#
# Starts ComfyUI with the deliberate risk-reduction flags for an unattended
# overnight run on a rented GPU, and refuses to hand control back until the
# #16312 zombie-worker guard (comfyui_guard_node/) has confirmed, from
# ComfyUI's own startup log, that its monkeypatch actually took.
#
# Safe to re-run: re-running just re-verifies the checkout, re-links the
# guard node, and (re)launches ComfyUI. It does not blow away a dirty
# checkout or duplicate the custom_nodes symlink.
#
# Intended callers:
#   - deploy/comfyui.service (systemd, standalone mode -- see that file's
#     header comment for when this is and isn't the right way to run it)
#   - HealthSupervisor's `launch_argv` (src/comfy_controller/supervisor.py),
#     configured by config.py, so that EVERY restart the supervisor performs
#     -- not just the very first boot -- re-verifies the guard patch took.
#     This script is written to be idempotent and side-effect-safe for
#     exactly that reason: the supervisor may invoke it many times a night.
#
# This script does not itself daemonize or background; it execs into a
# foreground wait on the ComfyUI child so a process supervisor (systemd or
# HealthSupervisor's asyncio subprocess handle) gets a faithful exit code and
# can send it signals normally.

set -euo pipefail

# --------------------------------------------------------------------------
# Pinned ComfyUI commit
#
# THIS IS DELIBERATE, not a leftover pin someone forgot to update. #16312
# (the zombie prompt_worker) and the DynamicVRAM OOM regression (~Feb 2026)
# are both open upstream with no merged fix as of this writing. Moving this
# pin means re-validating both bugs are still mitigated by
# comfyui_guard_node/ and by --disable-dynamic-vram before trusting an
# unattended run against the new commit. See README.md's "known
# limitations" section.
COMFYUI_COMMIT="${COMFYUI_COMMIT:?set COMFYUI_COMMIT to the pinned ComfyUI git SHA -- see README.md}"

# --------------------------------------------------------------------------
# Configuration (override via environment; see ops/env.example)
COMFYUI_REPO_URL="${COMFYUI_REPO_URL:-https://github.com/comfyanonymous/ComfyUI.git}"
COMFYUI_DIR="${COMFYUI_DIR:-/opt/ComfyUI}"
# 127.0.0.1, NOT 0.0.0.0: the controller (src/comfy_controller/config.py's
# comfy_base_url) only ever talks to ComfyUI at 127.0.0.1:8188, so a
# wildcard bind buys this deployment nothing functionally while exposing an
# UNAUTHENTICATED POST /prompt (arbitrary workflows on this GPU, file
# read/write via node inputs) and /history (job data disclosure) on
# whatever public IP this rented box has. See the loud opt-in check below
# if you genuinely need a wider bind (e.g. controller and ComfyUI on
# different hosts) -- and see docs/runbook.md's firewall section either way.
COMFYUI_HOST="${COMFYUI_HOST:-127.0.0.1}"
COMFYUI_ALLOW_NONLOOPBACK_HOST="${COMFYUI_ALLOW_NONLOOPBACK_HOST:-0}"
COMFYUI_PORT="${COMFYUI_PORT:-8188}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-}"                 # default resolved below
COMFYUI_LOG_DIR="${COMFYUI_LOG_DIR:-/var/log/comfy-controller}"
# THE authoritative output-path variable for this whole deployment -- see
# ops/env.example's "shared: output path" section. ops/sync_outputs.sh's
# SYNC_SRC defaults to this same value; do not override the two
# independently or they will drift apart again (that drift -- this script's
# side effectively defaulting to <COMFYUI_DIR>/output while sync_outputs.sh
# defaulted to /opt/comfy-controller/output -- was exactly the output-path
# mismatch this variable fixes). If the controller side (cli.py/config.py)
# needs to agree too, set COMFYCTL_COMFY_OUTPUT_DIR to this same path.
COMFY_OUTPUT_DIR="${COMFY_OUTPUT_DIR:-/opt/comfy-controller/output}"
GUARD_NODE_TIMEOUT_S="${GUARD_NODE_TIMEOUT_S:-90}"
GUARD_NODE_NAME="comfyui_guard_node"

# Repo root: this script lives in ops/, the guard node is a sibling dir.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
GUARD_NODE_SRC="${REPO_ROOT}/${GUARD_NODE_NAME}"

log() { printf '[launch_comfyui] %s\n' "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

# --------------------------------------------------------------------------
# 0. Refuse to bind wider than loopback without an explicit, loud opt-in.
#
# ComfyUI itself has no authentication on /prompt (accepts and runs
# arbitrary workflows -- including nodes that read/write arbitrary files) or
# /history (discloses job data). The controller only ever reaches ComfyUI
# at 127.0.0.1:8188, so binding this wider serves no purpose in the
# recommended deployment and only adds exposure on a publicly-routable
# rented GPU host. Fail fast, before doing any of the slower checkout/link
# work below, so a misconfigured host env doesn't waste time before hitting
# this.
case "${COMFYUI_HOST}" in
    127.0.0.1|localhost|::1) ;;
    *)
        if [ "${COMFYUI_ALLOW_NONLOOPBACK_HOST}" != "1" ]; then
            die "COMFYUI_HOST=${COMFYUI_HOST} is not loopback, and COMFYUI_ALLOW_NONLOOPBACK_HOST=1" \
                "was not set. ComfyUI's /prompt and /history endpoints have no auth of their own --" \
                "binding wider than 127.0.0.1 exposes this GPU box's job queue and" \
                "filesystem-adjacent node inputs to whatever network this host is reachable from." \
                "If you genuinely need that (e.g. the controller runs on a different host), set" \
                "COMFYUI_ALLOW_NONLOOPBACK_HOST=1 explicitly AND lock this down at the" \
                "firewall/security-group layer first -- see docs/runbook.md's firewall section."
        fi
        log "WARNING: COMFYUI_HOST=${COMFYUI_HOST} -- binding wider than loopback with" \
            "COMFYUI_ALLOW_NONLOOPBACK_HOST=1 explicitly set. Confirm your firewall/security-group" \
            "blocks port ${COMFYUI_PORT} from the public internet right now -- ComfyUI itself will not" \
            "stop you, or anyone else who can reach it, from submitting workflows."
        ;;
esac

# --------------------------------------------------------------------------
# 1. Checkout / pin ComfyUI

if [ ! -d "${COMFYUI_DIR}/.git" ]; then
    log "cloning ComfyUI into ${COMFYUI_DIR}"
    mkdir -p "$(dirname "${COMFYUI_DIR}")"
    git clone "${COMFYUI_REPO_URL}" "${COMFYUI_DIR}"
fi

cd "${COMFYUI_DIR}"

current_commit="$(git rev-parse HEAD)"
if [ "${current_commit}" != "${COMFYUI_COMMIT}" ]; then
    if [ -n "$(git status --porcelain)" ]; then
        die "checkout at ${COMFYUI_DIR} is dirty and not at the pinned commit" \
            "(want ${COMFYUI_COMMIT}, have ${current_commit}) -- refusing to" \
            "discard local changes. Resolve manually."
    fi
    log "checkout is at ${current_commit}, pinned commit is ${COMFYUI_COMMIT} -- fetching and checking out"
    git fetch --quiet origin "${COMFYUI_COMMIT}" || git fetch --quiet --tags origin
    git checkout --quiet --detach "${COMFYUI_COMMIT}"
fi
log "ComfyUI pinned at ${COMFYUI_COMMIT}"

# --------------------------------------------------------------------------
# 2. Link the guard node into custom_nodes/

[ -d "${GUARD_NODE_SRC}" ] || die "guard node source not found at ${GUARD_NODE_SRC}"

mkdir -p "${COMFYUI_DIR}/custom_nodes"
ln -sfn "${GUARD_NODE_SRC}" "${COMFYUI_DIR}/custom_nodes/${GUARD_NODE_NAME}"
log "linked ${GUARD_NODE_NAME} -> ${COMFYUI_DIR}/custom_nodes/${GUARD_NODE_NAME}"

# --------------------------------------------------------------------------
# 3. Resolve interpreter

if [ -z "${COMFYUI_PYTHON}" ]; then
    if [ -x "${COMFYUI_DIR}/.venv/bin/python" ]; then
        COMFYUI_PYTHON="${COMFYUI_DIR}/.venv/bin/python"
    else
        COMFYUI_PYTHON="python3"
    fi
fi
command -v "${COMFYUI_PYTHON}" >/dev/null 2>&1 || die "python interpreter not found: ${COMFYUI_PYTHON}"

# --------------------------------------------------------------------------
# 4. Build argv
#
# --cache-none: ComfyUI's DEFAULT cache backend is --cache-ram
# (RAMPressureCache). Its ram_release() -> scan_list_for_ram_usage() path is
# the documented trigger for #16312's zombie prompt_worker: an exception
# raised while scanning RAM usage propagates out of the worker thread and
# kills it silently, while /system_stats keeps answering 200. Cross-prompt
# RAM caching also buys nothing here -- this run pushes ~500 DISTINCT
# assets through, so there is no cache hit rate to protect. Trading a
# feature that doesn't help us for a feature that has a known crash path is
# not a close call. (comfyui_guard_node/ is defense in depth for the same
# bug in case this flag's behaviour ever changes upstream -- it is not a
# substitute for this flag.)
#
# --disable-dynamic-vram: DynamicVRAM became ComfyUI's default VRAM
# management strategy around Feb 2026 and has an open OOM regression as of
# this writing. An OOM mid-batch is worse for us than the more
# conservative, previously-default VRAM behaviour this flag restores.
#
# --output-directory: pinned explicitly to COMFY_OUTPUT_DIR rather than
# trusting ComfyUI's own default (<COMFYUI_DIR>/output). This is the fix for
# the output-path mismatch the audit found: ops/sync_outputs.sh's SYNC_SRC
# defaults to the SAME COMFY_OUTPUT_DIR value, so the two can no longer
# silently point at different directories.
extra_args=("$@")
cmd=(
    "${COMFYUI_PYTHON}" main.py
    --listen "${COMFYUI_HOST}"
    --port "${COMFYUI_PORT}"
    --cache-none
    --disable-dynamic-vram
    --output-directory "${COMFY_OUTPUT_DIR}"
)
cmd+=("${extra_args[@]}")

mkdir -p "${COMFY_OUTPUT_DIR}"
mkdir -p "${COMFYUI_LOG_DIR}"
log_file="${COMFYUI_LOG_DIR}/comfyui-$(date -u +%Y%m%dT%H%M%SZ).log"
ln -sfn "${log_file}" "${COMFYUI_LOG_DIR}/comfyui-latest.log"

log "launching: ${cmd[*]}"
log "log file: ${log_file}"

# --------------------------------------------------------------------------
# 5. Launch, forwarding signals to the child so systemd/the supervisor can
#    stop this cleanly.

"${cmd[@]}" >"${log_file}" 2>&1 &
comfy_pid=$!

# shellcheck disable=SC2329 # invoked indirectly via the traps just below
forward_signal() {
    sig="$1"
    if kill -0 "${comfy_pid}" 2>/dev/null; then
        log "forwarding ${sig} to ComfyUI (pid ${comfy_pid})"
        kill "-${sig}" "${comfy_pid}" 2>/dev/null || true
    fi
}
trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT

# --------------------------------------------------------------------------
# 6. Verify the guard patch took, from the log -- not assumed.
#
# A silent regression here (upstream rename, guard node failing to import,
# a shadowing re-import) puts us straight back into #16312's zombie
# behaviour with NO error anywhere else, so this has to be loud and this
# has to block. See comfyui_guard_node/__init__.py's own docstring for the
# "PATCH CONFIRMED" / "NOT installed" / "verification FAILED" log lines
# this greps for.
CONFIRM_PATTERN='PATCH CONFIRMED'
FAILURE_PATTERN='NOT installed|verification FAILED'

deadline=$(( $(date +%s) + GUARD_NODE_TIMEOUT_S ))
guard_confirmed=0
while [ "$(date +%s)" -lt "${deadline}" ]; do
    if ! kill -0 "${comfy_pid}" 2>/dev/null; then
        die "ComfyUI exited before the guard node could confirm its patch -- see ${log_file}"
    fi
    if grep -qE "${CONFIRM_PATTERN}" "${log_file}" 2>/dev/null; then
        guard_confirmed=1
        break
    fi
    if grep -qE "${FAILURE_PATTERN}" "${log_file}" 2>/dev/null; then
        break
    fi
    sleep 1
done

if [ "${guard_confirmed}" -ne 1 ]; then
    log "FATAL: comfyui_guard_node did not confirm its #16312 patch within ${GUARD_NODE_TIMEOUT_S}s"
    log "       (see ${log_file} for the guard's own log lines). Refusing to run unguarded --"
    log "       this would silently reproduce the zombie-worker bug with no error anywhere."
    kill "${comfy_pid}" 2>/dev/null || true
    wait "${comfy_pid}" 2>/dev/null || true
    exit 1
fi

log "guard node confirmed #16312 patch active -- ComfyUI running as pid ${comfy_pid}"

# Stay attached so our own exit code mirrors ComfyUI's, and so signals sent
# to this script's process group reach the child via the traps above.
set +e
wait "${comfy_pid}"
exit_code=$?
set -e
log "ComfyUI (pid ${comfy_pid}) exited with code ${exit_code}"
exit "${exit_code}"
