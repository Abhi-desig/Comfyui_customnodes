# Runbook

For the human who gets paged at 2am, or who opens their laptop at 8am to a
run that didn't finish cleanly. Every section below assumes you have shell
access to the box and gives exact commands, not just descriptions.

Paths below use placeholders:

- `$REPO` — this repo's checkout on the box, e.g. `/opt/comfy-controller`
- `$DB` — the controller's SQLite state file (path comes from the runner's
  own config; if you don't know it, `find $REPO -name '*.db'` or check
  `comfy-controller.env`'s `COMFY_DB_PATH`-style variable once `config.py`
  defines one — this doc doesn't guess a value that doesn't exist yet)

---

## 1. Is it a zombie, or is it just a slow render?

This is the single most important diagnosis in this whole system, because
**`/system_stats` returning HTTP 200 proves nothing** — that's the entire
content of [issue #16312](https://github.com/comfyanonymous/ComfyUI/issues/16312).
A framework-level exception can kill ComfyUI's background `prompt_worker`
thread while the HTTP server keeps answering 200 and `POST /prompt` keeps
accepting jobs into a queue that will never drain again. There is no crash,
no traceback in the ComfyUI log, nothing to grep for. A curl against
`/system_stats` succeeding tells you the HTTP server is alive; it tells you
*nothing* about whether the worker thread that actually runs prompts is
still there.

What actually distinguishes "dead" from "legitimately busy":

| Signal | Zombie (#16312) | Slow render |
|---|---|---|
| `curl -s $COMFYUI_URL/system_stats` | 200 OK (meaningless either way) | 200 OK |
| `curl -s $COMFYUI_URL/queue` | jobs stuck `pending`, never move to `running` | one job `running`, others `pending` |
| WS `progress`/`executing` events | none, ever | arriving periodically |
| GPU utilization (`nvidia-smi`) | idle (near 0%) | pegged |
| A `front=True` canary prompt | submitted, never reaches `executed` | n/a (don't need one — GPU is clearly busy) |

The controller already automates exactly this triage
(`src/comfy_controller/supervisor.py::HealthSupervisor`): it watches WS
`progress`/`executing` traffic, cross-checks GPU/CPU utilization when that
traffic goes quiet, and only when both look wrong does it fire a
`front=True` canary prompt — the one signal that can't be fooled, because a
canary that never reaches `executed` means the queue genuinely isn't being
consumed. If the supervisor decided ZOMBIE and restarted, you'll find a
marker recording why:

```bash
ls -lt $REPO/.comfy_supervisor/restart-*.json | head -5
cat $REPO/.comfy_supervisor/restart-<timestamp>.json   # {"reason": "...", "at": ...}
```

To check by hand right now, without waiting for the next supervisor tick:

```bash
# 1. HTTP liveness only -- do not stop here.
curl -s $COMFYUI_URL/system_stats | head -c 300; echo

# 2. Queue depth -- ComfyUI returns arrays of tuples, not dicts.
curl -s $COMFYUI_URL/queue

# 3. GPU utilization
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits

# 4. The real test: does a trivial, queue-jumping prompt ever complete?
curl -s -X POST $COMFYUI_URL/prompt -H 'Content-Type: application/json' -d '{
  "prompt": {"1": {"class_type": "_manual_canary", "inputs": {}}},
  "front": true,
  "client_id": "manual-canary"
}'
# then watch /history/<prompt_id> -- if it never appears, the worker is dead.
curl -s $COMFYUI_URL/history/<prompt_id>
```

If GPU utilization is pegged and `/queue` shows one job genuinely running,
**do not restart** — that's a slow render, and restarting mid-job is pure
loss. If GPU is idle, WS has been silent well past the job's expected
duration, and a canary never resolves: it's a zombie. Either let the
supervisor's own hysteresis handle it (never acts on one bad sample) or
force a restart per §3 below.

## 2. Reading the SQLite state

The database is the sole source of truth (see
`src/comfy_controller/adapters/sqlite_store.py`) — never infer state from
logs.

```bash
# Overall counts by state
sqlite3 -header -column "$DB" "SELECT state, COUNT(*) FROM assets GROUP BY state;"

# Rows stuck mid-flight (interrupted by a crash/restart, not yet resolved)
sqlite3 -header -column "$DB" \
  "SELECT id, state, attempt, comfy_prompt_id, crash_count FROM assets WHERE state IN ('claimed','submitted');"

# Anything that has crashed more than once on the same asset (design intent:
# skip after 2 crashes on one asset -- if you see crash_count >= 2 still
# retrying, that's a bug, not a config question)
sqlite3 -header -column "$DB" "SELECT id, state, crash_count FROM assets WHERE crash_count >= 2;"

# The failure-rate table a morning reviewer actually wants: by workflow
sqlite3 -header -column "$DB" "
  SELECT json_extract(asset_json, '$.workflow') AS workflow,
         COUNT(*) AS attempts,
         SUM(CASE WHEN state = 'saved' THEN 1 ELSE 0 END) AS successes
  FROM assets GROUP BY workflow;"

# Full history of one asset's attempts, in order
sqlite3 -header -column "$DB" \
  "SELECT attempt, ts, state, prompt_id, error_type, verdict_summary FROM run_log WHERE asset_id='<id>' ORDER BY log_id;"

# Assets waiting on a timed retry, and when
sqlite3 -header -column "$DB" \
  "SELECT id, attempt, datetime(retry_after,'unixepoch') AS retry_at FROM assets WHERE state='retry_wait';"
```

For a browsable UI instead of hand-written SQL (useful when you're
scanning, not querying something specific):

```bash
pip install datasette   # once, if not already on the box
datasette "$DB" --open  # or --host 0.0.0.0 --port 8001 if you're on a remote box
                         # without a browser -- then SSH-tunnel that port
```

`datasette` opens a browsable, filterable, sortable view of every table
(`assets` and `run_log`) with zero setup beyond pointing it at the file.
It's read-only against the file by default, which is exactly what you want
while the controller may still be writing to it.

## 3. How to resume a run

The store is designed so a `kill -9` at any instant resumes safely: on
startup the runner calls `needing_reconciliation()`, which hands back every
row left in `claimed`/`submitted` from before, without trusting the DB
alone — it's expected to re-verify each one against ComfyUI's `/history`
(via the job's deterministic `filename_prefix`) before deciding what
actually happened. In practice, resuming is:

```bash
# 1. Confirm ComfyUI is actually healthy first (§1) -- don't resume the
#    controller against a zombie.

# 2. Restart the controller. It re-reads the DB, reconciles anything left
#    mid-flight, and continues from wherever it left off -- no separate
#    "resume" flag needed, resuming IS just starting it again against the
#    same DB.
systemctl restart comfy-controller.service
# or, if not running under systemd yet:
#   comfyctl run --db "$DB" --manifest <manifest.json>   # confirm the real
#   flags against `comfyctl --help` once cli.py exists -- this is a
#   placeholder, not a verified interface.

# 3. Watch it pick back up
journalctl -u comfy-controller.service -f
```

If the supervisor's restart-intensity ceiling was hit (see next section),
resuming the controller process is not enough by itself — clear the hold
first or it will refuse to restart ComfyUI again.

### Restart-intensity ceiling ("held")

`HealthSupervisor` caps how many restarts it will attempt in a rolling
window (`intensity_max_restarts` within `intensity_window` seconds,
defaults 5 / 600s) and then hands the problem to a human rather than
restart-looping forever — restart loops make VRAM fragmentation *worse*,
not better. If you see repeated `restart_intensity_ceiling_hit` in the
controller's log, something is genuinely broken (not just flaky), and the
supervisor is intentionally not going to keep trying on its own. Diagnose
the actual cause (§1, and the ComfyUI log at
`$COMFYUI_LOG_DIR/comfyui-latest.log`) before clearing the hold — clearing
it without fixing the underlying problem just spends the next five
restarts the same way.

## 4. Approving parked assets

`PARKED_APPROVAL` means the judge ran and rendered a verdict but the
asset's rubric result needs a human sign-off before it's treated as final
(see `config/rubric.yaml` and `src/comfy_controller/models.py::QCVerdict`).
To review and act on one:

```bash
# See what's waiting
sqlite3 -header -column "$DB" "SELECT id, job_key, output_paths_json FROM assets WHERE state='parked_approval';"

# See why the judge parked it -- which checks failed and its evidence text
sqlite3 -header -column "$DB" "SELECT last_verdict_json FROM assets WHERE id='<id>';" | python3 -m json.tool

# Look at the actual output before deciding
open "<output_path_from_output_paths_json>"   # or scp it down / view via the sync destination
```

Once `cli.py` lands, the intended interface is an approve/reject subcommand
that writes the transition through the same `StorePort.transition()` path
everything else uses (never hand-edit `state` directly in SQL — that
bypasses the `run_log` audit trail and the crash-count bookkeeping). Until
that subcommand exists, treat approval as informational only: the correct
manual escape hatch is to give a human-reviewed disposition to whoever owns
`cli.py`'s approve flow, not to `UPDATE assets SET state=...` by hand.

## 5. `JUDGE_UNAVAILABLE` vs. a rubric failure

These look similar in a quick glance at `state` but mean opposite things,
and conflating them is exactly the mistake a morning reviewer must not
make (see `src/comfy_controller/reporting.py`'s comment on this):

- **`SKIPPED_REVIEW` / a failing `QCVerdict`** — the judge *ran successfully*
  and the image genuinely failed one or more `required: true` checks in
  `config/rubric.yaml` (e.g. `anatomy_correct`, `matches_brief`). This is a
  generation-quality problem. **Quality failures raise no error at all** —
  this is normal, expected output of a working system, not a fault.
- **`JUDGE_UNAVAILABLE`** — the judge *infrastructure* failed: rate limits
  exhausted after retry, a 529 overload, a refusal, or schema-valid garbage
  that didn't parse into a real verdict
  (`src/comfy_controller/ports.py::JudgeUnavailable`). The system
  deliberately never fails open into auto-approval here — an asset with no
  real verdict is parked as `JUDGE_UNAVAILABLE`, not silently treated as
  passing.

```bash
sqlite3 -header -column "$DB" "SELECT id, error_type, verdict_summary FROM run_log WHERE state='judge_unavailable' ORDER BY log_id DESC LIMIT 20;"
```

If you see a cluster of `JUDGE_UNAVAILABLE` rows, check the judge
provider's status page and your API key/quota before assuming anything is
wrong with the generated images themselves — those assets haven't actually
been judged yet.

## 6. Verifying the dead-man's switch actually armed

Don't assume it's armed — check both layers explicitly before walking away
from the box for the night.

```bash
# One-shot status report covering both layers (also reports layer 1's
# fallback mechanisms -- the 'at' queue and any detached background
# sleep+poweroff process -- which only matter if 'shutdown -h +N' itself
# failed to arm; see below)
$REPO/ops/deadman.sh status

# Layer 1 (fixed boot-ceiling shutdown) -- confirm systemd actually has one scheduled
cat /run/systemd/shutdown/scheduled 2>/dev/null || echo "NOT ARMED via shutdown(8) -- see below"
# If that file doesn't exist, `shutdown -h +N` itself didn't arm -- but that
# does NOT necessarily mean layer 1 is unarmed: arm_boot_shutdown() falls
# back to an 'at' job, then a fully detached background sleep+poweroff,
# neither of which need systemd-logind/D-Bus. Check those before assuming
# nothing is armed:
atq 2>/dev/null                                  # an 'at' job fallback would show here
pgrep -fa 'sleep [0-9]+ && { systemctl poweroff'  # the background-sleep fallback would show here
# If NONE of the three show anything, layer 1 genuinely failed to arm --
# check why and re-run the boot unit:
systemctl status comfy-deadman-boot.service
journalctl -u comfy-deadman-boot.service -b --no-pager
systemctl restart comfy-deadman-boot.service   # re-arms it manually if needed
# Also check whether its OnFailure= alert fired (wall(1) message / a
# daemon.crit journal line from comfy-deadman-alert.service) -- if it did
# and you didn't see it live, that's your signal something needs fixing:
journalctl -u comfy-deadman-alert.service -b --no-pager

# Layer 2 (heartbeat watchdog) -- confirm the timer is actually scheduled...
systemctl list-timers comfy-deadman.timer
# ...and that the heartbeat file is fresh (should be seconds/minutes old,
# not hours -- if the controller hasn't finished starting yet, or isn't
# running at all, this will look missing/stale, which the watchdog treats
# as fail-safe, not a bug -- see "Heartbeat contract" just below):
stat $DEADMAN_HEARTBEAT_FILE
```

If layer 1 shows nothing scheduled **and** nothing in the `at` queue **and**
no background fallback process, **stop and fix that before leaving the box
unattended overnight** — it's the one layer that's supposed to hold even
when everything else fails, so it not being armed is the single
highest-severity thing this runbook covers.

Remember the ordering of guarantees: layer 3 (in-process budget counters
inside the controller) is defense in depth *only*. It cannot substitute for
layers 1/2 — a dead controller counts nothing.

### Heartbeat contract (implemented in `runner.py`)

This is the precise spec layer 2 needs on the controller side, and what
`BatchRunner._maybe_touch_heartbeat` (`src/comfy_controller/runner.py`)
implements:

- **Path**: `$DEADMAN_HEARTBEAT_FILE` / `RunnerConfig.heartbeat_file`
  (`AppConfig.heartbeat_file`, overridable via `COMFYCTL_HEARTBEAT_FILE`),
  default `/opt/comfy-controller/.comfy_supervisor/heartbeat` (shares
  `HealthSupervisor`'s existing `marker_dir` so there's one state directory
  to know about, not two).
- **What it does**: touches the file (updates its mtime, atomically —
  temp-file + rename, so a reader never sees a half-written file) from
  every worker loop's own iteration, plus a couple of extra checkpoints
  right after a ComfyUI submit/poll round-trip resolves mid-asset. This is
  throttled to at most once per `RunnerConfig.heartbeat_interval_s` (default
  30s, comfortably more often than `DEADMAN_HEARTBEAT_STALE_MIN`'s default
  20min) — but deliberately NOT implemented as a separate timer task: it has
  to be driven by the workers actually making progress, or a wedged batch
  (deadlocked on something that doesn't block the rest of the event loop)
  would keep looking healthy forever. See that method's own comment for the
  full reasoning.
- **Content**: irrelevant to the contract. Only mtime is read
  (`ops/deadman.sh::_heartbeat_age_s()`). The controller writes a small JSON
  blob (`{"ts": ..., "iteration": N}`), which is a nice-to-have for a human
  running `stat`/`cat` by hand, never required.
- **Directory/permissions**: the controller creates the parent directory
  (`mkdir -p`) on its first write rather than assuming it exists; make sure
  it's writable by whatever user `comfy-controller.service` runs as
  (`User=` in that unit).
- **Startup grace period**: before the controller's first write (still
  starting up — its own boot, or ComfyUI's model load), the file doesn't
  exist yet. `ops/deadman.sh` tolerates that for `DEADMAN_STARTUP_GRACE_MIN`
  (default 15min) rather than terminating a box that just hasn't finished
  starting — see that script's `check_heartbeat()`.
- **On a clean finish**: the heartbeat simply stops being touched once
  `BatchRunner.run()` returns (every worker loop has exited) — it is
  intentionally NOT the mechanism that tears the box down for a
  successfully-finished run. `comfy-controller.service`'s `ExecStopPost`
  (`ops/on_controller_stop.sh`) does that immediately on a clean exit (see
  §7); the heartbeat's staleness timer exists for the wedged-but-still-
  running case, not the finished-and-idle case.

## 7. Confirming the instance actually terminated

"The script ran" and "the instance is gone" are different facts — verify
the second one, not just the first.

There are now **two** paths that can trigger termination, and both funnel
through the same `terminate_instance()` in `ops/deadman.sh`:

1. **Clean finish**: `comfy-controller.service` exits 0 (batch done) →
   its `ExecStopPost` (`ops/on_controller_stop.sh`) flushes outputs
   (`sync_outputs.sh --once`) and runs `ops/deadman.sh terminate "..."`
   immediately — it does not wait for the heartbeat to go stale.
2. **Stale heartbeat / never armed**: layer 2's timer-driven
   `ops/deadman.sh check-heartbeat`, as before.

```bash
# From ops/deadman.sh's own log, on the box (if you can still reach it):
tail -50 $DEADMAN_LOG_FILE

# Did the clean-finish path (path 1 above) actually run, and did it succeed?
systemctl status comfy-controller.service
# `Active: inactive (dead)` with no failure is the good case. If you see
# `Active: failed`, that specifically means ExecStopPost's termination
# attempt exhausted EVERY local escalation step (shutdown -h now, then
# systemctl poweroff --force --force, then /proc/sysrq-trigger) without
# powering the box off -- go straight to the provider console, don't assume
# a retry will fix it:
journalctl -u comfy-controller.service -b --no-pager | tail -50

# From the provider's own console/API (the authoritative check -- do this
# even if the in-box log looks clean, since a local shutdown does not
# always guarantee the provider stopped billing -- see ops/deadman.sh's
# terminate_instance() comment):
#   RunPod:  check the pod's status in the RunPod console, or its pod-status
#            API endpoint, for "terminated"/"exited" -- not just "the SSH
#            connection dropped".
#   Lambda:  check the instance's status in the Lambda Cloud console/API.
#   Vast.ai: check the instance's status in the Vast.ai console/API.
# None of these API calls are wired up yet in this repo (see the TODOs in
# ops/deadman.sh) -- until an operator fills them in, the guaranteed-local
# path is `shutdown -h now`, escalating to `systemctl poweroff --force
# --force` and then `echo o > /proc/sysrq-trigger` if that fails or is
# absent (common on a systemd-less rented container). Confirming real
# termination/billing-stop is a manual provider-console check until the
# provider TODOs are filled in.
```

If you configured `DEADMAN_PROVIDER` and filled in the TODOs yourself,
also confirm the API call actually returned success in
`$DEADMAN_LOG_FILE` rather than falling through to the local-shutdown
fallback (the log makes this explicit either way, and now also logs
`provider_terminated=0|1` on every termination attempt).

You can also trigger the clean-finish path manually, e.g. to test it
end-to-end before a real overnight run:

```bash
sudo -u comfy /opt/comfy-controller/ops/deadman.sh terminate "manual test"
```

## 8. Firewall / network exposure

ComfyUI has no authentication of its own: `POST /prompt` runs whatever
workflow it's handed (including nodes that read/write arbitrary files on
this box), and `GET /history` discloses job data to anyone who can reach
it. `ops/launch_comfyui.sh` now defaults `COMFYUI_HOST=127.0.0.1` for
exactly this reason — the controller only ever needs `127.0.0.1:8188`
(`src/comfy_controller/config.py`'s `comfy_base_url`), so a wider bind adds
exposure with no functional benefit in the recommended deployment.

Check this before an unattended run, not after:

```bash
# Confirm ComfyUI is actually on loopback only:
ss -tlnp | grep :8188        # want 127.0.0.1:8188, NOT 0.0.0.0:8188 or [::]:8188
# or, if ss isn't available:
netstat -tlnp 2>/dev/null | grep :8188
```

If you deliberately set `COMFYUI_ALLOW_NONLOOPBACK_HOST=1` (e.g. the
controller runs on a different host than ComfyUI), the loopback check above
won't apply — in that case, lock port `8188` down at the
firewall/security-group layer instead:

- **RunPod**: restrict exposed TCP ports for the pod to only what you
  actually need; don't map 8188 publicly unless you have a specific reason
  to reach it directly (you shouldn't — the controller is the only
  intended client).
- **Lambda / Vast.ai / a generic cloud host**: use that provider's
  security-group / firewall rules to block inbound 8188 from
  `0.0.0.0/0`, allowing only the specific host(s) that need it (ideally
  none, if the controller and ComfyUI are on the same box, which is the
  recommended deployment).
- Either way: `iptables`/`nft`/`ufw` on the box itself is a second layer,
  not a substitute for the provider-level firewall — a rented box's
  provider-assigned public IP is reachable regardless of in-guest rules
  unless the provider's own network layer also blocks it.

This is the same category of gap as the dead-man's switch's provider APIs:
an honest, checkable state rather than a guessed-at "should be fine."
