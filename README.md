# comfy-controller

A resilient overnight batch controller for ComfyUI: pushes ~500 distinct
assets through ComfyUI unattended, on a rented GPU pod, and is designed
around the fact that both the generation engine and the rented hardware can
fail silently.

## The problem

Two failure modes drive almost everything in this repo, and both share the
same dangerous property: **neither raises an error anywhere.**

1. **[ComfyUI issue #16312](https://github.com/comfyanonymous/ComfyUI/issues/16312)**
   — a framework-level exception can kill ComfyUI's background
   `prompt_worker` thread while the HTTP server keeps answering 200 and
   `POST /prompt` keeps accepting jobs into a queue that will never drain
   again. `/system_stats` returning 200 proves nothing. There is no crash,
   no traceback, nothing to grep for — just a batch that silently stops
   producing output while looking, from the outside, exactly like it's
   still working.
2. **Quality failures raise no error at all.** A generated image can be
   anatomically wrong, off-brief, or full of artifacts, and ComfyUI will
   report success — that's not a bug in ComfyUI, it's just not ComfyUI's
   job to know what "correct" means for your brief. Nothing catches this
   except an actual quality check.

On top of both of those: this runs on **rented GPU capacity** (RunPod
Secure Cloud / Lambda on-demand), where an idle instance keeps billing —
none of RunPod, Vast, or Lambda auto-stops an idle *Pod* (RunPod's
idle-timeout auto-stop is a Serverless-only feature, it does not apply
here) — and pod-local disk is gone the instant a pod is preempted or
stopped.

## Architecture

```
                         ┌─────────────────────────────┐
                         │   deploy/comfy-controller    │
                         │   .service (systemd; OS      │
                         │   supervises the controller)  │
                         └───────────────┬───────────────┘
                                         │ owns a direct
                                         │ subprocess handle
                                         ▼
┌────────────────────┐     spawns     ┌─────────────────────────────┐
│  ops/launch_comfyui │◄───────────────┤   HealthSupervisor          │
│  .sh                │                │   (src/.../supervisor.py)   │
│  - pins commit      │   restarts on  │   - WS progress/executing   │
│  - links guard node │   ZOMBIE/DEAD  │   - GPU/CPU idle check      │
│  - verifies patch   │   verdicts,    │   - front=True canary       │
│    took, from log   │   with         │   - hysteresis + backoff +  │
│  - --cache-none     │   hysteresis   │     intensity ceiling       │
│  - --disable-       │                └───────────────┬─────────────┘
│    dynamic-vram     │                                │ drives assets
└─────────┬───────────┘                                ▼
          │ runs                        ┌─────────────────────────────┐
          ▼                             │   SQLite store (source of    │
┌────────────────────┐   custom_nodes/  │   truth: assets, run_log)    │
│      ComfyUI        │◄────────────────┤   claim_next / transition /  │
│  (+ comfyui_guard_  │   symlink        │   needing_reconciliation     │
│   node/ patches      │                 └───────────────┬─────────────┘
│   PromptExecutor)    │                                 │ terminal states
└─────────┬────────────┘                                 ▼
          │ output files              ┌───────────────────────────────┐
          ▼                            │  Judge (Claude, config/rubric │
┌────────────────────┐                 │  .yaml) -- binary checks,     │
│ ops/sync_outputs.sh │                 │  never fails open: infra      │
│ (rclone/s3 watch     │                │  failure -> JUDGE_UNAVAILABLE,│
│  loop -- streams      │                │  never auto-pass             │
│  output off-box as     │               └───────────────────────────────┘
│  produced)             │
└────────────────────────┘

  Independent of all of the above:
┌──────────────────────────────────────────────────────────────────────┐
│  Dead-man's switch (ops/deadman.sh + deploy/comfy-deadman*)           │
│  Layer 1: shutdown -h +N, armed at boot, before anything else starts, │
│           with fallbacks (at / detached sleep+poweroff) if that fails │
│           to arm, and an OnFailure= alert unit if even those do.      │
│  Layer 2: systemd timer checks a controller heartbeat file; stale ->  │
│           terminate via provider API (pluggable), else local shutdown │
│           + an escalation ladder if THAT fails too.                   │
│  Layer 2.5: comfy-controller.service's ExecStopPost calls the SAME    │
│           termination path proactively on a clean batch finish, so    │
│           a successful run doesn't have to wait for the heartbeat to  │
│           go stale (or for layer 1) before the box stops billing.     │
│  Layer 3: in-process budget counters -- defense in depth ONLY.        │
└──────────────────────────────────────────────────────────────────────┘
```

## Quickstart

```bash
# 1. Install
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# 2. Configure (copy and fill in -- see "Config / env vars" below). One
#    shared template, copied per-unit since each systemd unit's
#    EnvironmentFile= only needs the variables relevant to it, but there's
#    no harm in each copy carrying all of them.
mkdir -p /etc/comfy-controller
for f in comfyui controller deadman sync; do
    cp ops/env.example "/etc/comfy-controller/${f}.env"
done
$EDITOR /etc/comfy-controller/*.env

# 3. Arm the dead-man's switch FIRST, before anything else runs
sudo systemctl enable --now comfy-deadman-boot.service
sudo systemctl enable --now comfy-deadman.timer
ops/deadman.sh status   # confirm both layers actually armed -- see docs/runbook.md §6
# (comfy-deadman-alert.service does not need to be separately enabled --
# it's only ever invoked as comfy-deadman-boot.service's OnFailure= target)

# 4. Start outputs streaming off-box
sudo systemctl enable --now comfy-outputs-sync.service

# 5. Start ComfyUI + the controller
sudo systemctl enable --now comfy-controller.service
journalctl -u comfy-controller.service -f
```

A broken run at 2am or 8am: go straight to **[docs/runbook.md](docs/runbook.md)**
— it has exact commands for telling a zombie from a slow render, reading
the SQLite state, resuming, approving parked assets, and verifying the
dead-man's switch.

## Config / env vars

Every environment variable the `ops/*.sh` scripts and `deploy/*.service`
units read is documented in one place: **[ops/env.example](ops/env.example)**.
Highlights:

| Variable | Purpose |
|---|---|
| `COMFYUI_COMMIT` | Pinned ComfyUI git SHA. No default — the launch script refuses to start without it. |
| `COMFY_OUTPUT_DIR` | THE single authoritative output path — passed to ComfyUI via `--output-directory` and defaulted-into by `sync_outputs.sh`'s `SYNC_SRC`. Set `COMFYCTL_COMFY_OUTPUT_DIR` to the same value for the controller side. |
| `COMFYUI_HOST` / `COMFYUI_ALLOW_NONLOOPBACK_HOST` | ComfyUI's bind address (default `127.0.0.1` — it has no auth of its own) and the loud, explicit opt-in required to bind wider. |
| `DEADMAN_HARD_CEILING_MIN` | Layer 1's fixed shutdown ceiling from boot (default 660min / 11h). |
| `DEADMAN_HEARTBEAT_STALE_MIN` | Layer 2's staleness threshold (default 20min). |
| `DEADMAN_PROVIDER` | `runpod` \| `lambda` \| `vast` \| `none` — which cloud API layer 2 calls to terminate the instance. |
| `SYNC_BACKEND` / `SYNC_DEST` | `rclone` or `s3`, and the remote destination outputs stream to. |
| `SYNC_LOCK_FILE` | Off-box sync's lock file — under the service user's own state dir, not `/tmp`. |

The rubric the judge checks against lives in `config/rubric.yaml`, not in
this doc — it's the single source of truth for what "quality failure" means
for this batch.

## Known limitations

Being honest about these matters more than sounding finished:

- **Neither upstream bug this repo works around has a merged fix.**
  [#16312](https://github.com/comfyanonymous/ComfyUI/issues/16312) (the
  zombie `prompt_worker`) and the DynamicVRAM OOM regression (~Feb 2026)
  are both open as of this writing. `comfyui_guard_node/` and
  `--disable-dynamic-vram` are mitigations, not fixes, and the ComfyUI
  commit is pinned (`COMFYUI_COMMIT`) specifically so moving forward is a
  deliberate, re-validated decision rather than an accidental one. Re-check
  both issues before ever bumping the pin.
- **The dead-man's switch's provider-termination calls are unfilled
  skeletons.** `ops/deadman.sh`'s `runpod_terminate` / `lambda_terminate` /
  `vast_terminate` read the right environment variables and log clearly
  when they're not configured, but the actual API calls are left as
  `TODO`s rather than guessed endpoints — a wrong guess against a
  bill-generating API is worse than an honest gap. Until an operator fills
  these in, termination falls through to a local `shutdown -h now` (with an
  escalation ladder — `systemctl poweroff --force --force`, then
  `echo o > /proc/sysrq-trigger` — if even that fails or is absent, e.g. on
  a systemd-less container), which is **not guaranteed to stop billing on
  every provider** (it halts the OS; it does not necessarily
  release/deallocate the rented resource). Confirming the instance is
  actually gone is still a manual provider-console check until these are
  filled in — see docs/runbook.md sect 7.
- **`comfy-controller.service`'s `ExecStart` is a placeholder.**
  `src/comfy_controller/cli.py` was being written concurrently with this
  ops layer and did not exist yet when `deploy/comfy-controller.service`
  was written. Confirm the real entrypoint/subcommand before relying on
  that unit. (Its `ExecStopPost`, by contrast, is fully wired — see below.)
- **`deploy/comfyui.service` is standalone-mode only.** In the primary
  deployment mode, `HealthSupervisor` spawns and owns the ComfyUI process
  directly (it needs that process handle for its canary/restart logic) —
  systemd managing ComfyUI at the same time would fight the supervisor's
  own restart discipline. See that unit file's header comment for when it
  does and doesn't apply.
- **The heartbeat contract is a contract, not (yet) a wired connection.**
  Layer 2 of the dead-man's switch expects the controller to touch
  `DEADMAN_HEARTBEAT_FILE` once per control-loop iteration, more often
  than `DEADMAN_HEARTBEAT_STALE_MIN` (`src/comfy_controller/runner.py`,
  also written concurrently with this ops layer; see `ops/deadman.sh`'s own
  comment above `DEADMAN_HEARTBEAT_FILE` for the exact contract, and
  docs/runbook.md's "Heartbeat contract" section). Until that's wired up,
  the heartbeat file never gets created and layer 2 will always see it as
  stale — by design this fails *safe* (it terminates rather than idling
  forever), but it means layer 2 fires on every run that reaches it, not
  just on a genuine hang. **This matters less than it used to**:
  `comfy-controller.service`'s `ExecStopPost` (`ops/on_controller_stop.sh`)
  now terminates the instance proactively the instant the controller exits
  cleanly (batch actually finished), independent of the heartbeat — so the
  still-unwired heartbeat mainly affects detecting a *hang*, not a normal
  successful run.
- **Layer 1's `OnFailure=` alert has no pager/email behind it.**
  `deploy/comfy-deadman-alert.service` fires `wall(1)` and a `daemon.crit`
  journal entry if boot-time arming (including both of its non-logind
  fallbacks) fails outright, but that only reaches someone who is watching
  an attached terminal or shipping the journal somewhere. It is not a
  substitute for actually checking `ops/deadman.sh status` before walking
  away from the box — see docs/runbook.md sect 6.
- **No shellcheck/CI wiring for the `ops/` scripts.** They were verified
  manually with `shellcheck` (0 warnings as of this writing, across every
  script in `ops/`) and by hand-running each script's logic end-to-end
  during development — including reproducing the auditor's
  unwritable-log-file scenario against `ops/deadman.sh` to confirm
  `shutdown` is now actually reached — but nothing in this repo's test
  suite (`tests/`) currently exercises them automatically on every change.
