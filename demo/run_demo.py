#!/usr/bin/env python
"""End-to-end demo: no GPU, no API key, no ComfyUI install.

Runs the REAL controller against FakeComfy and a fake judge, and injects the
#16312 zombie partway through so you can watch the watchdog notice, restart,
and finish the batch with nothing lost.

    .venv/bin/python demo/run_demo.py            # zombie injected (default)
    .venv/bin/python demo/run_demo.py --clean    # no faults, happy path
    .venv/bin/python demo/run_demo.py --fault oom
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comfy_controller.adapters.comfy_http import ComfyHTTP  # noqa: E402
from comfy_controller.adapters.console_notifier import ConsoleNotifier  # noqa: E402
from comfy_controller.adapters.sqlite_store import SQLiteStore  # noqa: E402
from comfy_controller.core.policy import ErrorClassifier, RetryPolicy  # noqa: E402
from comfy_controller.models import Asset, AssetState  # noqa: E402
from comfy_controller.runner import BatchRunner, RunnerConfig  # noqa: E402
from comfy_controller.supervisor import HealthSupervisor  # noqa: E402
from comfy_controller.testing.fake_comfy import FakeComfy, FakeComfyState, Fault  # noqa: E402
from comfy_controller.testing.fake_judge import FakeJudge  # noqa: E402

N_ASSETS = 12

# A minimal API-format graph: exactly the shape ComfyUI's "Save (API Format)"
# export produces, which is what the controller patches and POSTs.
GRAPH = {
    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 20}},
    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "a product on a plinth"}},
    "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "demo"}},
}


def banner(msg: str) -> None:
    print(f"\n\033[1;36m{'─' * 70}\n  {msg}\n{'─' * 70}\033[0m", flush=True)


async def inject_fault_midway(state: FakeComfyState, fault: Fault, after: int) -> None:
    """Flip FakeComfy into a fault once `after` jobs have gone through."""
    while state.submitted_count < after:
        await asyncio.sleep(0.05)
    banner(f"INJECTING {fault.value.upper()} after {state.submitted_count} jobs")
    if fault is Fault.ZOMBIE:
        print("  ComfyUI's worker thread is now dead.")
        print("  HTTP still answers 200. /prompt still accepts jobs. Nothing executes.")
        print("  Watch the supervisor notice via the canary rather than an HTTP probe.\n")
    state.fault = fault
    state.fault_after_n_jobs = 0

    # A real restart clears the fault; emulate that by healing shortly after
    # the supervisor would have relaunched the process.
    await asyncio.sleep(12)
    state.fault = Fault.NONE
    print("\n\033[1;32m  [ComfyUI relaunched by the watchdog — worker alive again]\033[0m\n")


def dump_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    banner("FINAL STATE (the database is the source of truth, not the logs)")
    rows = conn.execute(
        "SELECT state, COUNT(*) c FROM assets GROUP BY state ORDER BY c DESC"
    ).fetchall()
    for r in rows:
        print(f"  {r['state']:<20} {r['c']}")

    total = conn.execute("SELECT COUNT(*) c FROM assets").fetchone()["c"]
    terminal = sum(r["c"] for r in rows if AssetState(r["state"]).is_terminal)
    print(f"\n  terminal: {terminal}/{total}", end="  ")
    print("\033[1;32mOK — every asset ended somewhere\033[0m" if terminal == total
          else "\033[1;31mFAIL — some asset is stranded\033[0m")

    attempts = conn.execute("SELECT COUNT(*) c FROM run_log").fetchone()["c"]
    print(f"  run_log rows: {attempts} (attempts across {total} assets "
          f"= the failure-rate table)")
    conn.close()


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fault", default="zombie",
                    choices=[f.value for f in Fault if f is not Fault.NONE])
    ap.add_argument("--clean", action="store_true", help="no fault injection")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="comfyctl-demo-"))
    (tmp / "output").mkdir()
    (tmp / "wf.json").write_text(json.dumps(GRAPH))

    state = FakeComfyState(step_delay=0.02, output_dir=tmp / "output")
    fake = FakeComfy(state)
    base_url = await fake.start()

    banner("ComfyUI batch controller — demo")
    print(f"  FakeComfy   {base_url}   (stands in for a real ComfyUI on a GPU)")
    print(f"  judge       FakeJudge    (no ANTHROPIC_API_KEY needed)")
    print(f"  workdir     {tmp}")
    print(f"  assets      {N_ASSETS}")
    print(f"  fault       {'none' if args.clean else args.fault}")

    comfy = ComfyHTTP(base_url)
    store = SQLiteStore(tmp / "demo.db")
    judge = FakeJudge()
    notifier = ConsoleNotifier()
    supervisor = HealthSupervisor(
        comfy,
        # Tightened so the demo shows in seconds what takes minutes in production.
        stall_timeout=4.0, canary_timeout=6.0, hysteresis_n=1,
        backoff_base=1.0, gpu_util_fn=lambda: 0.0,
        launch_argv=["true"],  # a no-op "relaunch" -- the fault heals on a timer
    )

    assets = [Asset(id=f"asset-{i:02d}", workflow=str(tmp / "wf.json"))
              for i in range(N_ASSETS)]
    await store.load_batch(assets)

    runner = BatchRunner(
        config=RunnerConfig(
            comfy_output_dir=tmp / "output",
            concurrency=2,
            tick_interval_s=2.0,
            await_outcome_timeout_s=25.0,
        ),
        comfy=comfy, store=store, judge=judge, notifier=notifier,
        supervisor=supervisor, classifier=ErrorClassifier(),
        retry_policy=RetryPolicy(),
    )

    banner("RUNNING")
    tasks = [asyncio.create_task(runner.run())]
    if not args.clean:
        tasks.append(asyncio.create_task(
            inject_fault_midway(state, Fault(args.fault), after=4)))

    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=180)
    except asyncio.TimeoutError:
        print("\n\033[1;31m  demo timed out\033[0m")
    finally:
        for t in tasks:
            t.cancel()
        await comfy.close()
        await supervisor.aclose()
        await store.close()
        await fake.stop()

    dump_db(tmp / "demo.db")
    print(f"\n  Inspect it yourself:\n    sqlite3 {tmp / 'demo.db'} "
          f"'select id,state,attempt,crash_count from assets;'")
    print(f"    datasette {tmp / 'demo.db'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
