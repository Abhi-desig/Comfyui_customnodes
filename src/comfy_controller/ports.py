"""Port protocols. Frozen in Wave 0.

Every adapter implements one of these; the core depends only on these. Wave 1
agents can work in parallel because nothing here changes once reviewed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .models import (
    Asset,
    AssetRecord,
    AssetState,
    ExecutionError,
    HealthVerdict,
    JobOutcome,
    QCVerdict,
)


@runtime_checkable
class ComfyPort(Protocol):
    """Talks to one ComfyUI server."""

    async def submit(self, graph: dict[str, Any], job_key: str, front: bool = False) -> str:
        """POST /prompt. Returns prompt_id.

        Implementations MUST send a stable client_id and connect the websocket
        with the same value — `execution_error` is broadcast=False and reaches
        only the submitting client.
        `front=True` jumps the queue (required for the canary).
        """
        ...

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """Yield decoded WS events: status, execution_start, executing, progress,
        executed, execution_error, execution_success, execution_interrupted."""
        ...

    async def await_outcome(self, prompt_id: str, timeout: float) -> JobOutcome: ...

    async def queue_depth(self) -> tuple[int, int]:
        """(running, pending). Note ComfyUI returns arrays of tuples, not dicts."""
        ...

    async def history(self, prompt_id: str) -> dict[str, Any] | None: ...

    async def is_responding(self) -> bool:
        """HTTP liveness only. NEVER treat this as worker liveness — it returns
        200 straight through a zombie (#16312)."""
        ...

    async def free(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """POST /free. Sets flags consumed *between* jobs; not synchronous."""
        ...

    async def interrupt(self, prompt_id: str | None = None) -> None: ...


@runtime_checkable
class SupervisorPort(Protocol):
    """Owns the ComfyUI process lifecycle."""

    async def assess(self) -> HealthVerdict: ...
    async def restart(self, reason: str) -> bool:
        """Returns False when the restart-intensity ceiling is hit (hold + alert)."""
        ...
    async def maybe_recycle(self, jobs_since_start: int) -> bool:
        """Proactive recycle every N jobs, before rot sets in."""
        ...


@runtime_checkable
class JudgePort(Protocol):
    async def judge(self, asset: Asset, image_paths: list[Path]) -> QCVerdict:
        """Raises JudgeUnavailable when the judge itself failed — the caller must
        park the asset, never fail open into auto-approval."""
        ...


@runtime_checkable
class StorePort(Protocol):
    async def load_batch(self, assets: list[Asset]) -> None: ...

    async def claim_next(self) -> AssetRecord | None:
        """Write-intent-before-act. Conditional UPDATE is the lock:
        `WHERE id=? AND state IN ('pending','retry_wait')`."""
        ...

    async def record_submission(self, asset_id: str, prompt_id: str) -> None: ...

    async def transition(
        self, asset_id: str, state: AssetState, *,
        error: ExecutionError | None = None,
        verdict: QCVerdict | None = None,
        output_paths: list[str] | None = None,
        retry_after_s: float | None = None,
    ) -> None: ...

    async def needing_reconciliation(self) -> list[AssetRecord]:
        """Rows in CLAIMED/SUBMITTED at startup — verify against ComfyUI, never
        trust the DB alone."""
        ...

    async def summary(self) -> dict[AssetState, int]: ...


@runtime_checkable
class NotifierPort(Protocol):
    async def batch_started(self, total: int) -> None: ...
    async def milestone(self, done: int, total: int) -> None: ...
    async def alert(self, title: str, detail: str) -> None: ...
    async def morning_report(self, summary: dict[AssetState, int], records: list[AssetRecord]) -> None: ...


class JudgeUnavailable(RuntimeError):
    """Judge infrastructure failed (429 exhausted, 529, refusal, truncation,
    schema-valid garbage). Distinct from 'the asset failed the rubric'."""
