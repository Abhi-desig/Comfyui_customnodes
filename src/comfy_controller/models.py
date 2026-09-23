"""Domain models. Frozen in Wave 0 — every other module builds against these."""

from __future__ import annotations

import enum
import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- states


class AssetState(str, enum.Enum):
    """Every asset ends in exactly one terminal state."""

    PENDING = "pending"
    CLAIMED = "claimed"          # intent written, not yet submitted
    SUBMITTED = "submitted"      # prompt_id recorded
    RETRY_WAIT = "retry_wait"

    # terminal
    SAVED = "saved"
    PARKED_APPROVAL = "parked_approval"
    JUDGE_UNAVAILABLE = "judge_unavailable"
    SKIPPED_REVIEW = "skipped_review"
    PERMANENT_ERROR = "permanent_error"

    @property
    def is_terminal(self) -> bool:
        return self in _TERMINAL

    @property
    def needs_reconciliation(self) -> bool:
        """CLAIMED/SUBMITTED found at startup were interrupted mid-flight."""
        return self in (AssetState.CLAIMED, AssetState.SUBMITTED)


_TERMINAL = frozenset(
    {
        AssetState.SAVED,
        AssetState.PARKED_APPROVAL,
        AssetState.JUDGE_UNAVAILABLE,
        AssetState.SKIPPED_REVIEW,
        AssetState.PERMANENT_ERROR,
    }
)


class ErrorClass(str, enum.Enum):
    TEMPORARY = "temporary"    # OOM, timeout, busy -> retry 10/30/60s
    PERMANENT = "permanent"    # missing file, bad input -> skip now
    UNKNOWN = "unknown"        # 1 retry, then skip


class AssetKind(str, enum.Enum):
    SINGLE = "single"
    SEQUENCE = "sequence"


# --------------------------------------------------------------------------- assets


class Asset(BaseModel):
    """One unit of work from the batch manifest."""

    id: str
    kind: AssetKind = AssetKind.SINGLE
    workflow: str                                  # path to API-format JSON
    inputs: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    frame_count: int | None = None                 # sequences only

    def job_key(self, model_version: str, graph_hash: str) -> str:
        """Deterministic key. Drives filename_prefix so reconciliation can find
        orphaned outputs even when the prompt_id was never durably recorded."""
        payload = json.dumps(
            {
                "id": self.id,
                "inputs": self.inputs,
                "seed": self.seed,
                "graph": graph_hash,
                "model": model_version,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class AssetRecord(BaseModel):
    """Persisted row. The database is the source of truth, not the logs."""

    asset: Asset
    state: AssetState = AssetState.PENDING
    attempt: int = 0
    job_key: str | None = None
    comfy_prompt_id: str | None = None
    output_paths: list[str] = Field(default_factory=list)
    last_error: ExecutionError | None = None
    last_verdict: QCVerdict | None = None
    crash_count: int = 0          # design doc: skip after 2 crashes on one asset


# --------------------------------------------------------------------------- comfy


class ExecutionError(BaseModel):
    """Mirrors ComfyUI's `execution_error` WS payload.

    Note: ComfyUI sends this with broadcast=False, so it only reaches the client
    whose `client_id` submitted the prompt.
    """

    prompt_id: str
    node_id: str | None = None
    node_type: str | None = None
    exception_message: str = ""
    exception_type: str = ""
    traceback: list[str] = Field(default_factory=list)


class JobOutcome(BaseModel):
    prompt_id: str
    succeeded: bool
    output_paths: list[str] = Field(default_factory=list)
    error: ExecutionError | None = None


class HealthVerdict(str, enum.Enum):
    HEALTHY = "healthy"            # WS events arriving
    BUSY = "busy"                  # silent but GPU pegged
    SUSPECT = "suspect"            # silent + GPU idle -> run canary
    ZOMBIE = "zombie"              # canary never completed
    DEAD = "dead"                  # HTTP probe failed


# ----------------------------------------------------------------------- qc / judge


class CheckResult(BaseModel):
    """One binary rubric check.

    Field order is load-bearing: `evidence` is declared before `passed` so the
    model must state what it saw before committing to a verdict (G-Eval).
    """

    name: str
    evidence: str
    passed: bool


FixAction = Literal["reseed", "adjust_steps", "adjust_strength", "rewrite_slot", "none"]


class FixProposal(BaseModel):
    """Enumerated and bounded. The judge never emits free text that reaches a
    generation prompt — only an action from this closed vocabulary."""

    action: FixAction = "none"
    delta: float | None = None          # adjust_* only; clamped by the applier
    slot: str | None = None             # rewrite_slot only; must be allowlisted
    text: str | None = None             # rewrite_slot only; goes into a fixed template


class QCVerdict(BaseModel):
    checks: list[CheckResult]
    fix: FixProposal = Field(default_factory=FixProposal)
    critique: str = ""                  # display-only. Never reaches a prompt.

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failed_checks(self) -> list[str]:
        return [c.name for c in self.checks if not c.passed]


# AssetRecord forward-references types defined below it; resolve them now.
AssetRecord.model_rebuild()
