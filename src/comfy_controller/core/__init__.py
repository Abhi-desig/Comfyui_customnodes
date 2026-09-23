"""Pure decision logic for the batch controller: no I/O, no async, no network.

Everything here is a plain function/class of its inputs (plus an injected
clock where time matters), so it can be tested exhaustively without a fake
server, a database, or an event loop.
"""

from .policy import (
    QC_MAX_RETRIES,
    BreakerState,
    CircuitBreaker,
    ErrorClassifier,
    DEFAULT_CLASSIFIER,
    GlobalBreaker,
    RetryDecision,
    RetryPolicy,
)
from .sequence import (
    DEFAULT_MAX_RETRIES,
    QCCompare,
    RegenerateFrame,
    RepairPlan,
    RepairStep,
    SequenceRepairPlanner,
)

__all__ = [
    "QC_MAX_RETRIES",
    "BreakerState",
    "CircuitBreaker",
    "ErrorClassifier",
    "DEFAULT_CLASSIFIER",
    "GlobalBreaker",
    "RetryDecision",
    "RetryPolicy",
    "DEFAULT_MAX_RETRIES",
    "QCCompare",
    "RegenerateFrame",
    "RepairPlan",
    "RepairStep",
    "SequenceRepairPlanner",
]
