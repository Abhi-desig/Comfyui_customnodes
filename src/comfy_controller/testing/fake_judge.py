"""A scriptable fake `JudgePort` for other adapters/wave-1 code to build and
test against without depending on claude_judge.py, PIL, numpy, or a real API
key.

Usage:

    judge = FakeJudge()
    judge.queue_pass()
    judge.queue_fail(name="anatomy_correct", evidence="six fingers on the left hand")
    judge.queue_unavailable("simulated 529")

    verdict = await judge.judge(asset, [Path("a.png")])   # -> the first queued pass
    verdict = await judge.judge(asset, [Path("b.png")])   # -> the queued failure
    await judge.judge(asset, [Path("c.png")])              # -> raises JudgeUnavailable
    await judge.judge(asset, [Path("d.png")])              # -> queue empty: repeats default_verdict
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from ..models import Asset, CheckResult, FixProposal, QCVerdict
from ..ports import JudgeUnavailable


def _default_verdict() -> QCVerdict:
    return QCVerdict(checks=[CheckResult(name="ok", evidence="fake judge default pass", passed=True)])


@dataclass
class FakeJudge:
    """Implements `JudgePort`. Records every call it receives so a test can
    assert on what the caller sent, and plays back scripted outcomes in the
    order they were queued."""

    default_verdict: QCVerdict = field(default_factory=_default_verdict)
    _queue: deque[QCVerdict | Exception] = field(default_factory=deque)
    calls: list[tuple[str, list[Path]]] = field(default_factory=list)

    async def judge(self, asset: Asset, image_paths: list[Path]) -> QCVerdict:
        self.calls.append((asset.id, list(image_paths)))
        outcome = self._queue.popleft() if self._queue else self.default_verdict
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    # ------------------------------------------------------------------- scripting

    def queue_verdict(self, verdict: QCVerdict) -> None:
        self._queue.append(verdict)

    def queue_pass(self, *, name: str = "ok", evidence: str = "scripted pass", **verdict_kwargs) -> None:
        self._queue.append(
            QCVerdict(checks=[CheckResult(name=name, evidence=evidence, passed=True)], **verdict_kwargs)
        )

    def queue_fail(
        self,
        *,
        name: str = "check",
        evidence: str = "scripted failure",
        fix: FixProposal | None = None,
        **verdict_kwargs,
    ) -> None:
        self._queue.append(
            QCVerdict(
                checks=[CheckResult(name=name, evidence=evidence, passed=False)],
                fix=fix or FixProposal(action="none"),
                **verdict_kwargs,
            )
        )

    def queue_unavailable(self, message: str = "fake judge unavailable") -> None:
        self._queue.append(JudgeUnavailable(message))
