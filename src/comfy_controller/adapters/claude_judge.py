"""JudgePort implementation backed by the Claude Messages API (vision + structured
outputs).

Design constraints this file exists to satisfy (see task brief for the full
rationale, this is the short version):

  1. Binary checks, not a 1-10 score — see config/rubric.yaml.
  2. Evidence before verdict — `CheckResult` (models.py, frozen) already
     declares `evidence` before `passed`; we send that exact schema to
     `messages.parse(output_format=QCVerdict)` so the model is structurally
     forced to write evidence before it can commit to a verdict (G-Eval).
  3. Enumerated fix actions only, validated and clamped — `_combine_fix()`.
  4. Self-consistency N=3 majority vote, with a `low_confidence` signal for a
     genuine split — `_combine()`.
  5. Prompt caching: rubric + exemplars in a stable `system` block with a
     single trailing `cache_control` breakpoint (ephemeral, 1h), built once
     at construction time so nothing volatile can leak into the prefix.
  6. Model defaults to claude-opus-5, adaptive thinking, low effort.
  7. JudgePort contract: raise JudgeUnavailable rather than fail open, on
     every trigger listed in `_call_once()`.
  8. Cost accounting via `CostTracker`.

QCVerdict.passed (frozen, in models.py) is `all(c.passed for c in checks)` —
an unconditional AND over every entry. Because that property can't be
changed, only `required: true` rubric checks are allowed into
`QCVerdict.checks`: an optional/nice-to-have check failing must never flip
`passed` to False. Optional checks are still asked (for grounding) and their
results are folded into `critique` (display-only) instead.
"""

from __future__ import annotations

import asyncio
import base64
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import structlog
import yaml

from ..models import Asset, CheckResult, FixAction, FixProposal, QCVerdict
from ..ports import JudgeUnavailable
from . import prefilter
from .prefilter import PrefilterConfig

log = structlog.get_logger(__name__)

_MEDIA_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

# Evidence this short or this generic reads as templated, not observed — the
# whole point of "evidence before verdict" is defeated if the model can just
# write "N/A" and move on.
_TEMPLATED_PHRASES = {"n/a", "none", "na", "looks good", "no issues", "ok", "fine", ""}

_FIX_ACTIONS_WITH_DELTA = {"adjust_steps", "adjust_strength"}


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class RubricCheck:
    name: str
    question: str
    required: bool = True
    exemplar_images: tuple[Path, ...] = ()


@dataclass(frozen=True)
class RubricConfig:
    model: str = "claude-opus-5"
    cost_lever_model: str = "claude-sonnet-5"
    effort: str = "low"
    max_tokens: int = 4096
    cache_ttl: str = "1h"
    self_consistency_n: int = 3
    max_retry_attempts: int = 2
    retry_after_default_s: float = 5.0
    overload_backoff_base_s: float = 4.0
    min_evidence_length: int = 8
    checks: tuple[RubricCheck, ...] = ()
    fix_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    slot_allowlist: tuple[str, ...] = ()
    max_fix_text_length: int = 280
    prefilter: PrefilterConfig = field(default_factory=PrefilterConfig)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25

    @property
    def required_check_names(self) -> frozenset[str]:
        return frozenset(c.name for c in self.checks if c.required)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RubricConfig:
        path = Path(path)
        raw = yaml.safe_load(path.read_text()) or {}
        return cls.from_dict(raw, base_dir=path.parent)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, base_dir: Path) -> RubricConfig:
        checks = tuple(
            RubricCheck(
                name=c["name"],
                question=c["question"],
                required=bool(c.get("required", True)),
                exemplar_images=tuple(base_dir / p for p in c.get("exemplar_images", ())),
            )
            for c in raw.get("checks", ())
        )

        fix_raw = raw.get("fix", {})
        bounds = {
            action: (float(b["min"]), float(b["max"])) for action, b in fix_raw.get("bounds", {}).items()
        }
        slot_allowlist = tuple(fix_raw.get("slot_allowlist", ()))

        pricing_raw = dict(raw.get("pricing", {}))
        cache_read_multiplier = float(pricing_raw.pop("cache_read_multiplier", 0.1))
        cache_write_multiplier = float(pricing_raw.pop("cache_write_multiplier", 1.25))

        retry_raw = raw.get("retry", {})
        content_raw = raw.get("content_validation", {})
        sc_raw = raw.get("self_consistency", {})

        return cls(
            model=raw.get("model", "claude-opus-5"),
            cost_lever_model=raw.get("cost_lever_model", "claude-sonnet-5"),
            effort=raw.get("effort", "low"),
            max_tokens=int(raw.get("max_tokens", 4096)),
            cache_ttl=str(raw.get("cache_ttl", "1h")),
            self_consistency_n=int(sc_raw.get("n", 3)),
            max_retry_attempts=int(retry_raw.get("max_attempts", 2)),
            retry_after_default_s=float(retry_raw.get("retry_after_default_s", 5.0)),
            overload_backoff_base_s=float(retry_raw.get("overload_backoff_base_s", 4.0)),
            min_evidence_length=int(content_raw.get("min_evidence_length", 8)),
            checks=checks,
            fix_bounds=bounds,
            slot_allowlist=slot_allowlist,
            max_fix_text_length=int(fix_raw.get("max_text_length", 280)),
            prefilter=PrefilterConfig.from_dict(raw.get("prefilter")),
            pricing=pricing_raw,
            cache_read_multiplier=cache_read_multiplier,
            cache_write_multiplier=cache_write_multiplier,
        )


# ---------------------------------------------------------------- cost tracking


@dataclass
class CostTracker:
    """Running usage/cost total across a batch.

    This class only *records* — enforcing a hard cap is the runner's job
    (design constraint 8: "expose a running total so the runner can enforce a
    hard cap"). We just make sure the number is always right.
    """

    pricing: dict[str, dict[str, float]]
    cache_read_multiplier: float = 0.1
    cache_write_multiplier: float = 1.25
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    total_usd: float = 0.0
    calls: int = 0

    def record(self, model: str, usage: Any) -> None:
        rates = self.pricing.get(model, {})
        input_rate = float(rates.get("input_per_mtok", 0.0))
        output_rate = float(rates.get("output_per_mtok", 0.0))

        input_tok = getattr(usage, "input_tokens", 0) or 0
        output_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0

        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.cache_read_tokens += cache_read
        self.cache_creation_tokens += cache_creation
        self.calls += 1

        cost = (
            input_tok * input_rate
            + output_tok * output_rate
            + cache_read * input_rate * self.cache_read_multiplier
            + cache_creation * input_rate * self.cache_write_multiplier
        ) / 1_000_000
        self.total_usd += cost

        log.debug(
            "judge.usage",
            model=model,
            input_tokens=input_tok,
            output_tokens=output_tok,
            cache_read_input_tokens=cache_read,
            cache_creation_input_tokens=cache_creation,
            running_total_usd=round(self.total_usd, 4),
        )
        if cache_read == 0 and self.calls > 1:
            # Not necessarily wrong (first call after TTL expiry, config just
            # changed) — but persistent zero reads across a batch means a
            # silent invalidator crept into the "stable" system prefix.
            log.warning("judge.cache_read_zero", model=model, call_number=self.calls)


# -------------------------------------------------------------------- helpers


def _encode_image_block(path: Path) -> dict[str, Any]:
    media_type = _MEDIA_TYPES.get(path.suffix.lower(), "image/png")
    data = base64.standard_b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _retry_after_seconds(exc: anthropic.RateLimitError, *, default: float) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def _is_overload(exc: anthropic.InternalServerError) -> bool:
    return getattr(exc, "status_code", None) == 529 or getattr(exc, "type", None) == "overloaded_error"


# ----------------------------------------------------------------- the judge


class ClaudeJudge:
    """Vision QC judge for one asset's rendered image(s)."""

    def __init__(
        self,
        client: Any,
        config: RubricConfig,
        *,
        cost_tracker: CostTracker | None = None,
        sleep: Any = asyncio.sleep,
    ) -> None:
        """`client` is an `anthropic.AsyncAnthropic`-shaped object (only
        `.messages.parse(...)` is used) — tests inject a fake. `sleep` is
        injectable so retry-backoff tests don't actually block."""
        self._client = client
        self._config = config
        self.cost = cost_tracker or CostTracker(
            pricing=config.pricing,
            cache_read_multiplier=config.cache_read_multiplier,
            cache_write_multiplier=config.cache_write_multiplier,
        )
        self._sleep = sleep
        # Built once, from static config only — see the module docstring on
        # why nothing per-asset (ids, timestamps, uuids) may ever enter this.
        self._system_blocks = self._build_system_blocks()

    # -------------------------------------------------------------- system prompt

    def _build_system_blocks(self) -> list[dict[str, Any]]:
        lines = [
            "You are a strict, evidence-first QC judge for AI-generated marketing images.",
            "For every rubric check below, first describe exactly what you observe in the "
            "image(s), then decide pass/fail. Never invent details you cannot see — if you "
            "are uncertain, say so in the evidence and fail the check.",
            "",
            "Rubric checks (answer every single one, using exactly these names):",
        ]
        for i, check in enumerate(self._config.checks, 1):
            tag = "required" if check.required else "optional — does not block on its own"
            lines.append(f"{i}. [{check.name}] ({tag}) {check.question}")
        lines += [
            "",
            "Only propose a fix when at least one required check fails. `action` must be "
            "exactly one of: reseed, adjust_steps, adjust_strength, rewrite_slot, none.",
            f"If `action` is rewrite_slot, `slot` must be exactly one of: "
            f"{', '.join(self._config.slot_allowlist) or '(none configured)'}. Any other value "
            "will be rejected.",
            "`critique` is a free-text note for a human reviewer. It is never applied "
            "automatically and never reaches an image-generation prompt.",
        ]
        system_text = "\n".join(lines)

        blocks: list[dict[str, Any]] = [{"type": "text", "text": system_text}]
        for check in self._config.checks:
            for image_path in check.exemplar_images:
                blocks.append({"type": "text", "text": f"Exemplar for check '{check.name}':"})
                blocks.append(_encode_image_block(image_path))

        # A single trailing breakpoint caches this whole prefix (rubric text +
        # every exemplar image) as one unit — images go in the *next* message,
        # after this breakpoint, so a new asset's bytes never invalidate it.
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral", "ttl": self._config.cache_ttl}}
        return blocks

    def _user_content(self, asset: Asset, image_blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompt = (
            f"Evaluate the attached image(s) for asset `{asset.id}` (workflow: {asset.workflow}) "
            "against every rubric check above. Respond with the structured schema only."
        )
        return [*image_blocks, {"type": "text", "text": prompt}]

    # -------------------------------------------------------------------- public

    async def judge(self, asset: Asset, image_paths: list[Path]) -> QCVerdict:
        prefilter_failures = prefilter.evaluate(image_paths, self._config.prefilter)
        if prefilter_failures:
            log.info("judge.prefilter_rejected", asset_id=asset.id, checks=prefilter_failures[0].name)
            return QCVerdict(
                checks=prefilter_failures,
                fix=FixProposal(action="none"),
                critique="Rejected by the deterministic prefilter — the judge API was not called.",
            )

        image_blocks = [_encode_image_block(p) for p in image_paths]
        n = max(1, self._config.self_consistency_n)

        samples = await asyncio.gather(*[self._call_once(asset, image_blocks) for _ in range(n)])
        return self._combine(list(samples))

    # ------------------------------------------------------------------- one call

    async def _call_once(self, asset: Asset, image_blocks: list[dict[str, Any]]) -> QCVerdict:
        attempt = 0
        while True:
            attempt += 1
            try:
                response = await self._client.messages.parse(
                    model=self._config.model,
                    max_tokens=self._config.max_tokens,
                    system=self._system_blocks,
                    thinking={"type": "adaptive"},
                    output_config={"effort": self._config.effort},
                    messages=[{"role": "user", "content": self._user_content(asset, image_blocks)}],
                    output_format=QCVerdict,
                )
            except anthropic.RateLimitError as exc:
                retry_after = _retry_after_seconds(exc, default=self._config.retry_after_default_s)
                if attempt > self._config.max_retry_attempts:
                    raise JudgeUnavailable(
                        f"judge rate limited (429) after {attempt} attempts; retry-after={retry_after}s"
                    ) from exc
                log.warning("judge.rate_limited", asset_id=asset.id, attempt=attempt, retry_after=retry_after)
                await self._sleep(retry_after)
                continue
            except anthropic.InternalServerError as exc:
                overloaded = _is_overload(exc)
                if attempt > self._config.max_retry_attempts:
                    kind = "529 capacity" if overloaded else f"{getattr(exc, 'status_code', '5xx')}"
                    raise JudgeUnavailable(f"judge server error ({kind}) after {attempt} attempts") from exc
                # 529 backs off harder than a plain 5xx, and grows with each attempt —
                # distinct from 429, which instead honours the server's own retry-after.
                backoff = (
                    self._config.overload_backoff_base_s * (2 ** (attempt - 1))
                    if overloaded
                    else self._config.retry_after_default_s
                )
                log.warning("judge.server_error", asset_id=asset.id, attempt=attempt, overloaded=overloaded)
                await self._sleep(backoff)
                continue
            except anthropic.APIConnectionError as exc:
                raise JudgeUnavailable(f"judge network error: {exc}") from exc
            except anthropic.APIStatusError as exc:
                raise JudgeUnavailable(f"judge API error ({exc.status_code}): {exc.message}") from exc
            break

        if response.stop_reason == "refusal":
            raise JudgeUnavailable("judge refused to respond (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise JudgeUnavailable("judge response truncated (stop_reason=max_tokens)")

        verdict = response.parsed_output
        if verdict is None:
            raise JudgeUnavailable("judge response failed structured-output validation")

        self._validate_content(verdict)

        usage = getattr(response, "usage", None)
        if usage is not None:
            self.cost.record(self._config.model, usage)

        return verdict

    def _validate_content(self, verdict: QCVerdict) -> None:
        """Schema-valid is not the same as trustworthy. Catch a garbage
        response the schema itself can't rule out: missing required checks,
        or evidence that's empty/templated (defeats the whole G-Eval point)."""
        present = {c.name for c in verdict.checks}
        missing = self._config.required_check_names - present
        if missing:
            raise JudgeUnavailable(f"judge response missing required checks: {sorted(missing)}")

        for check in verdict.checks:
            evidence = check.evidence.strip()
            if len(evidence) < self._config.min_evidence_length or evidence.lower() in _TEMPLATED_PHRASES:
                raise JudgeUnavailable(
                    f"judge response has empty/templated evidence for check '{check.name}': {evidence!r}"
                )

    # ------------------------------------------------------------ self-consistency

    def _combine(self, samples: list[QCVerdict]) -> QCVerdict:
        required_names = [c.name for c in self._config.checks if c.required]
        optional_names = [c.name for c in self._config.checks if not c.required]

        combined_checks: list[CheckResult] = []
        split_notes: list[str] = []

        for name in required_names:
            votes = [c for s in samples for c in s.checks if c.name == name]
            if not votes:
                # _validate_content already guards every individual sample, so
                # this can only happen if judge() was called with a fabricated
                # sample list (e.g. a test) that skipped validation.
                raise JudgeUnavailable(f"required check '{name}' missing from every sample")

            pass_votes = sum(1 for v in votes if v.passed)
            majority_passed = pass_votes > len(votes) / 2
            if len(set(v.passed for v in votes)) > 1:
                split_notes.append(f"{name}: {pass_votes}/{len(votes)} samples passed")

            evidence = next(v.evidence for v in votes if v.passed == majority_passed)
            combined_checks.append(CheckResult(name=name, evidence=evidence, passed=majority_passed))

        # Optional checks never enter `checks` (see module docstring) — fold
        # their outcome into `critique` instead so the information survives.
        optional_summary = []
        for name in optional_names:
            votes = [c for s in samples for c in s.checks if c.name == name]
            if not votes:
                continue
            majority_passed = sum(1 for v in votes if v.passed) > len(votes) / 2
            optional_summary.append(f"{name}: {'pass' if majority_passed else 'fail'}")

        low_confidence = bool(split_notes)
        if low_confidence:
            # A synthetic, always-gating entry: the caller can distinguish
            # "genuinely failed a rubric check" from "the judge itself
            # couldn't agree with itself" by checking this exact name in
            # `verdict.failed_checks`, and should park rather than auto-fix.
            combined_checks.append(
                CheckResult(
                    name="_low_confidence",
                    evidence="self-consistency split: " + "; ".join(split_notes),
                    passed=False,
                )
            )

        fix = self._combine_fix(samples)

        critique_parts = [s.critique.strip() for s in samples if s.critique.strip()]
        if optional_summary:
            critique_parts.append("optional checks — " + ", ".join(optional_summary))
        critique = " | ".join(dict.fromkeys(critique_parts))  # de-dupe, keep order

        return QCVerdict(checks=combined_checks, fix=fix, critique=critique)

    def _combine_fix(self, samples: list[QCVerdict]) -> FixProposal:
        actions = [s.fix.action for s in samples]
        counts = Counter(actions)
        top = max(counts.values())
        winners = [a for a, c in counts.items() if c == top]
        # A tied vote never auto-applies anything — "none" is the only safe
        # default when the samples don't agree on what to do.
        action: FixAction = winners[0] if len(winners) == 1 else "none"
        if action == "none":
            return FixProposal(action="none")

        agreeing = [s.fix for s in samples if s.fix.action == action]
        return self._validate_and_clamp(action, agreeing)

    def _validate_and_clamp(self, action: FixAction, proposals: list[FixProposal]) -> FixProposal:
        """The prompt-injection defence (OWASP LLM01/LLM05): a hijacked judge
        can only move within this enumerated, bounded envelope."""
        if action == "reseed":
            return FixProposal(action="reseed")

        if action in _FIX_ACTIONS_WITH_DELTA:
            deltas = [p.delta for p in proposals if p.delta is not None]
            if not deltas:
                return FixProposal(action="none")
            bounds = self._config.fix_bounds.get(action)
            avg_delta = sum(deltas) / len(deltas)
            if bounds is not None:
                lo, hi = bounds
                avg_delta = max(lo, min(hi, avg_delta))
            return FixProposal(action=action, delta=avg_delta)

        if action == "rewrite_slot":
            slots = [p.slot for p in proposals if p.slot]
            if not slots:
                return FixProposal(action="none")
            slot = Counter(slots).most_common(1)[0][0]
            if slot not in self._config.slot_allowlist:
                # Unlisted slot — reject outright, never partially apply.
                return FixProposal(action="none")
            texts = [p.text for p in proposals if p.slot == slot and p.text]
            text = texts[0] if texts else None
            if text and len(text) > self._config.max_fix_text_length:
                text = text[: self._config.max_fix_text_length]
            return FixProposal(action="rewrite_slot", slot=slot, text=text)

        return FixProposal(action="none")
