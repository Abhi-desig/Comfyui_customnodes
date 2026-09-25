"""Calibration harness: measure the judge (`adapters/claude_judge.py`) against
human labels before anyone is allowed to let it auto-skip assets.

Why this file exists (see the task brief for the full rationale): every
existing test in this repo drives `ClaudeJudge` through a scripted fake
`client.messages.parse()` (see `tests/test_judge.py`). That proves the
plumbing -- retries, self-consistency voting, fix clamping -- is correct. It
proves nothing about whether the judge's *opinion* tracks a human's opinion
on a real image. This module is the harness that measures that, and only
that.

Two separate questions this harness is built to keep apart, because they
have different fixes:

  1. Judge vs. human disagreement -- fixed by a better rubric, better
     exemplars, or a different model.
  2. Judge vs. *itself* disagreement -- the same image, judged again, gets a
     different verdict. No amount of extra human labelling fixes this: if
     the judge can't agree with itself, it can't be trusted to agree with a
     human either. `--repeats N` (N >= 2) re-runs the judge on every image so
     this can be measured directly, and `render_summary()` reports it next to
     the human-disagreement rate instead of letting it hide inside "the
     judge was sometimes wrong."

With 30-50 labelled images -- a realistic size for a first labelling pass --
every metric below is reported as a bootstrap confidence interval, and the
interval *width* is the headline, not the point estimate. A precision of
0.80 means nothing on its own if the 95% CI is [0.45, 0.98].

Usage:

    # Real run (spends money -- estimates cost and asks for confirmation
    # first):
    python -m comfy_controller.calibration --labels labels.csv --repeats 3

    # Harness self-test: FakeJudge, no network, no spend, no confirmation:
    python -m comfy_controller.calibration --labels labels.csv --fake

See docs/calibration.md for how to assemble a labelled set, how to read the
output, and what result should block a production run.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from PIL import Image

from .adapters.claude_judge import ClaudeJudge, CostTracker, RubricConfig
from .models import Asset, QCVerdict
from .ports import JudgeUnavailable
from .testing.fake_judge import FakeJudge

# --------------------------------------------------------------------- labels


@dataclass(frozen=True)
class Label:
    """One human-labelled row.

    `should_reject` mirrors the judge's own decision axis (`not
    QCVerdict.passed`): True means a human marked this image as a QC failure
    that a production run must reject. `violated_checks`, when the human
    bothered to name them, is what makes per-check agreement (as opposed to
    just overall pass/fail agreement) possible at all.
    """

    image: Path
    should_reject: bool
    violated_checks: frozenset[str] = frozenset()


def _parse_bool_label(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"fail", "failed", "reject", "rejected", "true", "1"}:
        return True
    if value in {"pass", "passed", "accept", "accepted", "false", "0"}:
        return False
    raise ValueError(f"unrecognised label value {raw!r} -- expected pass/fail (or reject/accept)")


def _parse_violated_checks(raw: Any) -> frozenset[str]:
    if raw is None or raw == "":
        return frozenset()
    if isinstance(raw, str):
        # Semicolons are the primary separator so a plain CSV field never
        # needs quoting; a comma-separated string (as YAML flow lists render,
        # or as a human might just type it) works too.
        parts = raw.replace(",", ";").split(";")
        return frozenset(p.strip() for p in parts if p.strip())
    return frozenset(str(p).strip() for p in raw if str(p).strip())


def _load_labels_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _load_labels_yaml(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text()) or []
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a YAML list of {{image, label, violated_checks?}} entries")
    return raw


def load_labels(labels_path: str | Path, *, images_dir: str | Path | None = None) -> list[Label]:
    """Parses a CSV or YAML labels file. Image paths are resolved against
    `images_dir` if given, otherwise against the labels file's own directory
    -- either way, a labelled set is "a directory of images plus a labels
    file" as a single portable unit."""
    labels_path = Path(labels_path)
    base = Path(images_dir) if images_dir else labels_path.parent
    suffix = labels_path.suffix.lower()

    if suffix == ".csv":
        rows = _load_labels_csv(labels_path)
    elif suffix in (".yaml", ".yml"):
        rows = _load_labels_yaml(labels_path)
    else:
        raise ValueError(f"unsupported labels file type {labels_path.suffix!r} -- use .csv or .yaml")

    labels = []
    for row in rows:
        if "image" not in row or "label" not in row:
            raise ValueError(f"{labels_path}: row {row!r} is missing an 'image' or 'label' column/key")
        labels.append(
            Label(
                image=base / str(row["image"]),
                should_reject=_parse_bool_label(str(row["label"])),
                violated_checks=_parse_violated_checks(row.get("violated_checks")),
            )
        )
    return labels


def validate_labels_against_rubric(labels: Sequence[Label], rubric: RubricConfig) -> list[str]:
    """Non-fatal sanity check: a typo'd check name in the labels file
    silently drops that image out of per-check agreement with no error
    anywhere. Catch it up front instead."""
    known = {c.name for c in rubric.checks}
    warnings = []
    for label in labels:
        unknown = label.violated_checks - known
        if unknown:
            warnings.append(
                f"{label.image.name}: violated_checks {sorted(unknown)} not in rubric.yaml "
                f"(known checks: {sorted(known)}) -- typo?"
            )
        if not label.image.exists():
            warnings.append(f"{label.image}: file does not exist")
    return warnings


# ---------------------------------------------------------------- cost estimate

# Anthropic's documented rough proxy for vision token cost: tokens scale with
# pixel count. This is intentionally crude -- it is a pre-flight estimate to
# gate a spend decision, not a billing reconciliation.
_PX_PER_TOKEN = 750.0
_FALLBACK_IMAGE_TOKENS = 1200
# Used only if the rubric ever wires up exemplar images and one can't be
# opened for its real dimensions (see config/rubric.yaml's `exemplar_images`
# slots, currently empty).
_FALLBACK_EXEMPLAR_TOKENS = 1200


def _estimate_image_tokens(path: Path) -> int:
    try:
        with Image.open(path) as img:
            w, h = img.size
        return max(1, int((w * h) / _PX_PER_TOKEN))
    except OSError:
        return _FALLBACK_IMAGE_TOKENS


def _estimate_system_prompt_tokens(rubric: RubricConfig, *, chars_per_token: float = 4.0) -> int:
    """A rough, independent proxy for the cached system prefix
    `ClaudeJudge._build_system_blocks()` sends -- not a byte-for-byte mirror
    of that private method, deliberately, so this estimator doesn't silently
    drift out of sync with an internal refactor there. Good enough to gate a
    go/no-go spend decision; re-check `actual_cost_usd` in the report after
    the first real run and adjust `chars_per_token` here if it's consistently
    off by a lot.
    """
    chars = 600  # fixed instructional boilerplate (see claude_judge.py's system text)
    for check in rubric.checks:
        chars += len(check.name) + len(check.question) + 40
    tokens = int(chars / chars_per_token)
    for check in rubric.checks:
        for image_path in check.exemplar_images:
            tokens += _estimate_image_tokens(image_path) if image_path.exists() else _FALLBACK_EXEMPLAR_TOKENS
    return tokens


@dataclass
class _Usage:
    """Duck-types whatever `CostTracker.record()` reads off a real
    `anthropic` usage object, so the exact same cost formula backs both the
    pre-flight estimate and the real accounting -- one place defines what a
    token costs."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0


@dataclass(frozen=True)
class CostEstimate:
    model: str
    n_images: int
    repeats: int
    internal_self_consistency: int
    total_judge_calls: int
    estimated_usd: float
    avg_image_tokens: int
    avg_output_tokens: int

    def pretty(self) -> str:
        return (
            "Calibration cost estimate (upper bound -- the deterministic prefilter may reject\n"
            "some images before the paid API is ever called, which this estimate assumes does\n"
            "not happen):\n"
            f"  model: {self.model}\n"
            f"  {self.n_images} images x {self.repeats} repeats x {self.internal_self_consistency} "
            f"internal self-consistency samples = {self.total_judge_calls} judge API calls\n"
            f"  ~{self.avg_image_tokens} input tokens/image, ~{self.avg_output_tokens} output "
            "tokens/call, one prompt-cache warm-up assumed to cover the whole run\n"
            f"  ESTIMATED SPEND: ${self.estimated_usd:.2f}"
        )


def estimate_cost(
    rubric: RubricConfig,
    labels: Sequence[Label],
    *,
    repeats: int,
    avg_output_tokens: int = 700,
) -> CostEstimate:
    internal_n = max(1, rubric.self_consistency_n)
    total_calls = len(labels) * repeats * internal_n
    if total_calls == 0:
        return CostEstimate(rubric.model, len(labels), repeats, internal_n, 0, 0.0, 0, avg_output_tokens)

    system_tokens = _estimate_system_prompt_tokens(rubric)
    image_tokens = [_estimate_image_tokens(lb.image) for lb in labels] or [_FALLBACK_IMAGE_TOKENS]
    avg_image_tokens = int(sum(image_tokens) / len(image_tokens))

    tracker = CostTracker(
        pricing=rubric.pricing,
        cache_read_multiplier=rubric.cache_read_multiplier,
        cache_write_multiplier=rubric.cache_write_multiplier,
    )
    # First call anywhere in the run pays to *write* the cache; every call
    # after that (within the rubric's cache_ttl, assumed to cover one
    # calibration run) reads it instead -- mirrors CostTracker's own model of
    # what the API actually bills.
    warmup = _Usage(
        input_tokens=avg_image_tokens, output_tokens=avg_output_tokens, cache_creation_input_tokens=system_tokens
    )
    steady = _Usage(input_tokens=avg_image_tokens, output_tokens=avg_output_tokens, cache_read_input_tokens=system_tokens)
    tracker.record(rubric.model, warmup)
    for _ in range(total_calls - 1):
        tracker.record(rubric.model, steady)

    return CostEstimate(
        model=rubric.model,
        n_images=len(labels),
        repeats=repeats,
        internal_self_consistency=internal_n,
        total_judge_calls=total_calls,
        estimated_usd=tracker.total_usd,
        avg_image_tokens=avg_image_tokens,
        avg_output_tokens=avg_output_tokens,
    )


def confirm_spend(estimate: CostEstimate, *, auto_yes: bool = False, input_fn: Any = input) -> bool:
    """Prints the estimate and gates on an explicit yes. Spending real money
    on a calibration run must never happen silently."""
    print(estimate.pretty())
    if auto_yes:
        print("(--yes passed: proceeding without an interactive prompt)")
        return True
    reply = input_fn(f"Proceed and spend an estimated ${estimate.estimated_usd:.2f}? [y/N] ")
    return reply.strip().lower() in {"y", "yes"}


# -------------------------------------------------------------------- running


def _majority_bool(values: Sequence[bool]) -> bool:
    """Ties resolve to True (reject) -- the same fail-safe philosophy as the
    rest of this codebase (`ClaudeJudge._combine_fix`'s tied-vote -> `none`):
    when the judge can't agree with itself, that is itself evidence against
    auto-approving, not a coin flip to break in the judge's favour."""
    true_n = sum(1 for v in values if v)
    return true_n * 2 >= len(values)


@dataclass
class ImageTrial:
    """Every repeat run of the judge against one labelled image."""

    label: Label
    outcomes: list[QCVerdict | JudgeUnavailable] = field(default_factory=list)

    @property
    def valid_verdicts(self) -> list[QCVerdict]:
        return [o for o in self.outcomes if isinstance(o, QCVerdict)]

    @property
    def n_errors(self) -> int:
        return len(self.outcomes) - len(self.valid_verdicts)

    @property
    def decisions(self) -> list[bool]:
        """True == the judge rejected this trial (`not verdict.passed`)."""
        return [not v.passed for v in self.valid_verdicts]

    @property
    def majority_decision(self) -> bool | None:
        d = self.decisions
        return _majority_bool(d) if d else None

    @property
    def self_agreement(self) -> float | None:
        """Fraction of valid repeat runs that agree with this image's own
        majority decision. None (not 1.0) when there's only 0-1 valid runs to
        compare -- "the judge agreed with itself" is meaningless without a
        second opinion to check against."""
        d = self.decisions
        if len(d) < 2:
            return None
        maj = _majority_bool(d)
        return sum(1 for x in d if x == maj) / len(d)

    def check_majority(self, name: str) -> bool | None:
        votes = [next((c.passed for c in v.checks if c.name == name), None) for v in self.valid_verdicts]
        votes = [v for v in votes if v is not None]
        return _majority_bool(votes) if votes else None


async def run_calibration(judge: Any, labels: Sequence[Label], *, repeats: int) -> list[ImageTrial]:
    """Runs `judge.judge()` `repeats` times per labelled image, sequentially.

    Deliberately sequential, not `asyncio.gather`'d across images: a real
    run is already paying for `repeats x internal self-consistency` calls
    per image, and firing all of those as one giant concurrent burst against
    a live rate limit is the opposite of what a careful, budget-gated
    calibration script should do. It also keeps `FakeJudge`'s FIFO queue
    (`src/comfy_controller/testing/fake_judge.py`) trivially scriptable in a
    fixed, predictable order for `--fake` runs and tests.
    """
    trials: list[ImageTrial] = []
    for i, label in enumerate(labels):
        asset = Asset(id=f"calib-{i:04d}-{label.image.stem}", workflow="calibration", inputs={})
        trial = ImageTrial(label=label)
        for _ in range(repeats):
            try:
                verdict = await judge.judge(asset, [label.image])
            except JudgeUnavailable as exc:
                trial.outcomes.append(exc)
            else:
                trial.outcomes.append(verdict)
        trials.append(trial)
    return trials


# -------------------------------------------------------------------- metrics


@dataclass(frozen=True)
class ConfusionCounts:
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    def recall(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    def f1(self) -> float | None:
        p, r = self.precision(), self.recall()
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    def cohens_kappa(self) -> float | None:
        n = self.n
        if n == 0:
            return None
        po = (self.tp + self.tn) / n
        p_reject_human = (self.tp + self.fn) / n
        p_reject_judge = (self.tp + self.fp) / n
        pe = p_reject_human * p_reject_judge + (1 - p_reject_human) * (1 - p_reject_judge)
        if pe >= 1.0:
            # Degenerate marginals (everyone agrees on everything, both axes
            # constant) -- fall back to "did they actually agree" rather than
            # a divide-by-zero NaN.
            return 1.0 if po >= 1.0 else 0.0
        return (po - pe) / (1 - pe)


def confusion_from_pairs(pairs: Sequence[tuple[bool, bool]]) -> ConfusionCounts:
    """`pairs` is (human_should_reject, judge_rejected)."""
    tp = sum(1 for h, j in pairs if h and j)
    fp = sum(1 for h, j in pairs if not h and j)
    fn = sum(1 for h, j in pairs if h and not j)
    tn = sum(1 for h, j in pairs if not h and not j)
    return ConfusionCounts(tp=tp, fp=fp, fn=fn, tn=tn)


def cohens_kappa_from_pairs(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    return confusion_from_pairs(pairs).cohens_kappa()


@dataclass(frozen=True)
class MetricEstimate:
    point: float | None
    ci_low: float | None
    ci_high: float | None
    n_valid_resamples: int
    n_resamples: int

    @property
    def ci_width(self) -> float | None:
        if self.ci_low is None or self.ci_high is None:
            return None
        return self.ci_high - self.ci_low


def bootstrap_metric(
    pairs: Sequence[tuple[bool, bool]],
    metric_fn: Any,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> MetricEstimate:
    """Percentile bootstrap over *images* (each pair is one image's human
    label vs. its judge majority decision) -- resampling images, not repeat
    runs, is what makes the resulting interval answer "how much would this
    number move with a different sample of images," which is exactly the
    question a 30-50 image labelled set can't answer precisely. That's the
    point: the interval should be wide, and this function is what makes the
    width visible instead of hidden behind a single point estimate.
    """
    point = metric_fn(confusion_from_pairs(pairs))
    if not pairs:
        return MetricEstimate(point, None, None, 0, n_resamples)

    rng = np.random.default_rng(seed)
    n = len(pairs)
    values: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resample = [pairs[i] for i in idx]
        v = metric_fn(confusion_from_pairs(resample))
        if v is not None:
            values.append(v)

    if not values:
        return MetricEstimate(point, None, None, 0, n_resamples)
    lo = float(np.percentile(values, 100 * alpha / 2))
    hi = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return MetricEstimate(point, lo, hi, len(values), n_resamples)


@dataclass(frozen=True)
class DecisionMetrics:
    n_images_used: int
    n_images_skipped_no_valid_trial: int
    precision: MetricEstimate
    recall: MetricEstimate
    f1: MetricEstimate
    kappa: MetricEstimate


def decision_metrics(trials: Sequence[ImageTrial], *, n_resamples: int = 2000, seed: int = 0) -> DecisionMetrics:
    pairs: list[tuple[bool, bool]] = []
    skipped = 0
    for t in trials:
        maj = t.majority_decision
        if maj is None:
            skipped += 1
            continue
        pairs.append((t.label.should_reject, maj))

    return DecisionMetrics(
        n_images_used=len(pairs),
        n_images_skipped_no_valid_trial=skipped,
        precision=bootstrap_metric(pairs, lambda c: c.precision(), n_resamples=n_resamples, seed=seed),
        recall=bootstrap_metric(pairs, lambda c: c.recall(), n_resamples=n_resamples, seed=seed + 1),
        f1=bootstrap_metric(pairs, lambda c: c.f1(), n_resamples=n_resamples, seed=seed + 2),
        kappa=bootstrap_metric(pairs, lambda c: c.cohens_kappa(), n_resamples=n_resamples, seed=seed + 3),
    )


@dataclass(frozen=True)
class SelfConsistencySummary:
    n_images_with_repeats: int
    n_images_unstable: int
    mean_self_agreement: float | None
    instability_rate: float | None


def self_consistency_summary(trials: Sequence[ImageTrial]) -> SelfConsistencySummary:
    agreements = [t.self_agreement for t in trials if t.self_agreement is not None]
    unstable = sum(1 for a in agreements if a < 1.0)
    mean_agree = sum(agreements) / len(agreements) if agreements else None
    return SelfConsistencySummary(
        n_images_with_repeats=len(agreements),
        n_images_unstable=unstable,
        mean_self_agreement=mean_agree,
        instability_rate=(1 - mean_agree) if mean_agree is not None else None,
    )


def _expected_check_pass(label: Label, check_name: str) -> bool | None:
    """None means "this image's ground truth for this specific check is
    unknown" -- a human who marked an image as a failure without saying
    which check it violated tells us nothing about any individual check, so
    that image is excluded from that check's agreement count entirely rather
    than guessed at."""
    if not label.should_reject:
        return True
    if label.violated_checks:
        return check_name not in label.violated_checks
    return None


@dataclass(frozen=True)
class CheckAgreement:
    name: str
    n_compared: int
    n_agree: int
    agreement_rate: float | None
    n_false_reject: int  # judge failed it; expected pass -- judge is over-strict here
    n_false_pass: int  # judge passed it; expected fail -- judge is over-lenient here


def per_check_agreement(trials: Sequence[ImageTrial], required_check_names: Sequence[str]) -> list[CheckAgreement]:
    """Per-check agreement, restricted to required checks: those are the
    only checks the judge port exposes structurally (`QCVerdict.checks`) --
    optional checks are folded into free-text `critique` by `ClaudeJudge`
    and can't be read back reliably here. Sorted worst-first, so a single
    badly-worded check surfaces immediately instead of being averaged away
    inside an aggregate score."""
    results = []
    for name in required_check_names:
        n_compared = n_agree = n_false_reject = n_false_pass = 0
        for t in trials:
            expected_pass = _expected_check_pass(t.label, name)
            if expected_pass is None:
                continue
            judge_pass = t.check_majority(name)
            if judge_pass is None:
                continue
            n_compared += 1
            if judge_pass == expected_pass:
                n_agree += 1
            elif judge_pass is False and expected_pass is True:
                n_false_reject += 1
            else:
                n_false_pass += 1
        rate = n_agree / n_compared if n_compared else None
        results.append(CheckAgreement(name, n_compared, n_agree, rate, n_false_reject, n_false_pass))
    return sorted(results, key=lambda r: (r.agreement_rate is None, r.agreement_rate if r.agreement_rate is not None else 0.0))


@dataclass(frozen=True)
class ThresholdRow:
    k: int
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None


def threshold_sweep(trials: Sequence[ImageTrial], required_check_names: Sequence[str]) -> list[ThresholdRow]:
    """`QCVerdict.passed` is a fixed, unconditional AND over every required
    check (frozen in models.py) -- in production terms, k=1 failing check
    already rejects. This sweep asks the counterfactual "what if the cutoff
    were k failing checks instead of 1?" purely from the labelled data, so a
    different cutoff is a decision backed by a precision/recall table, not a
    default nobody re-examined."""
    per_image: list[tuple[bool, int]] = []
    for t in trials:
        votes = [t.check_majority(name) for name in required_check_names]
        known = [v for v in votes if v is not None]
        if not known:
            continue
        n_failed = sum(1 for v in known if v is False)
        per_image.append((t.label.should_reject, n_failed))

    rows = []
    for k in range(1, len(required_check_names) + 1):
        pairs = [(human, n_failed >= k) for human, n_failed in per_image]
        c = confusion_from_pairs(pairs)
        rows.append(ThresholdRow(k=k, tp=c.tp, fp=c.fp, fn=c.fn, tn=c.tn, precision=c.precision(), recall=c.recall(), f1=c.f1()))
    return rows


# -------------------------------------------------------------- production gate

_MIN_IMAGES = 30
_MAX_CI_WIDTH = 0.25
_MIN_SELF_AGREEMENT = 0.9
_MIN_CHECK_AGREEMENT = 0.7


def evaluate_production_gate(
    decision: DecisionMetrics,
    self_consistency: SelfConsistencySummary,
    per_check: Sequence[CheckAgreement],
) -> tuple[bool, list[str]]:
    """A concrete, evidence-based go/no-go rule (see docs/calibration.md for
    the rationale behind each threshold) -- not a judgment call left to
    whoever happens to be reading the report."""
    reasons = []
    if decision.n_images_used < _MIN_IMAGES:
        reasons.append(f"only {decision.n_images_used} labelled images (need >= {_MIN_IMAGES})")

    for metric_name, estimate in (("precision", decision.precision), ("recall", decision.recall)):
        width = estimate.ci_width
        if width is None or width > _MAX_CI_WIDTH:
            shown = "undetermined" if width is None else f"{width:.2f}"
            reasons.append(f"{metric_name} 95% CI width is {shown} (need <= {_MAX_CI_WIDTH})")

    agreement = self_consistency.mean_self_agreement
    if agreement is None or agreement < _MIN_SELF_AGREEMENT:
        shown = "undetermined (no image was repeated >=2 times)" if agreement is None else f"{agreement:.0%}"
        reasons.append(f"judge self-agreement across repeats is {shown} (need >= {_MIN_SELF_AGREEMENT:.0%})")

    bad_checks = [c.name for c in per_check if c.agreement_rate is not None and c.agreement_rate < _MIN_CHECK_AGREEMENT]
    if bad_checks:
        reasons.append(f"check(s) {bad_checks} agree with humans <{_MIN_CHECK_AGREEMENT:.0%} of the time -- reword them")

    return (len(reasons) == 0, reasons)


# --------------------------------------------------------------------- report


def _metric_dict(e: MetricEstimate) -> dict[str, Any]:
    return {
        "point": e.point,
        "ci_low": e.ci_low,
        "ci_high": e.ci_high,
        "ci_width": e.ci_width,
        "n_valid_resamples": e.n_valid_resamples,
        "n_resamples": e.n_resamples,
    }


def build_report(
    trials: Sequence[ImageTrial],
    rubric: RubricConfig,
    *,
    n_resamples: int = 2000,
    seed: int = 0,
    actual_cost_usd: float | None = None,
    fake_mode: bool = False,
) -> dict[str, Any]:
    required = sorted(rubric.required_check_names)
    decision = decision_metrics(trials, n_resamples=n_resamples, seed=seed)
    self_cons = self_consistency_summary(trials)
    per_check = per_check_agreement(trials, required)
    sweep = threshold_sweep(trials, required)
    gate_ok, gate_reasons = evaluate_production_gate(decision, self_cons, per_check)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "fake_mode": fake_mode,
        "model": rubric.model,
        "n_images": len(trials),
        "actual_cost_usd": actual_cost_usd,
        "decision_metrics": {
            "n_images_used": decision.n_images_used,
            "n_images_skipped_no_valid_trial": decision.n_images_skipped_no_valid_trial,
            "precision": _metric_dict(decision.precision),
            "recall": _metric_dict(decision.recall),
            "f1": _metric_dict(decision.f1),
            "kappa": _metric_dict(decision.kappa),
        },
        "self_consistency": dataclasses.asdict(self_cons),
        "per_check_agreement": [dataclasses.asdict(c) for c in per_check],
        "threshold_sweep": [dataclasses.asdict(r) for r in sweep],
        "per_image": [
            {
                "image": str(t.label.image),
                "human_should_reject": t.label.should_reject,
                "violated_checks": sorted(t.label.violated_checks),
                "judge_majority_decision": t.majority_decision,
                "self_agreement": t.self_agreement,
                "n_valid_trials": len(t.valid_verdicts),
                "n_errors": t.n_errors,
            }
            for t in trials
        ],
        "production_gate": {"ok": gate_ok, "reasons": gate_reasons},
    }


def _fmt(x: float | None, spec: str = ".2f") -> str:
    return "n/a" if x is None else format(x, spec)


def render_summary(report: dict[str, Any]) -> str:
    lines = []
    dm = report["decision_metrics"]
    sc = report["self_consistency"]

    lines.append("CALIBRATION SUMMARY" + (" (FAKE JUDGE -- plumbing test only, not a real measurement)" if report["fake_mode"] else ""))
    lines.append("=" * 70)
    lines.append(
        f"Images labelled: {report['n_images']}  |  used in decision metrics: {dm['n_images_used']}  "
        f"|  skipped (no valid judge trial): {dm['n_images_skipped_no_valid_trial']}"
    )
    cost = report["actual_cost_usd"]
    lines.append(f"Actual spend: {'$' + format(cost, '.2f') if cost is not None else 'n/a'}")
    lines.append("")
    lines.append("HEADLINE -- read the interval WIDTH first. With a small labelled set the width")
    lines.append("is the honest answer; the point estimate on its own overstates confidence.")
    for name in ("precision", "recall", "f1"):
        m = dm[name]
        width = m["ci_width"]
        flag = "" if width is None else ("  <- WIDE, treat as unresolved" if width > _MAX_CI_WIDTH else "  (narrow enough to act on)")
        lines.append(f"  {name:9s}: {_fmt(m['point'])}   95% CI [{_fmt(m['ci_low'])}, {_fmt(m['ci_high'])}]   width={_fmt(width)}{flag}")
    k = dm["kappa"]
    lines.append(f"  {'kappa':9s}: {_fmt(k['point'])}   95% CI [{_fmt(k['ci_low'])}, {_fmt(k['ci_high'])}]   width={_fmt(k['ci_width'])}")
    lines.append("")

    lines.append("SELF-CONSISTENCY vs. HUMAN-DISAGREEMENT")
    n_rep = sc["n_images_with_repeats"]
    if n_rep:
        lines.append(
            f"  The judge disagreed with ITSELF on {sc['n_images_unstable']}/{n_rep} images across repeats "
            f"({_fmt(sc['instability_rate'], '.0%')})."
        )
    else:
        lines.append("  No image had >=2 valid repeat trials -- self-consistency could not be measured.")
    human_disagree_n = sum(
        1 for row in report["per_image"] if row["judge_majority_decision"] is not None and row["judge_majority_decision"] != row["human_should_reject"]
    )
    human_n = dm["n_images_used"]
    if human_n:
        human_rate = human_disagree_n / human_n
        lines.append(f"  The judge disagreed with the HUMAN on {human_disagree_n}/{human_n} images ({human_rate:.0%}).")
        if sc["instability_rate"] is not None and human_rate > 0:
            share = min(1.0, sc["instability_rate"] / human_rate)
            lines.append(
                f"  -> Self-inconsistency alone could account for up to ~{share:.0%} of the human-disagreement "
                "rate. More labels do not fix that share -- it needs rubric/prompt work, not more data."
            )
    lines.append("")

    lines.append("PER-CHECK AGREEMENT (worst first)")
    for c in report["per_check_agreement"]:
        rate = "n/a (no comparable labels)" if c["agreement_rate"] is None else f"{c['agreement_rate']:.0%} ({c['n_agree']}/{c['n_compared']})"
        lines.append(
            f"  {c['name']:24s} {rate:28s} false-reject={c['n_false_reject']} false-pass={c['n_false_pass']}"
        )
    lines.append("")

    lines.append("THRESHOLD SWEEP (how many failing required checks should trigger reject)")
    lines.append(f"  {'k':>2} {'precision':>10} {'recall':>10} {'f1':>10}   tp/fp/fn/tn")
    for row in report["threshold_sweep"]:
        lines.append(
            f"  {row['k']:>2} {_fmt(row['precision']):>10} {_fmt(row['recall']):>10} {_fmt(row['f1']):>10}   "
            f"{row['tp']}/{row['fp']}/{row['fn']}/{row['tn']}"
        )
    lines.append("  (production currently rejects at k=1 -- QCVerdict.passed ANDs every required check)")
    lines.append("")

    gate = report["production_gate"]
    lines.append("PRODUCTION GATE")
    if gate["ok"]:
        lines.append("  OK to proceed toward letting the judge auto-skip assets, per docs/calibration.md's rule.")
    else:
        lines.append("  BLOCKED -- do not enable judge auto-skip in production yet:")
        for reason in gate["reasons"]:
            lines.append(f"    - {reason}")

    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "calibration_results.json"
    md_path = output_dir / "calibration_summary.md"
    json_path.write_text(json.dumps(report, indent=2))
    md_path.write_text(render_summary(report))
    return json_path, md_path


# ------------------------------------------------------------------------ CLI


def _script_fake_judge(
    judge: FakeJudge,
    labels: Sequence[Label],
    *,
    repeats: int,
    required_check_names: Sequence[str],
    seed: int = 0,
    noise: float = 0.1,
) -> None:
    """Fake mode exists to test THIS HARNESS (parsing, metrics, bootstrap,
    report writing, the CLI itself) with no network and no API key -- not to
    produce a meaningful calibration number. It mirrors each label back with
    a small seeded chance of flipping, just enough to exercise the
    self-consistency and disagreement code paths instead of trivially
    collapsing to perfect agreement on every run.
    """
    rng = random.Random(seed)
    fallback_name = required_check_names[0] if required_check_names else "check"
    for label in labels:
        target = next(iter(label.violated_checks), None) or fallback_name
        for _ in range(repeats):
            flipped = rng.random() < noise
            reject_this_trial = label.should_reject != flipped
            if reject_this_trial:
                judge.queue_fail(name=target, evidence="fake-mode scripted failure (harness self-test, not a real verdict)")
            else:
                judge.queue_pass(evidence="fake-mode scripted pass (harness self-test, not a real verdict)")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m comfy_controller.calibration",
        description="Measure the Claude judge against human labels before letting it auto-skip assets.",
    )
    p.add_argument("--labels", required=True, help="CSV or YAML labels file (see docs/calibration.md)")
    p.add_argument("--images-dir", default=None, help="Directory image paths resolve against (default: the labels file's own directory)")
    p.add_argument("--rubric", default="config/rubric.yaml")
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="How many times to re-run the judge per image (>=2 -- needed to separate self-inconsistency from human disagreement)",
    )
    p.add_argument("--bootstrap-samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default="calibration_results")
    p.add_argument("--fake", action="store_true", help="Use FakeJudge -- no network, no spend. For testing the harness itself.")
    p.add_argument("--yes", action="store_true", help="Skip the interactive spend confirmation (real mode only)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.repeats < 2:
        print(
            "error: --repeats must be >= 2 -- with only one run per image there is no way to tell "
            "judge-vs-self instability apart from judge-vs-human disagreement",
            file=sys.stderr,
        )
        return 2

    rubric = RubricConfig.from_yaml(args.rubric)
    labels = load_labels(args.labels, images_dir=args.images_dir)
    if not labels:
        print(f"error: no labels found in {args.labels}", file=sys.stderr)
        return 2

    for warning in validate_labels_against_rubric(labels, rubric):
        print(f"warning: {warning}", file=sys.stderr)

    judge: Any
    if args.fake:
        judge = FakeJudge()
        _script_fake_judge(
            judge, labels, repeats=args.repeats, required_check_names=sorted(rubric.required_check_names), seed=args.seed
        )
    else:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("error: ANTHROPIC_API_KEY is not set (required unless --fake is passed)", file=sys.stderr)
            return 2
        estimate = estimate_cost(rubric, labels, repeats=args.repeats)
        if not confirm_spend(estimate, auto_yes=args.yes):
            print("Aborted -- no API calls made.")
            return 1
        import anthropic  # local import: only needed for a real, paid run

        judge = ClaudeJudge(anthropic.AsyncAnthropic(api_key=api_key), rubric)

    import asyncio

    trials = asyncio.run(run_calibration(judge, labels, repeats=args.repeats))

    actual_cost = judge.cost.total_usd if hasattr(judge, "cost") else 0.0
    report = build_report(
        trials, rubric, n_resamples=args.bootstrap_samples, seed=args.seed, actual_cost_usd=actual_cost, fake_mode=args.fake
    )
    json_path, md_path = write_report(report, Path(args.output_dir))

    print(render_summary(report))
    print(f"\nWrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
