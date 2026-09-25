"""Tests for calibration.py -- the judge-vs-human calibration harness.

No network, no API key, ever: real-mode judging goes through `ClaudeJudge`
only inside `main()`'s non-`--fake` branch, which this file never exercises.
Everything here either drives pure functions directly or runs the harness
end to end against `FakeJudge` / a scripted `JudgePort`-shaped stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from comfy_controller.adapters.claude_judge import RubricCheck, RubricConfig
from comfy_controller.calibration import (
    CheckAgreement,
    ImageTrial,
    Label,
    bootstrap_metric,
    build_arg_parser,
    build_report,
    cohens_kappa_from_pairs,
    confirm_spend,
    confusion_from_pairs,
    decision_metrics,
    estimate_cost,
    evaluate_production_gate,
    load_labels,
    main,
    per_check_agreement,
    run_calibration,
    self_consistency_summary,
    threshold_sweep,
    validate_labels_against_rubric,
    write_report,
)
from comfy_controller.models import Asset, CheckResult, QCVerdict
from comfy_controller.ports import JudgeUnavailable
from comfy_controller.testing.fake_judge import FakeJudge

# ------------------------------------------------------------------- fixtures


def _make_image(path: Path, size: tuple[int, int] = (64, 48)) -> None:
    Image.new("RGB", size, color=(100, 120, 140)).save(path)


CHECKS = (
    RubricCheck(name="subject_present", question="Is the subject visible?", required=True),
    RubricCheck(name="anatomy_correct", question="Any anatomy errors?", required=True),
    RubricCheck(name="aesthetic_appeal", question="Scroll-stopping?", required=False),
)


def make_rubric(**overrides: Any) -> RubricConfig:
    base: dict[str, Any] = dict(
        model="claude-opus-5",
        self_consistency_n=1,
        checks=CHECKS,
        pricing={"claude-opus-5": {"input_per_mtok": 5.0, "output_per_mtok": 25.0}},
    )
    base.update(overrides)
    return RubricConfig(**base)


def verdict(*, subject: bool = True, anatomy: bool = True) -> QCVerdict:
    return QCVerdict(
        checks=[
            CheckResult(name="subject_present", evidence="clearly visible in frame", passed=subject),
            CheckResult(name="anatomy_correct", evidence="hands look correct", passed=anatomy),
        ]
    )


# --------------------------------------------------------------- label loading


def test_load_labels_csv(tmp_path: Path) -> None:
    (tmp_path / "a.png").touch()
    (tmp_path / "b.png").touch()
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("image,label,violated_checks\na.png,pass,\nb.png,fail,anatomy_correct;subject_present\n")

    labels = load_labels(labels_csv)

    assert len(labels) == 2
    assert labels[0].image == tmp_path / "a.png"
    assert labels[0].should_reject is False
    assert labels[0].violated_checks == frozenset()
    assert labels[1].should_reject is True
    assert labels[1].violated_checks == frozenset({"anatomy_correct", "subject_present"})


def test_load_labels_yaml(tmp_path: Path) -> None:
    labels_yaml = tmp_path / "labels.yaml"
    labels_yaml.write_text(
        "- image: a.png\n  label: pass\n- image: b.png\n  label: fail\n  violated_checks: [anatomy_correct]\n"
    )

    labels = load_labels(labels_yaml)

    assert len(labels) == 2
    assert labels[1].violated_checks == frozenset({"anatomy_correct"})


def test_load_labels_images_dir_override(tmp_path: Path) -> None:
    images_dir = tmp_path / "imgs"
    images_dir.mkdir()
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("image,label\na.png,pass\n")

    labels = load_labels(labels_csv, images_dir=images_dir)

    assert labels[0].image == images_dir / "a.png"


def test_load_labels_rejects_bad_label_value(tmp_path: Path) -> None:
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("image,label\na.png,maybe\n")
    with pytest.raises(ValueError, match="unrecognised label"):
        load_labels(labels_csv)


def test_load_labels_unsupported_extension(tmp_path: Path) -> None:
    bad = tmp_path / "labels.txt"
    bad.write_text("nonsense")
    with pytest.raises(ValueError, match="unsupported labels file type"):
        load_labels(bad)


def test_validate_labels_against_rubric_flags_typo_and_missing_file(tmp_path: Path) -> None:
    rubric = make_rubric()
    present = tmp_path / "present.png"
    _make_image(present)
    labels = [
        Label(image=present, should_reject=True, violated_checks=frozenset({"anatomy_korrect"})),
        Label(image=tmp_path / "missing.png", should_reject=False),
    ]

    warnings = validate_labels_against_rubric(labels, rubric)

    assert any("anatomy_korrect" in w for w in warnings)
    assert any("missing.png" in w and "does not exist" in w for w in warnings)


# ------------------------------------------------------------------- metrics


def test_confusion_from_pairs() -> None:
    pairs = [(True, True), (True, False), (False, True), (False, False), (True, True)]
    c = confusion_from_pairs(pairs)
    assert (c.tp, c.fp, c.fn, c.tn) == (2, 1, 1, 1)
    assert c.precision() == pytest.approx(2 / 3)
    assert c.recall() == pytest.approx(2 / 3)
    assert c.f1() == pytest.approx(2 / 3)


def test_confusion_precision_recall_undefined_when_denominator_zero() -> None:
    # Judge never predicts reject -> precision undefined (no predicted positives).
    c = confusion_from_pairs([(True, False), (False, False)])
    assert c.precision() is None
    assert c.recall() == 0.0


def test_cohens_kappa_perfect_agreement() -> None:
    pairs = [(True, True), (False, False), (True, True), (False, False)]
    assert cohens_kappa_from_pairs(pairs) == pytest.approx(1.0)


def test_cohens_kappa_chance_level_near_zero() -> None:
    # Constructed so judge decisions are independent of human labels.
    pairs = [(True, True), (True, False), (False, True), (False, False)] * 5
    kappa = cohens_kappa_from_pairs(pairs)
    assert kappa is not None
    assert abs(kappa) < 0.05


def test_bootstrap_metric_ci_contains_point_estimate() -> None:
    rng_pairs = [(True, True)] * 6 + [(True, False)] * 4 + [(False, True)] * 3 + [(False, False)] * 7
    est = bootstrap_metric(rng_pairs, lambda c: c.precision(), n_resamples=500, seed=1)
    assert est.point is not None
    assert est.ci_low is not None and est.ci_high is not None
    assert est.ci_low <= est.point <= est.ci_high
    assert est.ci_width is not None and est.ci_width >= 0


def test_bootstrap_metric_empty_pairs_returns_no_ci() -> None:
    est = bootstrap_metric([], lambda c: c.precision(), n_resamples=100, seed=0)
    assert est.point is None
    assert est.ci_low is None
    assert est.ci_high is None


def test_bootstrap_ci_is_wide_with_few_images_narrow_with_many() -> None:
    """Point of the whole exercise: a 10-image sample should show a much
    wider CI than a 200-image sample built from the same underlying rate."""
    rng = np.random.default_rng(42)

    def make_pairs(n: int) -> list[tuple[bool, bool]]:
        human = rng.random(n) < 0.5
        judge = human ^ (rng.random(n) < 0.2)  # judge agrees with human ~80% of the time
        return list(zip(human.tolist(), judge.tolist()))

    small = bootstrap_metric(make_pairs(10), lambda c: c.precision(), n_resamples=1000, seed=2)
    large = bootstrap_metric(make_pairs(200), lambda c: c.precision(), n_resamples=1000, seed=2)

    assert small.ci_width is not None and large.ci_width is not None
    assert small.ci_width > large.ci_width


# ------------------------------------------------------------- ImageTrial etc.


def test_image_trial_majority_and_self_agreement() -> None:
    label = Label(image=Path("x.png"), should_reject=False)
    trial = ImageTrial(label=label, outcomes=[verdict(), verdict(), verdict(anatomy=False)])

    # 2/3 runs passed everything -> majority decision is "accept" (not reject).
    assert trial.majority_decision is False
    assert trial.self_agreement == pytest.approx(2 / 3)
    assert trial.check_majority("anatomy_correct") is True  # 2 of 3 said anatomy_correct passed
    assert trial.check_majority("subject_present") is True
    assert trial.check_majority("nonexistent") is None


def test_image_trial_self_agreement_none_with_fewer_than_two_valid_runs() -> None:
    label = Label(image=Path("x.png"), should_reject=False)
    trial = ImageTrial(label=label, outcomes=[verdict()])
    assert trial.self_agreement is None


def test_image_trial_handles_judge_unavailable_outcomes() -> None:
    label = Label(image=Path("x.png"), should_reject=False)
    trial = ImageTrial(label=label, outcomes=[verdict(), JudgeUnavailable("boom"), verdict()])

    assert trial.n_errors == 1
    assert len(trial.valid_verdicts) == 2
    assert trial.majority_decision is False


def test_image_trial_no_valid_verdicts_majority_is_none() -> None:
    label = Label(image=Path("x.png"), should_reject=False)
    trial = ImageTrial(label=label, outcomes=[JudgeUnavailable("boom"), JudgeUnavailable("boom")])
    assert trial.majority_decision is None
    assert trial.self_agreement is None


# ---------------------------------------------------------- decision_metrics


def _trial(should_reject: bool, decisions: list[bool]) -> ImageTrial:
    label = Label(image=Path(f"{'reject' if should_reject else 'accept'}.png"), should_reject=should_reject)
    outcomes = [verdict(anatomy=not d) for d in decisions]
    return ImageTrial(label=label, outcomes=outcomes)


def test_decision_metrics_counts_and_skips_no_valid_trial() -> None:
    trials = [
        _trial(True, [True, True, True]),  # correctly caught
        _trial(True, [False, False]),  # missed -> FN
        _trial(False, [True]),  # false alarm -> FP
        _trial(False, [False, False]),  # correctly accepted
        ImageTrial(label=Label(image=Path("err.png"), should_reject=True), outcomes=[JudgeUnavailable("x")]),
    ]

    dm = decision_metrics(trials, n_resamples=200, seed=0)

    assert dm.n_images_used == 4
    assert dm.n_images_skipped_no_valid_trial == 1
    assert dm.precision.point == pytest.approx(0.5)
    assert dm.recall.point == pytest.approx(0.5)


# -------------------------------------------------------- self-consistency


def test_self_consistency_summary_flags_unstable_images() -> None:
    trials = [
        _trial(True, [True, True, True]),  # stable
        _trial(False, [False, True]),  # unstable (split)
        _trial(False, [False]),  # only one run -> excluded (None self_agreement)
    ]

    summary = self_consistency_summary(trials)

    assert summary.n_images_with_repeats == 2
    assert summary.n_images_unstable == 1
    assert summary.mean_self_agreement == pytest.approx((1.0 + 0.5) / 2)


# ------------------------------------------------------------ per-check / sweep


def test_per_check_agreement_flags_the_bad_check() -> None:
    # anatomy_correct: judge always disagrees with the human ground truth.
    # subject_present: judge always agrees.
    trials = []
    for i in range(6):
        should_reject = i % 2 == 0
        violated = frozenset({"anatomy_correct"}) if should_reject else frozenset()
        label = Label(image=Path(f"img{i}.png"), should_reject=should_reject, violated_checks=violated)
        # anatomy_correct always wrong relative to ground truth; subject_present always right.
        expected_anatomy_pass = not (should_reject and "anatomy_correct" in violated)
        v = QCVerdict(
            checks=[
                CheckResult(name="subject_present", evidence="clearly visible", passed=True),
                CheckResult(name="anatomy_correct", evidence="looks fine", passed=not expected_anatomy_pass),
            ]
        )
        trials.append(ImageTrial(label=label, outcomes=[v, v, v]))

    results = per_check_agreement(trials, ["subject_present", "anatomy_correct"])
    by_name = {r.name: r for r in results}

    assert by_name["anatomy_correct"].agreement_rate == pytest.approx(0.0)
    assert by_name["subject_present"].agreement_rate == pytest.approx(1.0)
    # Sorted worst-first.
    assert [r.name for r in results][0] == "anatomy_correct"


def test_per_check_agreement_skips_unknown_ground_truth() -> None:
    # fail label with no violated_checks specified -> ground truth for any
    # individual check is unknown, must not be compared.
    label = Label(image=Path("x.png"), should_reject=True, violated_checks=frozenset())
    v = QCVerdict(checks=[CheckResult(name="subject_present", evidence="ok", passed=False)])
    trials = [ImageTrial(label=label, outcomes=[v])]

    results = per_check_agreement(trials, ["subject_present"])
    assert results[0].n_compared == 0
    assert results[0].agreement_rate is None


def test_threshold_sweep_recall_is_monotonically_non_increasing() -> None:
    trials = [
        _trial(True, [True, True, True]),
        _trial(True, [True, False, False]),
        _trial(False, [False, False, False]),
        _trial(False, [True, False, False]),
    ]
    # Give each trial a second, independent failing check on some images so a
    # 2-required-check sweep is meaningful.
    required = ["subject_present", "anatomy_correct"]
    rows = threshold_sweep(trials, required)

    assert [r.k for r in rows] == [1, 2]
    recalls = [r.recall for r in rows if r.recall is not None]
    assert recalls == sorted(recalls, reverse=True)


# --------------------------------------------------------------- cost estimate


def test_estimate_cost_scales_with_repeats(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    _make_image(img, size=(512, 512))
    labels = [Label(image=img, should_reject=False)]
    rubric = make_rubric(self_consistency_n=1)

    est_1 = estimate_cost(rubric, labels, repeats=2)
    est_2 = estimate_cost(rubric, labels, repeats=4)

    assert est_2.total_judge_calls == 2 * est_1.total_judge_calls
    assert est_2.estimated_usd > est_1.estimated_usd


def test_estimate_cost_zero_labels() -> None:
    rubric = make_rubric()
    est = estimate_cost(rubric, [], repeats=3)
    assert est.total_judge_calls == 0
    assert est.estimated_usd == 0.0


def test_estimate_cost_handles_missing_image_file(tmp_path: Path) -> None:
    labels = [Label(image=tmp_path / "does_not_exist.png", should_reject=False)]
    rubric = make_rubric()
    est = estimate_cost(rubric, labels, repeats=2)
    assert est.estimated_usd > 0  # falls back to a constant estimate, doesn't crash


def test_confirm_spend_auto_yes_skips_prompt(tmp_path: Path) -> None:
    rubric = make_rubric()
    est = estimate_cost(rubric, [Label(image=tmp_path / "x.png", should_reject=False)], repeats=2)

    def boom(_prompt: str) -> str:
        raise AssertionError("must not prompt when auto_yes=True")

    assert confirm_spend(est, auto_yes=True, input_fn=boom) is True


def test_confirm_spend_respects_no(tmp_path: Path) -> None:
    rubric = make_rubric()
    est = estimate_cost(rubric, [Label(image=tmp_path / "x.png", should_reject=False)], repeats=2)
    assert confirm_spend(est, auto_yes=False, input_fn=lambda _p: "n") is False
    assert confirm_spend(est, auto_yes=False, input_fn=lambda _p: "y") is True


# --------------------------------------------------------- production gate


def test_evaluate_production_gate_blocks_on_too_few_images() -> None:
    trials = [_trial(True, [True, True]), _trial(False, [False, False])]
    dm = decision_metrics(trials, n_resamples=100, seed=0)
    sc = self_consistency_summary(trials)
    ok, reasons = evaluate_production_gate(dm, sc, [])
    assert ok is False
    assert any("labelled images" in r for r in reasons)


def test_evaluate_production_gate_blocks_on_bad_check() -> None:
    dm = decision_metrics([], n_resamples=10, seed=0)
    sc = self_consistency_summary([])
    bad_check = CheckAgreement(name="lighting_quality", n_compared=10, n_agree=3, agreement_rate=0.3, n_false_reject=5, n_false_pass=2)
    ok, reasons = evaluate_production_gate(dm, sc, [bad_check])
    assert ok is False
    assert any("lighting_quality" in r for r in reasons)


# --------------------------------------------------------------- end to end


async def _run_end_to_end(tmp_path: Path, *, repeats: int = 3):
    images = []
    for name in ("good1", "good2", "bad1", "bad2"):
        p = tmp_path / f"{name}.png"
        _make_image(p)
        images.append(p)

    labels = [
        Label(image=images[0], should_reject=False),
        Label(image=images[1], should_reject=False),
        Label(image=images[2], should_reject=True, violated_checks=frozenset({"anatomy_correct"})),
        Label(image=images[3], should_reject=True, violated_checks=frozenset({"subject_present"})),
    ]

    judge = FakeJudge()
    for label in labels:
        for _ in range(repeats):
            if label.should_reject:
                name = next(iter(label.violated_checks))
                judge.queue_fail(name=name, evidence="scripted failure for the harness test")
            else:
                judge.queue_verdict(verdict())

    rubric = make_rubric()
    trials = await run_calibration(judge, labels, repeats=repeats)
    report = build_report(trials, rubric, n_resamples=200, seed=0, actual_cost_usd=0.0, fake_mode=True)
    return trials, report


async def test_run_calibration_and_build_report_end_to_end(tmp_path: Path) -> None:
    trials, report = await _run_end_to_end(tmp_path)

    assert len(trials) == 4
    assert all(t.self_agreement == 1.0 for t in trials)  # every repeat scripted identically -> perfectly stable

    dm = report["decision_metrics"]
    assert dm["n_images_used"] == 4
    assert dm["precision"]["point"] == pytest.approx(1.0)
    assert dm["recall"]["point"] == pytest.approx(1.0)
    assert report["self_consistency"]["mean_self_agreement"] == pytest.approx(1.0)
    assert report["production_gate"]["ok"] is False  # far too few images regardless of perfect agreement
    assert any("labelled images" in r for r in report["production_gate"]["reasons"])


async def test_write_report_creates_json_and_markdown(tmp_path: Path) -> None:
    _trials, report = await _run_end_to_end(tmp_path)
    out_dir = tmp_path / "out"

    json_path, md_path = write_report(report, out_dir)

    assert json_path.exists()
    assert md_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["n_images"] == 4
    summary_text = md_path.read_text()
    assert "CALIBRATION SUMMARY" in summary_text
    assert "PRODUCTION GATE" in summary_text
    assert "THRESHOLD SWEEP" in summary_text


# ------------------------------------------------------------------------ CLI


def test_build_arg_parser_requires_repeats_at_least_two_is_enforced_in_main(tmp_path: Path) -> None:
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("image,label\na.png,pass\n")
    _make_image(tmp_path / "a.png")

    rc = main(["--labels", str(labels_csv), "--fake", "--repeats", "1", "--output-dir", str(tmp_path / "out")])
    assert rc == 2


def test_main_fake_mode_end_to_end(tmp_path: Path) -> None:
    for name in ("a", "b", "c"):
        _make_image(tmp_path / f"{name}.png")
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("image,label,violated_checks\na.png,pass,\nb.png,fail,anatomy_correct\nc.png,pass,\n")

    out_dir = tmp_path / "out"
    rc = main(
        [
            "--labels",
            str(labels_csv),
            "--rubric",
            str(Path(__file__).resolve().parents[1] / "config" / "rubric.yaml"),
            "--fake",
            "--repeats",
            "2",
            "--bootstrap-samples",
            "50",
            "--output-dir",
            str(out_dir),
        ]
    )

    assert rc == 0
    assert (out_dir / "calibration_results.json").exists()
    assert (out_dir / "calibration_summary.md").exists()


def test_main_real_mode_without_api_key_errors_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _make_image(tmp_path / "a.png")
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("image,label\na.png,pass\n")

    rc = main(
        [
            "--labels",
            str(labels_csv),
            "--rubric",
            str(Path(__file__).resolve().parents[1] / "config" / "rubric.yaml"),
            "--repeats",
            "2",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2  # must fail before ever touching the network


def test_main_no_labels_found_errors_cleanly(tmp_path: Path) -> None:
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text("image,label\n")  # header only, no rows
    rc = main(["--labels", str(labels_csv), "--fake", "--repeats", "2", "--output-dir", str(tmp_path / "out")])
    assert rc == 2


def test_build_arg_parser_defaults() -> None:
    args = build_arg_parser().parse_args(["--labels", "labels.csv"])
    assert args.repeats == 3
    assert args.bootstrap_samples == 2000
    assert args.fake is False
    assert args.yes is False
