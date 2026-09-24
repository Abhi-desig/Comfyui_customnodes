"""Tests for `FrameConsistencyChecker` (adapters/frame_consistency.py).

Every frame is synthesised with numpy -- no image fixtures on disk. Frames
are passed straight in as numpy arrays; `FrameConsistencyChecker` accepts
those directly (as well as `Path`s, for the runner's real on-disk frames).

No torch/clip/lpips is installed in this environment (by design -- they're
an optional extra, see `pyproject.toml`), so every test here exercises the
pure numpy/Pillow fallback path. Anything that would need the real
CLIP/LPIPS backend is skipped, not failed, via `pytest.importorskip`.
"""

from __future__ import annotations

import numpy as np
import pytest

from comfy_controller.adapters.frame_consistency import (
    ConsistencyConfig,
    FrameConsistencyChecker,
    _region_label,
    _try_import_clip_backend,
    _try_import_lpips_backend,
    cumulative_drift,
    histogram_correlation,
    ssim_global,
    tile_grid_distances,
)


def _frame(
    *,
    brightness: float = 130.0,
    noise_std: float = 4.0,
    size: int = 32,
    seed: int = 0,
    patch: tuple[tuple[int, int, int, int], float] | None = None,
) -> np.ndarray:
    """A flat-colour frame plus noise, optionally with a small rectangular
    patch stamped in -- the numpy stand-in for "a logo appeared/changed"."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), brightness, dtype=np.float64)
    noise = rng.normal(0, noise_std, size=(size, size, 3))
    arr = base + noise
    if patch is not None:
        (r0, r1, c0, c1), value = patch
        arr[r0:r1, c0:c1] = value
    return np.clip(arr, 0, 255).astype(np.uint8)


@pytest.fixture()
def checker() -> FrameConsistencyChecker:
    return FrameConsistencyChecker()


# --------------------------------------------------------------- backend selection


def test_torch_is_not_installed_in_this_environment():
    """Sanity check for every other test's premise: torch/clip/lpips are an
    optional extra (pyproject.toml's `consistency` group), not a core
    dependency, and this suite must pass with none of them installed."""
    assert _try_import_clip_backend() is None
    assert _try_import_lpips_backend() is None


def test_fallback_backend_selected_and_exposed_when_torch_absent(checker: FrameConsistencyChecker):
    """`FrameConsistencyChecker` must degrade explicitly -- never silently
    give a weaker answer while looking like the strong one -- so the backend
    actually used is a first-class, inspectable property."""
    assert checker.embedding_backend == "histogram_fallback"
    assert checker.perceptual_backend == "ssim_fallback"


@pytest.mark.skipif(_try_import_clip_backend() is None, reason="torch/clip not installed")
def test_real_clip_backend_selected_when_torch_present():  # pragma: no cover - not exercised in CI
    """Skipped (not failed) in this environment. Documents the expectation
    for a machine with the `consistency` extra installed."""
    checker = FrameConsistencyChecker()
    assert checker.embedding_backend == "clip"


@pytest.mark.skipif(_try_import_lpips_backend() is None, reason="torch/lpips not installed")
def test_real_lpips_backend_selected_when_torch_present():  # pragma: no cover - not exercised in CI
    checker = FrameConsistencyChecker()
    assert checker.perceptual_backend == "lpips"


# ---------------------------------------------------------------- identical frames


def test_identical_frames_are_consistent(checker: FrameConsistencyChecker):
    frame = _frame(seed=1)
    pair = checker.check_pair(frame, frame.copy(), index_a=0, index_b=1)
    assert pair.passed
    assert pair.tripped_signals == ()
    assert pair.changed_regions == ()


def test_identical_sequence_is_consistent(checker: FrameConsistencyChecker):
    frames = [_frame(seed=1) for _ in range(6)]
    report = checker.check_sequence(frames)
    assert report.passed
    assert report.failed_pairs == ()
    assert not report.brightness_drift.tripped
    assert not report.color_drift.tripped


# ------------------------------------------------------- headline: localised patch


def test_small_moved_patch_caught_by_tile_signal_not_global_metrics(checker: FrameConsistencyChecker):
    """The headline case: a small bright patch (simulating a logo appearing/
    moving) covering ~1.5% of the frame. Global metrics -- whole-frame
    histogram correlation (embedding-similarity fallback) and SSIM
    (perceptual-distance fallback) -- are diluted by the otherwise-identical
    frame and stay comfortably inside their thresholds. Only the per-tile
    signal, which looks at one small region at a time, catches it.
    """
    frame_a = _frame(seed=1)
    frame_b = _frame(seed=1, patch=((0, 4, 0, 4), 255.0))

    pair = checker.check_pair(frame_a, frame_b, index_a=3, index_b=4)

    # The whole-frame metrics alone would have missed this -- assert that
    # explicitly, not just the aggregate outcome.
    assert not pair.embedding_similarity.tripped
    assert pair.embedding_similarity.value > pair.embedding_similarity.threshold
    assert not pair.perceptual_distance.tripped
    assert pair.perceptual_distance.value < pair.perceptual_distance.threshold

    # The tile signal is what actually catches it, and localises it.
    assert pair.tile_signal.tripped
    assert pair.tripped_signals == ("tile",)
    assert pair.changed_regions == ("top-left",)

    # Because at least one signal tripped, the pair as a whole must fail even
    # though two of its three signals individually passed.
    assert not pair.passed


def test_tile_grid_distances_localises_a_patch_to_one_tile():
    frame_a = _frame(seed=7, size=32)
    frame_b = _frame(seed=7, size=32, patch=((0, 4, 0, 4), 255.0))
    tiles = tile_grid_distances(
        frame_a.astype(np.float64), frame_b.astype(np.float64), grid=(4, 4), distance_max=40.0
    )
    tripped = [t for t in tiles if t.tripped]
    assert len(tripped) == 1
    assert (tripped[0].row, tripped[0].col) == (0, 0)
    assert tripped[0].label == "top-left"


# ---------------------------------------------------------------- global flicker


def test_global_brightness_shift_flagged_as_flicker(checker: FrameConsistencyChecker):
    """A uniform whole-frame brightness shift (the opposite of the localised
    patch above) is exactly what the perceptual-distance signal exists to
    catch -- unlike the patch case, there's no single region to blame, so it
    trips the *global* signal rather than the tile signal alone."""
    frame_a = _frame(brightness=50.0, noise_std=4.0, seed=2)
    frame_b = np.clip(frame_a.astype(np.float64) + 90.0, 0, 255).astype(np.uint8)

    pair = checker.check_pair(frame_a, frame_b, index_a=0, index_b=1)

    assert pair.perceptual_distance.tripped
    assert not pair.passed


def test_ssim_and_histogram_correlation_are_near_perfect_for_identical_arrays():
    frame = _frame(seed=9).astype(np.float64)
    assert histogram_correlation(frame, frame) == pytest.approx(1.0, abs=1e-9)
    assert ssim_global(frame, frame) == pytest.approx(1.0, abs=1e-9)


# ------------------------------------------------------------- cumulative drift


def test_slow_cumulative_drift_untripped_pairwise_but_flagged_overall():
    """10 frames, each 2 brightness units warmer than the last. Every
    adjacent pair is far too small a step to trip any pairwise signal, but
    the cumulative distance from frame 0 to frame 9 (~18 units) exceeds a
    (tightened, for this test) cumulative-drift threshold. This is exactly
    the case no pairwise-only check (including the design doc's own N-1/N+1
    compare) can ever catch."""
    config = ConsistencyConfig(cumulative_brightness_drift_max=15.0, cumulative_color_drift_max=15.0)
    checker = FrameConsistencyChecker(config)

    frames = [
        _frame(brightness=100.0 + 2.0 * i, noise_std=5.0, size=64, seed=100 + i) for i in range(10)
    ]
    report = checker.check_sequence(frames)

    assert all(p.passed for p in report.pairs), [
        (p.frame_a, p.frame_b, p.tripped_signals) for p in report.pairs if not p.passed
    ]
    assert report.brightness_drift.tripped
    assert not report.passed


def test_cumulative_drift_helper_measures_against_first_frame_not_previous():
    arrays = [
        np.full((4, 4, 3), 100.0),
        np.full((4, 4, 3), 101.0),
        np.full((4, 4, 3), 102.0),
        np.full((4, 4, 3), 130.0),  # big jump relative to frame 0, not frame 2
    ]
    points, max_brightness_drift, max_color_drift = cumulative_drift(arrays)
    assert len(points) == 4
    assert max_brightness_drift == pytest.approx(30.0)
    assert max_color_drift == pytest.approx(30.0 * (3**0.5))


# --------------------------------------------------------------------- region labels


@pytest.mark.parametrize(
    "row,col,expected",
    [
        (0, 0, "top-left"),
        (0, 3, "top-right"),
        (3, 0, "bottom-left"),
        (3, 3, "bottom-right"),
        (1, 1, "middle-center"),
    ],
)
def test_region_label_buckets_grid_position(row, col, expected):
    assert _region_label(row, col, rows=4, cols=4) == expected


# --------------------------------------------------------------------------- config


def test_consistency_config_defaults():
    cfg = ConsistencyConfig()
    assert cfg.embedding_similarity_min == 0.92
    assert cfg.tile_grid == (4, 4)


def test_consistency_config_from_dict_overrides_and_tuples_the_grid():
    cfg = ConsistencyConfig.from_dict({"tile_grid": [2, 3], "tile_distance_max": 10.0})
    assert cfg.tile_grid == (2, 3)
    assert cfg.tile_distance_max == 10.0
    # Unspecified keys keep module defaults.
    assert cfg.embedding_similarity_min == 0.92


def test_consistency_config_from_dict_none_is_all_defaults():
    assert ConsistencyConfig.from_dict(None) == ConsistencyConfig()


def test_consistency_config_from_yaml_reads_rubric_yaml_section(tmp_path):
    yaml_path = tmp_path / "rubric.yaml"
    yaml_path.write_text(
        "frame_consistency:\n"
        "  embedding_similarity_min: 0.5\n"
        "  tile_grid: [2, 2]\n"
        "checks: []\n"
    )
    cfg = ConsistencyConfig.from_yaml(yaml_path)
    assert cfg.embedding_similarity_min == 0.5
    assert cfg.tile_grid == (2, 2)
    # Keys the test file doesn't set still fall back to module defaults.
    assert cfg.tile_distance_max == 40.0


def test_consistency_config_from_yaml_missing_section_is_all_defaults(tmp_path):
    yaml_path = tmp_path / "rubric.yaml"
    yaml_path.write_text("checks: []\n")
    assert ConsistencyConfig.from_yaml(yaml_path) == ConsistencyConfig()


# ------------------------------------------------------------------------- mismatched sizes


def test_mismatched_frame_sizes_are_resized_rather_than_crashing(checker: FrameConsistencyChecker):
    frame_a = _frame(seed=1, size=32)
    frame_b = _frame(seed=1, size=48)
    pair = checker.check_pair(frame_a, frame_b, index_a=0, index_b=1)
    assert isinstance(pair.passed, bool)  # didn't raise


# ------------------------------------------------------------------------- errors


def test_check_sequence_requires_at_least_two_frames(checker: FrameConsistencyChecker):
    with pytest.raises(ValueError):
        checker.check_sequence([_frame(seed=1)])
