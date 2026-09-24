"""Frame-to-frame consistency checker for SEQUENCE assets.

The design doc requires sequences to be checked for "consistency across
frames (product, colours, no flicker)", and the repair loop
(`core/sequence.py`) says: regenerate frame N using N-1 as reference, then
**QC the new frame N against both N-1 and N+1**. `JudgePort.judge()` (frozen,
`ports.py`) only ever judges one frame in isolation -- it has no notion of
"compare to a neighbour" -- so "the product changed between frame 4 and 5"
goes undetected today. This module fills that gap with cheap, deterministic,
non-VLM signals cheap enough to run on every adjacent frame pair.

Four signals, in priority order (see `ConsistencyConfig` for thresholds):

  1. **Embedding similarity** between adjacent frames (CLIP cosine similarity
     when torch+CLIP are installed, vendored the way VBench computes
     `subject_consistency`/`background_consistency` -- NOT a dependency on
     VBench itself, which is a whole benchmark harness with dataset loaders).
     This is the signal that catches *localised* content change: the product
     or logo changing while everything else in the frame stays put.

  2. **Perceptual distance** (LPIPS when torch+lpips are installed) as a
     flicker / global-shift cross-check. IMPORTANT CAVEAT, by design: LPIPS
     (and its fallback, global SSIM, below) spatially averages over the
     whole frame. A small logo change gets diluted by an otherwise-static
     frame and barely moves this number -- it is a flicker/global-shift
     detector, not a localised-change detector. That is exactly why signal 1
     (and signal 3) exist, and it is why this signal alone is insufficient.

  3. **Per-tile comparison**: a grid of per-tile mean-color deltas that
     localises *where* a frame changed, so a report can say "top-left
     changed" instead of just "something changed". numpy/Pillow only, no
     extra dependency, and always run regardless of which backend signals
     1/2 landed on -- this is what keeps a localised change catchable even
     with zero heavy deps installed.

  4. **Cumulative colour/brightness drift**: brightness and mean-colour are
     tracked against the FIRST frame (not just the previous one) across the
     whole sequence, so a slow drift that no single adjacent pair would flag
     (each step is tiny) still trips once the cumulative distance from the
     start gets too large.

Dependency policy (torch/clip/lpips are heavy and NOT core dependencies --
see `pyproject.toml`'s `consistency` extra):
  - Real CLIP/LPIPS backends are imported lazily, only inside the functions
    that need them, and only attempted once per process (see the module-level
    singleton caches below).
  - A pure numpy/Pillow fallback (`histogram_correlation` / `ssim_global`)
    always works with nothing extra installed. `FrameConsistencyChecker`
    picks a backend at construction time and logs which one -- callers can
    always tell a real CLIP/LPIPS score from a fallback proxy via
    `SignalScore.backend`, and results never claim more precision than the
    active backend actually provides.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, Union

import numpy as np
import structlog
from PIL import Image

log = structlog.get_logger(__name__)

FrameSource = Union[str, Path, "np.ndarray", Image.Image]

# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class ConsistencyConfig:
    # Signal 1: CLIP cosine similarity (or histogram-correlation fallback).
    # Below this floor between adjacent frames -> localised content change.
    embedding_similarity_min: float = 0.92
    # Signal 2: LPIPS (or SSIM-fallback distance = 1 - ssim). Above this
    # ceiling -> flicker / whole-frame shift.
    perceptual_distance_max: float = 0.30
    # Signal 3: grid shape and per-tile mean-color-delta ceiling.
    tile_grid: tuple[int, int] = (4, 4)
    tile_distance_max: float = 40.0
    # Signal 4: cumulative drift from the sequence's first frame.
    cumulative_brightness_drift_max: float = 20.0
    cumulative_color_drift_max: float = 25.0
    # Histogram fallback resolution (per channel).
    histogram_bins: int = 32

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ConsistencyConfig":
        raw = raw or {}
        grid = raw.get("tile_grid", (4, 4))
        return cls(
            embedding_similarity_min=float(raw.get("embedding_similarity_min", 0.92)),
            perceptual_distance_max=float(raw.get("perceptual_distance_max", 0.30)),
            tile_grid=(int(grid[0]), int(grid[1])),
            tile_distance_max=float(raw.get("tile_distance_max", 40.0)),
            cumulative_brightness_drift_max=float(raw.get("cumulative_brightness_drift_max", 20.0)),
            cumulative_color_drift_max=float(raw.get("cumulative_color_drift_max", 25.0)),
            histogram_bins=int(raw.get("histogram_bins", 32)),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConsistencyConfig":
        """Reads the `frame_consistency:` section of a rubric-style YAML file
        (see `config/rubric.yaml`). Falls back to module defaults for any key
        the file doesn't set, and to an all-defaults config if the file has
        no `frame_consistency` section at all."""
        import yaml

        raw = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(raw.get("frame_consistency"))


# ------------------------------------------------------------------- results


@dataclass(frozen=True)
class SignalScore:
    """One measured signal: its raw value, the threshold it was judged
    against, whether it tripped, and which backend produced it (so a caller
    can always tell a real CLIP/LPIPS score apart from a fallback proxy)."""

    name: str
    value: float
    threshold: float
    tripped: bool
    backend: str


@dataclass(frozen=True)
class TileScore:
    row: int
    col: int
    distance: float
    tripped: bool
    label: str  # human region name, e.g. "top-left"


@dataclass(frozen=True)
class PairVerdict:
    """Consistency between two specific frames (usually, but not
    necessarily, adjacent)."""

    frame_a: int
    frame_b: int
    embedding_similarity: SignalScore
    perceptual_distance: SignalScore
    tile_signal: SignalScore
    tiles: tuple[TileScore, ...]
    passed: bool

    @property
    def tripped_signals(self) -> tuple[str, ...]:
        return tuple(
            s.name
            for s in (self.embedding_similarity, self.perceptual_distance, self.tile_signal)
            if s.tripped
        )

    @property
    def changed_regions(self) -> tuple[str, ...]:
        """Which tile regions localise the change, e.g. ("top-left",). Empty
        when the tile signal didn't trip -- a whole-frame/flicker failure has
        no single region to point at."""
        return tuple(t.label for t in self.tiles if t.tripped)


@dataclass(frozen=True)
class SequenceConsistencyReport:
    pairs: tuple[PairVerdict, ...]
    brightness_drift: SignalScore
    color_drift: SignalScore
    embedding_backend: str
    perceptual_backend: str
    passed: bool

    @property
    def failed_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple((p.frame_a, p.frame_b) for p in self.pairs if not p.passed)


# --------------------------------------------------------------- frame loading


def _as_array_and_image(source: FrameSource) -> tuple[np.ndarray, Image.Image]:
    """Accepts a path (real sequence frames, on disk) or a bare numpy array /
    PIL Image (tests -- "synthesise frames with numpy", no fixtures on disk).
    """
    if isinstance(source, (str, Path)):
        with Image.open(source) as img:
            img.load()
            image = img.convert("RGB").copy()
    elif isinstance(source, Image.Image):
        image = source.convert("RGB")
    elif isinstance(source, np.ndarray):
        image = Image.fromarray(np.clip(source, 0, 255).astype(np.uint8), "RGB")
    else:
        raise TypeError(f"unsupported frame source: {type(source)!r}")
    return np.asarray(image, dtype=np.float64), image


def _match_size(
    arr_a: np.ndarray, arr_b: np.ndarray, img_b: Image.Image
) -> tuple[np.ndarray, Image.Image]:
    """Sequence frames should always share a resolution, but a mismatched
    regenerated frame shouldn't crash the checker -- resize `b` onto `a`."""
    if arr_a.shape[:2] == arr_b.shape[:2]:
        return arr_b, img_b
    resized = img_b.resize((arr_a.shape[1], arr_a.shape[0]))
    return np.asarray(resized, dtype=np.float64), resized


def _grayscale(arr: np.ndarray) -> np.ndarray:
    # Rec. 601 luma weights -- the same convention Pillow's "L" convert uses.
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def _region_label(row: int, col: int, rows: int, cols: int) -> str:
    def bucket(i: int, n: int, labels: tuple[str, str, str]) -> str:
        if n <= 1:
            return labels[1]
        frac = i / (n - 1)
        idx = min(len(labels) - 1, int(frac * len(labels)))
        return labels[idx]

    vertical = bucket(row, rows, ("top", "middle", "bottom"))
    horizontal = bucket(col, cols, ("left", "center", "right"))
    return f"{vertical}-{horizontal}"


# ------------------------------------------------------- signal 1: embedding


def _try_import_clip_backend() -> tuple[str, Any, Any] | None:
    """Lazy, best-effort import. Supports either `open_clip` (PyPI-installable
    reimplementation, what `pyproject.toml`'s `consistency` extra pulls in)
    or OpenAI's original `clip` package if that happens to be present
    instead. Returns None (never raises) when neither is importable, which is
    the expected/normal state outside the optional extra -- torch/clip are
    NOT core dependencies."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    try:
        import open_clip

        return ("open_clip", torch, open_clip)
    except ImportError:
        pass
    try:
        import clip

        return ("clip", torch, clip)
    except ImportError:
        return None


_clip_singleton: dict[str, Any] = {}


def _get_clip_model() -> dict[str, Any] | None:
    if "unavailable" in _clip_singleton:
        return None
    if "model" in _clip_singleton:
        return _clip_singleton
    modules = _try_import_clip_backend()
    if modules is None:
        _clip_singleton["unavailable"] = True
        return None
    flavor, torch, clip_mod = modules
    if flavor == "open_clip":
        model, _, preprocess = clip_mod.create_model_and_transforms("ViT-B-32", pretrained="openai")
    else:
        model, preprocess = clip_mod.load("ViT-B/32", device="cpu")
    model.eval()
    _clip_singleton.update(torch=torch, model=model, preprocess=preprocess, flavor=flavor)
    return _clip_singleton


def clip_cosine_similarity(image_a: Image.Image, image_b: Image.Image) -> float:
    """Real CLIP embedding cosine similarity. Raises RuntimeError (caught by
    the checker, never by a caller expecting a score) when torch/CLIP aren't
    installed -- callers should use `histogram_correlation` instead."""
    bundle = _get_clip_model()
    if bundle is None:
        raise RuntimeError("clip backend unavailable (torch/open_clip/clip not installed)")
    torch = bundle["torch"]
    model, preprocess = bundle["model"], bundle["preprocess"]
    with torch.no_grad():
        tensor_a = preprocess(image_a).unsqueeze(0)
        tensor_b = preprocess(image_b).unsqueeze(0)
        emb_a = model.encode_image(tensor_a).float()
        emb_b = model.encode_image(tensor_b).float()
        emb_a = emb_a / emb_a.norm(dim=-1, keepdim=True)
        emb_b = emb_b / emb_b.norm(dim=-1, keepdim=True)
        return float((emb_a @ emb_b.T).item())


def histogram_correlation(arr_a: np.ndarray, arr_b: np.ndarray, *, bins: int = 32) -> float:
    """Global colour-histogram Pearson correlation across the whole frame --
    the numpy/Pillow fallback for embedding similarity when CLIP is
    unavailable.

    Like LPIPS, this is a *global* statistic: it is diluted by whatever
    fraction of the frame changed, so a small localised change (e.g. a logo
    swap covering a few percent of the frame) barely moves it. That is
    precisely why signal 3 (the tile grid) exists and always runs regardless
    of which embedding backend is active.
    """

    def hist(arr: np.ndarray) -> np.ndarray:
        channels = [np.histogram(arr[..., c], bins=bins, range=(0, 255))[0].astype(np.float64) for c in range(3)]
        return np.concatenate(channels)

    hist_a, hist_b = hist(arr_a), hist(arr_b)
    hist_a = hist_a / (hist_a.sum() or 1.0)
    hist_b = hist_b / (hist_b.sum() or 1.0)
    a_centered = hist_a - hist_a.mean()
    b_centered = hist_b - hist_b.mean()
    denom = np.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    if denom == 0:
        return 1.0 if np.allclose(hist_a, hist_b) else 0.0
    return float((a_centered * b_centered).sum() / denom)


# --------------------------------------------------------- signal 2: perceptual


def _try_import_lpips_backend() -> tuple[Any, Any] | None:
    try:
        import torch
        import lpips
    except ImportError:
        return None
    return torch, lpips


_lpips_singleton: dict[str, Any] = {}


def _get_lpips_model() -> dict[str, Any] | None:
    if "unavailable" in _lpips_singleton:
        return None
    if "model" in _lpips_singleton:
        return _lpips_singleton
    modules = _try_import_lpips_backend()
    if modules is None:
        _lpips_singleton["unavailable"] = True
        return None
    torch, lpips_mod = modules
    model = lpips_mod.LPIPS(net="alex")
    model.eval()
    _lpips_singleton.update(torch=torch, model=model)
    return _lpips_singleton


def lpips_distance(image_a: Image.Image, image_b: Image.Image) -> float:
    """Real LPIPS perceptual distance. Raises RuntimeError when torch/lpips
    aren't installed -- callers should use `ssim_global` instead."""
    bundle = _get_lpips_model()
    if bundle is None:
        raise RuntimeError("lpips backend unavailable (torch/lpips not installed)")
    torch = bundle["torch"]
    model = bundle["model"]

    def to_tensor(image: Image.Image) -> Any:
        arr = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        distance = model(to_tensor(image_a), to_tensor(image_b))
    return float(distance.item())


def ssim_global(arr_a: np.ndarray, arr_b: np.ndarray, *, window: int = 8) -> float:
    """Hand-rolled mean structural similarity, i.e. the standard windowed
    SSIM averaged over non-overlapping blocks (no scipy/scikit-image
    dependency) -- the fallback perceptual-distance cross-check when
    LPIPS/torch are unavailable.

    This is "spatially averaged" in the literal sense the module docstring's
    caveat describes: each window gets its own SSIM score and the final
    number is the mean across all of them, exactly like LPIPS averages a
    per-location deep-feature distance map. A single small patch therefore
    only tanks the one or two windows it falls in -- diluted by every other
    window that didn't change -- while a whole-frame shift moves every
    window's score together. A single whole-image (unwindowed) statistic
    would instead be dominated by the patch's outsized effect on global
    variance, which defeats the point of this being a *global/flicker*
    signal rather than a second localised-change detector (that's signal 3's
    job). Returns a value in [-1, 1]; 1 == identical.
    """
    gray_a, gray_b = _grayscale(arr_a), _grayscale(arr_b)
    height, width = gray_a.shape
    win = max(1, min(window, height, width))
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2

    scores: list[float] = []
    weights: list[float] = []
    for r0 in range(0, height, win):
        for c0 in range(0, width, win):
            block_a = gray_a[r0 : r0 + win, c0 : c0 + win]
            block_b = gray_b[r0 : r0 + win, c0 : c0 + win]
            if block_a.size == 0:
                continue
            mu_a, mu_b = block_a.mean(), block_b.mean()
            var_a, var_b = block_a.var(), block_b.var()
            covariance = ((block_a - mu_a) * (block_b - mu_b)).mean()
            numerator = (2 * mu_a * mu_b + c1) * (2 * covariance + c2)
            denominator = (mu_a**2 + mu_b**2 + c1) * (var_a + var_b + c2)
            scores.append(numerator / denominator if denominator else 1.0)
            weights.append(block_a.size)

    return float(np.average(scores, weights=weights))


# -------------------------------------------------------------- signal 3: tiles


def tile_grid_distances(
    arr_a: np.ndarray, arr_b: np.ndarray, *, grid: tuple[int, int], distance_max: float
) -> list[TileScore]:
    """Per-tile mean-colour delta. Cheap on purpose (no per-tile histogram
    needed): a localised patch shifts its own tile's mean drastically even
    when it's invisible in the whole-frame average, which is exactly the
    failure mode signals 1 (fallback) and 2 dilute away."""
    rows, cols = grid
    height, width, _ = arr_a.shape
    row_edges = np.linspace(0, height, rows + 1).astype(int)
    col_edges = np.linspace(0, width, cols + 1).astype(int)

    scores: list[TileScore] = []
    for r in range(rows):
        for c in range(cols):
            r0, r1 = row_edges[r], row_edges[r + 1]
            c0, c1 = col_edges[c], col_edges[c + 1]
            tile_a = arr_a[r0:r1, c0:c1]
            tile_b = arr_b[r0:r1, c0:c1]
            if tile_a.size == 0 or tile_b.size == 0:
                continue
            distance = float(np.abs(tile_a.mean(axis=(0, 1)) - tile_b.mean(axis=(0, 1))).sum())
            scores.append(
                TileScore(
                    row=r,
                    col=c,
                    distance=distance,
                    tripped=distance > distance_max,
                    label=_region_label(r, c, rows, cols),
                )
            )
    return scores


# ------------------------------------------------------- signal 4: cumulative


@dataclass(frozen=True)
class _DriftPoint:
    frame_index: int
    brightness: float
    mean_rgb: tuple[float, float, float]


def cumulative_drift(arrays: Sequence[np.ndarray]) -> tuple[list[_DriftPoint], float, float]:
    """Brightness/colour drift measured against the FIRST frame (not the
    previous one) across the whole sequence, so a slow drift that no single
    adjacent pair would flag still trips once the distance from the start
    gets too large."""
    points = [
        _DriftPoint(
            frame_index=i,
            brightness=float(_grayscale(arr).mean()),
            mean_rgb=tuple(float(x) for x in arr.mean(axis=(0, 1))),
        )
        for i, arr in enumerate(arrays)
    ]
    baseline = points[0]
    max_brightness_drift = max(abs(p.brightness - baseline.brightness) for p in points)
    max_color_drift = max(
        float(np.linalg.norm(np.array(p.mean_rgb) - np.array(baseline.mean_rgb))) for p in points
    )
    return points, max_brightness_drift, max_color_drift


# ------------------------------------------------------------------ checker


class FrameConsistencyChecker:
    """Cheap, deterministic "is frame N consistent with its neighbours"
    checker for sequence assets. No VLM call -- affordable to run on every
    adjacent frame pair, unlike `JudgePort.judge()`.
    """

    def __init__(self, config: ConsistencyConfig | None = None) -> None:
        self._config = config or ConsistencyConfig()
        # Backend probed once at construction and logged -- never silently
        # give a weaker (fallback) answer while looking like the strong one.
        self._embedding_backend = "clip" if _get_clip_model() is not None else "histogram_fallback"
        self._perceptual_backend = "lpips" if _get_lpips_model() is not None else "ssim_fallback"
        log.info(
            "frame_consistency.backend_selected",
            embedding_backend=self._embedding_backend,
            perceptual_backend=self._perceptual_backend,
        )

    @property
    def embedding_backend(self) -> str:
        return self._embedding_backend

    @property
    def perceptual_backend(self) -> str:
        return self._perceptual_backend

    # ------------------------------------------------------------- public

    def check_pair(self, frame_a: FrameSource, frame_b: FrameSource, *, index_a: int, index_b: int) -> PairVerdict:
        arr_a, img_a = _as_array_and_image(frame_a)
        arr_b, img_b = _as_array_and_image(frame_b)
        return self._check_pair(arr_a, img_a, arr_b, img_b, index_a, index_b)

    def check_sequence(self, frames: Sequence[FrameSource]) -> SequenceConsistencyReport:
        if len(frames) < 2:
            raise ValueError("check_sequence needs at least 2 frames")
        decoded = [_as_array_and_image(f) for f in frames]

        pairs = tuple(
            self._check_pair(decoded[i][0], decoded[i][1], decoded[i + 1][0], decoded[i + 1][1], i, i + 1)
            for i in range(len(decoded) - 1)
        )

        arrays = [d[0] for d in decoded]
        _, max_brightness_drift, max_color_drift = cumulative_drift(arrays)
        brightness_signal = SignalScore(
            name="cumulative_brightness_drift",
            value=max_brightness_drift,
            threshold=self._config.cumulative_brightness_drift_max,
            tripped=max_brightness_drift > self._config.cumulative_brightness_drift_max,
            backend="cumulative",
        )
        color_signal = SignalScore(
            name="cumulative_color_drift",
            value=max_color_drift,
            threshold=self._config.cumulative_color_drift_max,
            tripped=max_color_drift > self._config.cumulative_color_drift_max,
            backend="cumulative",
        )

        return SequenceConsistencyReport(
            pairs=pairs,
            brightness_drift=brightness_signal,
            color_drift=color_signal,
            embedding_backend=self._embedding_backend,
            perceptual_backend=self._perceptual_backend,
            passed=all(p.passed for p in pairs) and not brightness_signal.tripped and not color_signal.tripped,
        )

    # ------------------------------------------------------------ internals

    def _check_pair(
        self,
        arr_a: np.ndarray,
        img_a: Image.Image,
        arr_b: np.ndarray,
        img_b: Image.Image,
        index_a: int,
        index_b: int,
    ) -> PairVerdict:
        arr_b, img_b = _match_size(arr_a, arr_b, img_b)
        cfg = self._config

        embedding_value, embedding_backend = self._embedding_similarity(img_a, img_b, arr_a, arr_b)
        embedding_signal = SignalScore(
            name="embedding_similarity",
            value=embedding_value,
            threshold=cfg.embedding_similarity_min,
            tripped=embedding_value < cfg.embedding_similarity_min,
            backend=embedding_backend,
        )

        perceptual_value, perceptual_backend = self._perceptual_distance(img_a, img_b, arr_a, arr_b)
        perceptual_signal = SignalScore(
            name="perceptual_distance",
            value=perceptual_value,
            threshold=cfg.perceptual_distance_max,
            tripped=perceptual_value > cfg.perceptual_distance_max,
            backend=perceptual_backend,
        )

        tiles = tile_grid_distances(arr_a, arr_b, grid=cfg.tile_grid, distance_max=cfg.tile_distance_max)
        worst_tile_distance = max((t.distance for t in tiles), default=0.0)
        tile_signal = SignalScore(
            name="tile",
            value=worst_tile_distance,
            threshold=cfg.tile_distance_max,
            tripped=any(t.tripped for t in tiles),
            backend="tile_grid",
        )

        passed = not (embedding_signal.tripped or perceptual_signal.tripped or tile_signal.tripped)
        return PairVerdict(
            frame_a=index_a,
            frame_b=index_b,
            embedding_similarity=embedding_signal,
            perceptual_distance=perceptual_signal,
            tile_signal=tile_signal,
            tiles=tuple(tiles),
            passed=passed,
        )

    def _embedding_similarity(
        self, img_a: Image.Image, img_b: Image.Image, arr_a: np.ndarray, arr_b: np.ndarray
    ) -> tuple[float, str]:
        if self._embedding_backend == "clip":
            try:
                return clip_cosine_similarity(img_a, img_b), "clip"
            except RuntimeError:
                log.warning("frame_consistency.clip_call_failed_falling_back")
        return histogram_correlation(arr_a, arr_b, bins=self._config.histogram_bins), "histogram_fallback"

    def _perceptual_distance(
        self, img_a: Image.Image, img_b: Image.Image, arr_a: np.ndarray, arr_b: np.ndarray
    ) -> tuple[float, str]:
        if self._perceptual_backend == "lpips":
            try:
                return lpips_distance(img_a, img_b), "lpips"
            except RuntimeError:
                log.warning("frame_consistency.lpips_call_failed_falling_back")
        return 1.0 - ssim_global(arr_a, arr_b), "ssim_fallback"
