"""Deterministic, free pre-checks that run before the paid VLM judge call.

numpy + Pillow only — no network, no model, no API key. An objective failure
here (blur, blown-out exposure, wrong aspect ratio, an off-brand colour wash,
a subject that runs off the frame) short-circuits the whole judge call: an
asset that fails on simple physics never costs a Claude request.

These are intentionally crude heuristics, not ML models — that's the point.
They exist to catch the cheap, unambiguous failures so the VLM budget is
spent only on judgement calls that actually need judgement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..models import CheckResult

# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class PrefilterConfig:
    min_variance_of_laplacian: float = 40.0
    brightness_min: float = 15.0
    brightness_max: float = 240.0
    aspect_ratio_min: float = 0.4
    aspect_ratio_max: float = 2.6
    brand_palette_rgb: tuple[tuple[int, int, int], ...] = ()
    max_palette_distance: float = 140.0
    edge_margin_fraction: float = 0.02
    edge_density_threshold: float = 0.35

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> PrefilterConfig:
        raw = raw or {}
        palette = tuple(tuple(int(c) for c in rgb) for rgb in raw.get("brand_palette_rgb", ()))
        return cls(
            min_variance_of_laplacian=float(raw.get("min_variance_of_laplacian", 40.0)),
            brightness_min=float(raw.get("brightness_min", 15.0)),
            brightness_max=float(raw.get("brightness_max", 240.0)),
            aspect_ratio_min=float(raw.get("aspect_ratio_min", 0.4)),
            aspect_ratio_max=float(raw.get("aspect_ratio_max", 2.6)),
            brand_palette_rgb=palette,
            max_palette_distance=float(raw.get("max_palette_distance", 140.0)),
            edge_margin_fraction=float(raw.get("edge_margin_fraction", 0.02)),
            edge_density_threshold=float(raw.get("edge_density_threshold", 0.35)),
        )


# ------------------------------------------------------------------- measures


def _to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float64)


def variance_of_laplacian(image: Image.Image) -> float:
    """Focus measure. Low variance of the Laplacian == few sharp edges == blur.

    Implemented by hand (no scipy dependency) as a 3x3 discrete Laplacian via
    shifted-and-summed slices of an edge-padded array.
    """
    gray = _to_gray(image)
    padded = np.pad(gray, 1, mode="edge")
    lap = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:] - 4 * padded[1:-1, 1:-1]
    )
    return float(lap.var())


def mean_brightness(image: Image.Image) -> float:
    return float(_to_gray(image).mean())


def aspect_ratio(image: Image.Image) -> float:
    w, h = image.size
    return w / h if h else 0.0


def dominant_color(image: Image.Image, *, sample_size: int = 64) -> tuple[int, int, int]:
    """Cheap dominant-colour estimate: shrink, then take the mean pixel.

    A mean (not a mode/histogram peak) is enough to catch a gross colour-cast
    failure — the rubric cares about "this whole image reads off-brand", not
    exact swatch matching, and a mean is far cheaper than clustering.
    """
    small = image.convert("RGB").resize((sample_size, sample_size))
    arr = np.asarray(small, dtype=np.float64).reshape(-1, 3)
    mean = arr.mean(axis=0)
    return (int(round(mean[0])), int(round(mean[1])), int(round(mean[2])))


def palette_distance(color: tuple[int, int, int], palette: tuple[tuple[int, int, int], ...]) -> float:
    if not palette:
        return 0.0
    c = np.array(color, dtype=np.float64)
    return float(min(np.linalg.norm(c - np.array(p, dtype=np.float64)) for p in palette))


def subject_touches_edge(image: Image.Image, *, margin_fraction: float, density_threshold: float) -> bool:
    """Crude crop/cutoff detector.

    Estimates the background colour from the four corners, then flags images
    where pixels that sharply differ from that background crowd the border
    band — that reads as a subject running off the edge of the frame rather
    than being deliberately composed inset from it.
    """
    arr = np.asarray(image.convert("RGB"), dtype=np.float64)
    h, w, _ = arr.shape
    corner = max(1, min(h, w) // 20)
    corners = np.concatenate(
        [
            arr[:corner, :corner].reshape(-1, 3),
            arr[:corner, -corner:].reshape(-1, 3),
            arr[-corner:, :corner].reshape(-1, 3),
            arr[-corner:, -corner:].reshape(-1, 3),
        ]
    )
    background = corners.mean(axis=0)
    diff = np.linalg.norm(arr - background, axis=2)
    fg_mask = diff > 40.0  # empirically "clearly not background"

    margin = max(1, int(round(min(h, w) * margin_fraction)))
    border = np.zeros_like(fg_mask, dtype=bool)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True

    border_fg_fraction = float(fg_mask[border].mean()) if border.any() else 0.0
    return border_fg_fraction > density_threshold


# ------------------------------------------------------------------- checks


def evaluate(image_paths: list[Path], config: PrefilterConfig) -> list[CheckResult]:
    """Run every deterministic check across all of an asset's images.

    Returns only the *failing* checks (empty list == the asset clears the
    prefilter). One `CheckResult` per failing dimension, aggregated across
    every image the asset produced — a sequence asset fails a dimension if
    any one of its frames does.
    """
    failures: dict[str, list[str]] = {
        "prefilter_sharpness": [],
        "prefilter_exposure": [],
        "prefilter_aspect_ratio": [],
        "prefilter_palette": [],
        "prefilter_framing": [],
    }

    for path in image_paths:
        with Image.open(path) as img:
            img.load()

            vlap = variance_of_laplacian(img)
            if vlap < config.min_variance_of_laplacian:
                failures["prefilter_sharpness"].append(
                    f"{path.name}: variance-of-Laplacian={vlap:.1f} < {config.min_variance_of_laplacian}"
                )

            brightness = mean_brightness(img)
            if not (config.brightness_min <= brightness <= config.brightness_max):
                failures["prefilter_exposure"].append(
                    f"{path.name}: mean brightness={brightness:.1f} outside "
                    f"[{config.brightness_min}, {config.brightness_max}]"
                )

            ar = aspect_ratio(img)
            if not (config.aspect_ratio_min <= ar <= config.aspect_ratio_max):
                failures["prefilter_aspect_ratio"].append(
                    f"{path.name}: aspect ratio={ar:.2f} outside "
                    f"[{config.aspect_ratio_min}, {config.aspect_ratio_max}]"
                )

            if config.brand_palette_rgb and config.max_palette_distance > 0:
                dom = dominant_color(img)
                dist = palette_distance(dom, config.brand_palette_rgb)
                if dist > config.max_palette_distance:
                    failures["prefilter_palette"].append(
                        f"{path.name}: dominant colour {dom} is {dist:.0f} from the brand "
                        f"palette (max {config.max_palette_distance})"
                    )

            if subject_touches_edge(
                img,
                margin_fraction=config.edge_margin_fraction,
                density_threshold=config.edge_density_threshold,
            ):
                failures["prefilter_framing"].append(
                    f"{path.name}: subject mass crowds the frame border — likely cropped/cut off"
                )

    return [
        CheckResult(name=name, evidence="; ".join(msgs), passed=False)
        for name, msgs in failures.items()
        if msgs
    ]
