"""Manifest loader: YAML or CSV -> `list[Asset]`, with clear validation errors.

Every error message names the offending row/asset id and what was wrong with
it -- a 2am batch failing to even start over a typo should never require
reading this module's source to fix.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml

from .models import Asset, AssetKind


class ManifestError(RuntimeError):
    """Anything wrong with a manifest file: missing file, unknown workflow
    path, bad `kind`, missing `frame_count` on a sequence, duplicate id, etc."""


# A sequence asset generates one ComfyUI job per frame. Nothing downstream
# bounds that, so a typo (or a copied-in `frame_count: 200000`) is a batch
# that is still running at noon. Callers can raise or lower this; they cannot
# switch it off by accident.
DEFAULT_MAX_FRAME_COUNT = 1000


def load_manifest(
    path: str | Path,
    *,
    workflow_dir: str | Path | None = None,
    max_frame_count: int = DEFAULT_MAX_FRAME_COUNT,
) -> list[Asset]:
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        rows = _load_yaml_rows(path)
    elif suffix == ".csv":
        rows = _load_csv_rows(path)
    else:
        raise ManifestError(f"unsupported manifest extension {suffix!r} (use .yaml/.yml/.csv): {path}")

    assets: list[Asset] = []
    seen_ids: set[str] = set()
    for i, row in enumerate(rows):
        asset = _row_to_asset(row, index=i, workflow_dir=workflow_dir, max_frame_count=max_frame_count)
        if asset.id in seen_ids:
            raise ManifestError(f"row {i}: duplicate asset id {asset.id!r}")
        seen_ids.add(asset.id)
        assets.append(asset)

    if not assets:
        raise ManifestError(f"manifest {path} contains no assets")
    return assets


def _load_yaml_rows(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text()) or []
    if isinstance(raw, dict):
        raw = raw.get("assets", [])
    if not isinstance(raw, list):
        raise ManifestError(f"{path}: expected a list of assets (optionally under an `assets:` key)")
    for i, row in enumerate(raw):
        if not isinstance(row, dict):
            raise ManifestError(f"{path}: row {i} is not a mapping: {row!r}")
    return raw


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _row_to_asset(
    row: dict[str, Any],
    *,
    index: int,
    workflow_dir: str | Path | None,
    max_frame_count: int = DEFAULT_MAX_FRAME_COUNT,
) -> Asset:
    asset_id = row.get("id")
    if not asset_id:
        raise ManifestError(f"row {index}: missing required field `id`")
    label = str(asset_id)

    workflow = row.get("workflow")
    if not workflow:
        raise ManifestError(f"asset {label!r}: missing required field `workflow`")
    workflow_path = _resolve_workflow(str(workflow), label=label, workflow_dir=workflow_dir)

    kind_raw = row.get("kind") or "single"
    try:
        kind = AssetKind(str(kind_raw).strip().lower())
    except ValueError:
        raise ManifestError(
            f"asset {label!r}: unknown kind {kind_raw!r} (must be one of {[k.value for k in AssetKind]})"
        ) from None

    inputs = row.get("inputs") or {}
    if isinstance(inputs, str):
        inputs = _parse_json_object(inputs, label=label, field_name="inputs")
    elif not isinstance(inputs, dict):
        raise ManifestError(f"asset {label!r}: `inputs` must be a mapping, got {type(inputs).__name__}")

    seed = _parse_optional_int(row.get("seed"), label=label, field_name="seed")
    frame_count = _parse_optional_int(row.get("frame_count"), label=label, field_name="frame_count")

    if kind is AssetKind.SEQUENCE and (frame_count is None or frame_count <= 0):
        raise ManifestError(f"asset {label!r}: kind=sequence requires a positive `frame_count`")
    if kind is AssetKind.SINGLE and frame_count:
        raise ManifestError(f"asset {label!r}: kind=single must not set `frame_count`")
    if frame_count is not None and frame_count > max_frame_count:
        raise ManifestError(
            f"asset {label!r}: `frame_count` {frame_count} exceeds the ceiling of {max_frame_count} "
            f"(one ComfyUI job is submitted per frame; raise `max_frame_count` deliberately if this "
            f"is really intended)"
        )

    return Asset(
        id=label,
        kind=kind,
        workflow=str(workflow_path),
        inputs=inputs,
        seed=seed,
        frame_count=frame_count,
    )


def _resolve_workflow(workflow: str, *, label: str, workflow_dir: str | Path | None) -> Path:
    """Resolve, contain, and validate one manifest row's `workflow` path.

    Three separate checks, all of which were missing or too weak:

    * **Containment.** `workflow: ../../etc/passwd` (or an absolute path)
      escaped `workflow_dir` entirely -- the manifest is operator input, but a
      configured `workflow_dir` is a stated boundary and should hold.
    * **Parses.** `exists()` only proved a file was there. A truncated or
      half-synced JSON on a shared drive passed the manifest and then blew up
      hours later, per-asset, at claim time.
    * **Shape.** An API-format graph is a JSON object, not a list or a scalar.
    """
    workflow_path = Path(workflow)
    if workflow_dir is not None:
        root = Path(workflow_dir).resolve()
        if not workflow_path.is_absolute():
            workflow_path = Path(workflow_dir) / workflow_path
        resolved = workflow_path.resolve()
        if not resolved.is_relative_to(root):
            raise ManifestError(
                f"asset {label!r}: workflow {workflow!r} resolves to {resolved}, which is outside "
                f"the configured workflow_dir {root}"
            )
        workflow_path = resolved

    if not workflow_path.exists():
        raise ManifestError(f"asset {label!r}: workflow file not found: {workflow_path}")

    try:
        parsed = json.loads(workflow_path.read_text())
    except OSError as exc:
        raise ManifestError(f"asset {label!r}: workflow {workflow_path} is unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"asset {label!r}: workflow {workflow_path} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ManifestError(
            f"asset {label!r}: workflow {workflow_path} must be an API-format JSON object, "
            f"got {type(parsed).__name__}"
        )
    return workflow_path


def _parse_optional_int(value: Any, *, label: str, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ManifestError(f"asset {label!r}: `{field_name}` must be an integer, got {value!r}") from None


def _parse_json_object(value: str, *, label: str, field_name: str) -> dict[str, Any]:
    value = value.strip()
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"asset {label!r}: `{field_name}` is not valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ManifestError(f"asset {label!r}: `{field_name}` must be a JSON object")
    return parsed
