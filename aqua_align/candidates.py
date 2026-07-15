"""Batch candidate generation for traditional enhancement baselines."""

from __future__ import annotations

import csv
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from aqua_align.config import project_relative_path, project_root, resolve_project_path
from aqua_align.enhancement import apply_enhancement


CANDIDATE_FIELDS = [
    "sample_id",
    "split",
    "method",
    "parameters",
    "raw_image",
    "candidate_image",
    "reference_image",
    "has_reference",
]


def _enabled(config: dict[str, Any]) -> bool:
    return bool(config.get("enabled", True))


def _without_enabled(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if key != "enabled"}


def method_specs(config: dict[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    """Expand YAML method configuration into output name, method, parameter tuples."""
    generation = config.get("candidate_generation", {})
    methods = generation.get("methods", {})
    if not isinstance(methods, dict):
        raise ValueError("candidate_generation.methods must be a mapping")
    specs: list[tuple[str, str, dict[str, Any]]] = []
    aliases = {
        "simplest_color_balance": "color_balance",
        "white_balance_clahe": "wb_clahe",
    }
    for method in (
        "gray_world",
        "white_patch",
        "clahe",
        "simplest_color_balance",
        "retinex",
        "white_balance_clahe",
    ):
        settings = methods.get(method, {})
        if not isinstance(settings, dict):
            raise ValueError(f"configuration for {method} must be a mapping")
        if _enabled(settings):
            specs.append((aliases.get(method, method), method, _without_enabled(settings)))

    gamma_settings = methods.get("gamma", {})
    if not isinstance(gamma_settings, dict):
        raise ValueError("configuration for gamma must be a mapping")
    if _enabled(gamma_settings):
        values = gamma_settings.get("values", [])
        if not isinstance(values, list) or not values:
            raise ValueError("gamma.values must be a non-empty list")
        for value in values:
            gamma = float(value)
            label = np.format_float_positional(gamma, trim="-")
            specs.append((f"gamma_{label}", "gamma_correction", {"gamma": gamma}))
    if not specs:
        raise ValueError("at least one enhancement method must be enabled")
    names = [name for name, _, _ in specs]
    if len(set(names)) != len(names):
        raise ValueError("candidate method names must be unique")
    return specs


def read_metadata(path: str | Path) -> list[dict[str, str]]:
    """Read and minimally validate prepared metadata rows."""
    metadata_path = resolve_project_path(path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata CSV not found: {metadata_path}")
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "raw_image", "reference_image", "has_reference", "split"}
    if not rows:
        raise ValueError("metadata CSV contains no rows")
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"metadata CSV is missing fields: {', '.join(sorted(missing))}")
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("metadata CSV contains duplicate sample_id values")
    return rows


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _save_jpeg(image: np.ndarray, path: Path, quality: int, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path, format="JPEG", quality=quality, subsampling=0)


def _process_sample(
    metadata_row: dict[str, str],
    output_dir: Path,
    specs: list[tuple[str, str, dict[str, Any]]],
    jpeg_quality: int,
    overwrite: bool,
) -> tuple[list[dict[str, Any]], str | None]:
    sample_id = metadata_row["sample_id"]
    try:
        raw_source = resolve_project_path(metadata_row["raw_image"])
        if not raw_source.is_file():
            raise FileNotFoundError(f"raw image not found: {raw_source}")
        raw = _load_rgb(raw_source)
        has_reference = _as_bool(metadata_row["has_reference"])
        reference_source: Path | None = None
        reference: np.ndarray | None = None
        if has_reference:
            if not metadata_row["reference_image"]:
                raise ValueError("has_reference=true but reference_image is empty")
            reference_source = resolve_project_path(metadata_row["reference_image"])
            if not reference_source.is_file():
                raise FileNotFoundError(f"reference image not found: {reference_source}")
            reference = _load_rgb(reference_source)

        sample_dir = output_dir / sample_id
        raw_destination = sample_dir / "raw.jpg"
        _save_jpeg(raw, raw_destination, jpeg_quality, overwrite)
        reference_destination: Path | None = None
        if reference is not None:
            reference_destination = sample_dir / "reference.jpg"
            _save_jpeg(reference, reference_destination, jpeg_quality, overwrite)

        candidate_rows: list[dict[str, Any]] = []
        for output_name, method, parameters in specs:
            candidate_path = sample_dir / f"{output_name}.jpg"
            if not candidate_path.exists() or overwrite:
                enhanced = apply_enhancement(raw, method, parameters)
                _save_jpeg(enhanced, candidate_path, jpeg_quality, overwrite=True)
            candidate_rows.append(
                {
                    "sample_id": sample_id,
                    "split": metadata_row["split"],
                    "method": output_name,
                    "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    "raw_image": project_relative_path(raw_destination),
                    "candidate_image": project_relative_path(candidate_path),
                    "reference_image": (
                        project_relative_path(reference_destination) if reference_destination else ""
                    ),
                    "has_reference": has_reference,
                }
            )
        return candidate_rows, None
    except Exception as exc:
        return [], f"{sample_id}: {type(exc).__name__}: {exc}"


def generate_candidates(
    metadata_path: str | Path,
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    output_csv: str | Path = "data/processed/candidates.csv",
    limit: int | None = None,
    overwrite: bool = False,
    workers: int = 1,
    dry_run: bool = False,
    logger: logging.Logger | None = None,
) -> list[dict[str, Any]]:
    """Generate configured candidates and write candidates.csv."""
    log = logger or logging.getLogger(__name__)
    rows = read_metadata(metadata_path)
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        rows = rows[:limit]
    if workers <= 0:
        raise ValueError("workers must be greater than zero")
    specs = method_specs(config)
    output = resolve_project_path(output_dir)
    project_relative_path(output)
    csv_path = resolve_project_path(output_csv)
    project_relative_path(csv_path)
    if csv_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"candidates CSV already exists; pass --overwrite: {csv_path}")
    if dry_run:
        log.info("Dry run: %d samples, %d methods, output=%s", len(rows), len(specs), output)
        return []

    jpeg_quality = int(config.get("candidate_generation", {}).get("jpeg_quality", 95))
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("candidate_generation.jpeg_quality must be in [1, 100]")
    output.mkdir(parents=True, exist_ok=True)
    worker_args = [(row, output, specs, jpeg_quality, overwrite) for row in rows]
    if workers == 1:
        results = [_process_sample(*args) for args in worker_args]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(lambda args: _process_sample(*args), worker_args))

    candidate_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for generated, error in results:
        candidate_rows.extend(generated)
        if error:
            errors.append(error)
            log.error(error)
    if not candidate_rows:
        raise RuntimeError("candidate generation produced no valid rows")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANDIDATE_FIELDS)
        writer.writeheader()
        writer.writerows(candidate_rows)
    log.info("Generated %d candidate rows at %s", len(candidate_rows), csv_path)
    if errors:
        raise RuntimeError(f"candidate generation failed for {len(errors)} samples; see log")
    return candidate_rows
