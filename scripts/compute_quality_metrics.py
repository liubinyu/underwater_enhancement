"""Compute reference and no-reference metrics for generated candidates."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.config import load_config, project_relative_path, resolve_project_path
from aqua_align.image_quality import compute_quality_metrics
from aqua_align.utils import setup_logger


METRIC_FIELDS = [
    "sample_id",
    "split",
    "method",
    "candidate_image",
    "reference_image",
    "has_reference",
    "psnr",
    "ssim",
    "mean_brightness",
    "grayscale_std",
    "laplacian_variance",
    "dark_pixel_ratio",
    "saturated_pixel_ratio",
    "red_mean",
    "green_mean",
    "blue_mean",
    "channel_imbalance",
    "colorfulness",
    "error",
]


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _bool(value: str) -> bool:
    return value.strip().casefold() in {"true", "1", "yes"}


def _finite_or_blank(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ValueError("metric output contains NaN or Infinity")
    return value


def calculate_metrics_csv(
    candidates_path: str | Path,
    output_path: str | Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    """Calculate metrics, retaining failed rows with an explicit error string."""
    candidates_csv = resolve_project_path(candidates_path)
    output_csv = resolve_project_path(output_path)
    project_relative_path(output_csv)
    if not candidates_csv.is_file():
        raise FileNotFoundError(f"candidates CSV not found: {candidates_csv}")
    with candidates_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        candidates = list(csv.DictReader(handle))
    if not candidates:
        raise ValueError("candidates CSV contains no rows")
    settings = config.get("quality_metrics", {})
    metric_rows: list[dict[str, Any]] = []
    failures = 0
    for candidate in candidates:
        base = {
            "sample_id": candidate.get("sample_id", ""),
            "split": candidate.get("split", ""),
            "method": candidate.get("method", ""),
            "candidate_image": candidate.get("candidate_image", ""),
            "reference_image": candidate.get("reference_image", ""),
            "has_reference": _bool(candidate.get("has_reference", "false")),
        }
        row = {field: base.get(field, "") for field in METRIC_FIELDS}
        try:
            candidate_path = resolve_project_path(base["candidate_image"])
            image = _load_rgb(candidate_path)
            reference = None
            if base["has_reference"]:
                if not base["reference_image"]:
                    raise ValueError("has_reference=true but reference_image is empty")
                reference = _load_rgb(resolve_project_path(base["reference_image"]))
            metrics = compute_quality_metrics(
                image,
                reference,
                dark_threshold=int(settings.get("dark_threshold", 15)),
                saturated_threshold=int(settings.get("saturated_threshold", 250)),
                identical_psnr_db=float(settings.get("identical_psnr_db", 100.0)),
            )
            for key, value in metrics.items():
                if key in row:
                    row[key] = _finite_or_blank(value)
            row["error"] = ""
        except Exception as exc:
            failures += 1
            row["error"] = f"{type(exc).__name__}: {exc}"
        metric_rows.append(row)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)
    return metric_rows, failures


def parse_args() -> argparse.Namespace:
    """Parse metric calculation arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    return parser.parse_args()


def main() -> int:
    """Calculate metrics and return nonzero if any row failed."""
    args = parse_args()
    output = resolve_project_path(args.output)
    logger = setup_logger("aqua_align.compute_metrics", log_file=output.parent / "quality_metrics.log")
    try:
        config = load_config(resolve_project_path(args.config))
        rows, failures = calculate_metrics_csv(args.candidates, output, config)
    except Exception:
        logger.exception("Metric calculation failed")
        return 1
    logger.info("Wrote %d metric rows to %s", len(rows), output)
    if failures:
        logger.error("Metric calculation failed for %d rows; errors were saved in CSV", failures)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
