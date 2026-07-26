"""Summarize train-only degradation features without overwriting official rule thresholds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.config import resolve_project_path
from aqua_align.degradation_features import extract_degradation_features
from aqua_align.utils import set_seed, setup_logger


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def calibrate(metadata: Path, *, split: str, limit: int | None) -> dict:
    """Compute descriptive percentiles from one non-evaluation split."""
    if split != "train":
        raise ValueError("calibration is restricted to split=train to prevent evaluation leakage")
    raw = metadata.read_bytes()
    with metadata.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == split]
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise ValueError("no train rows available for calibration")
    records, failures = [], []
    for row in rows:
        try:
            records.append(extract_degradation_features(_load_rgb(resolve_project_path(row["raw_image"]))))
        except Exception as exc:
            failures.append({"sample_id": row.get("sample_id", ""), "error": f"{type(exc).__name__}: {exc}"})
    if not records:
        raise RuntimeError("feature extraction failed for all calibration rows")
    numeric = sorted(key for key, value in records[0].items() if isinstance(value, (int, float)))
    statistics = {}
    for key in numeric:
        values = np.asarray([float(record[key]) for record in records], dtype=np.float64)
        statistics[key] = {
            "mean": float(values.mean()), "std": float(values.std()), "min": float(values.min()),
            "p10": float(np.percentile(values, 10)), "p25": float(np.percentile(values, 25)),
            "median": float(np.median(values)), "p75": float(np.percentile(values, 75)),
            "p90": float(np.percentile(values, 90)), "max": float(values.max()),
            "suggested_low": {"mild": float(np.percentile(values, 40)), "moderate": float(np.percentile(values, 25)), "severe": float(np.percentile(values, 10))},
            "suggested_high": {"mild": float(np.percentile(values, 60)), "moderate": float(np.percentile(values, 75)), "severe": float(np.percentile(values, 90))},
        }
    return {
        "report_type": "train_only_feature_calibration",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata.relative_to(ROOT).as_posix(),
        "metadata_sha256": hashlib.sha256(raw).hexdigest(),
        "split": split, "requested_rows": len(rows), "successful_rows": len(records),
        "failed_rows": failures, "feature_version": records[0].get("feature_version", "unknown"),
        "warning": "Suggested percentiles are review aids, not validated diagnostic thresholds. This script never edits diagnosis_rules.yaml.",
        "statistics": statistics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/uieb/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/rule_calibration.json"))
    parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); set_seed(args.seed)
    logger = setup_logger("aqua_align.calibrate_rules")
    metadata, output = resolve_project_path(args.metadata), resolve_project_path(args.output)
    try:
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        report = calibrate(metadata, split=args.split, limit=args.limit)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Rule calibration failed"); return 1
    logger.info("Wrote train-only calibration report to %s", output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
