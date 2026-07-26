"""Run reproducible rule diagnosis and strategy mapping over metadata images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.config import load_config, resolve_project_path
from aqua_align.degradation_features import extract_degradation_features
from aqua_align.rule_diagnosis import diagnose_features
from aqua_align.strategy_mapping import map_diagnosis_to_strategy
from aqua_align.utils import set_seed, setup_logger


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/uieb/metadata.csv"))
    parser.add_argument("--rules", type=Path, default=Path("configs/diagnosis_rules.yaml"))
    parser.add_argument("--strategy", type=Path, default=Path("configs/strategy_mapping.yaml"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/rule_diagnosis.jsonl"))
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); set_seed(args.seed)
    logger = setup_logger("aqua_align.run_rule_diagnosis")
    output = resolve_project_path(args.output)
    try:
        if output.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"output exists; pass --overwrite: {output}")
        rules = load_config(resolve_project_path(args.rules)); strategy_config = load_config(resolve_project_path(args.strategy))
        with resolve_project_path(args.metadata).open("r", newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        if args.split != "all": rows = [row for row in rows if row.get("split") == args.split]
        if args.limit is not None: rows = rows[:args.limit]
        if not rows: raise ValueError("no metadata rows selected")
        if args.dry_run:
            logger.info("Dry run selected %d rows; no images were processed and no output was written", len(rows)); return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        failures = 0
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                try:
                    features = extract_degradation_features(_load_rgb(resolve_project_path(row["raw_image"])))
                    result = diagnose_features(features, rules, sample_id=row["sample_id"])
                    result.update({"image_path": row["raw_image"], "split": row["split"], "features": features})
                    result["strategy"] = map_diagnosis_to_strategy(result, strategy_config)
                except Exception as exc:
                    failures += 1
                    result = {"sample_id": row.get("sample_id", ""), "image_path": row.get("raw_image", ""), "split": row.get("split", ""), "success": False, "error": f"{type(exc).__name__}: {exc}"}
                else: result["success"] = True
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    except Exception:
        logger.exception("Rule diagnosis failed"); return 1
    logger.info("Wrote %d rows to %s (%d failures)", len(rows), output, failures)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
