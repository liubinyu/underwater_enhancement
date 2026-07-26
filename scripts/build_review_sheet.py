"""Build a human-review CSV for rule diagnoses and candidate preference pairs."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.config import resolve_project_path
from aqua_align.utils import setup_logger


FIELDS = [
    "review_id", "review_type", "sample_id", "split", "image_path", "original_image", "diagnosis_source",
    "rule_color_cast", "human_color_cast", "rule_low_light", "human_low_light",
    "rule_low_contrast", "human_low_contrast", "rule_blur", "human_blur",
    "rule_haze", "human_haze", "rule_strategy", "human_strategy", "rule_confidence",
    "rule_degradations", "rule_severities",
    "candidate_a_method", "candidate_a_image", "candidate_b_method", "candidate_b_image",
    "human_degradation_labels", "human_severities", "human_preferred_strategy",
    "human_preference", "human_tie", "reviewer", "reviewed_at", "review_status", "comments", "review_notes",
]


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as exc: raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
    return rows


def build_rows(diagnoses: list[dict], candidates: list[dict], *, diagnosis_limit: int, preference_limit: int, seed: int) -> list[dict]:
    """Build deterministic diagnosis and pairwise-review rows with blank human fields."""
    output: list[dict] = []
    diagnosis_by_id = {row.get("sample_id", ""): row for row in diagnoses if row.get("success", True)}
    for index, item in enumerate(list(diagnosis_by_id.values())[:diagnosis_limit], 1):
        active = {name: value for name, value in item.get("degradations", {}).items() if int(value.get("severity", 0)) > 0}
        row = {field: "" for field in FIELDS}
        row.update({
            "review_id": f"diagnosis_{index:04d}", "review_type": "diagnosis", "sample_id": item.get("sample_id", ""),
            "split": item.get("split", ""), "original_image": item.get("image_path", ""), "diagnosis_source": "rule_based",
            "image_path": item.get("image_path", ""),
            "rule_color_cast": max((active.get("blue_green_color_cast", {}).get("severity", 0), active.get("red_color_cast", {}).get("severity", 0))),
            "rule_low_light": active.get("low_light", {}).get("severity", 0),
            "rule_low_contrast": active.get("low_contrast", {}).get("severity", 0),
            "rule_blur": active.get("blur", {}).get("severity", 0),
            "rule_haze": active.get("possible_haze", {}).get("severity", 0),
            "rule_degradations": json.dumps(sorted(active), ensure_ascii=False),
            "rule_severities": json.dumps({key: value["severity"] for key, value in active.items()}, ensure_ascii=False, sort_keys=True),
            "rule_confidence": item.get("global_confidence", ""),
            "rule_strategy": json.dumps(item.get("strategy", {}).get("recommended_pipeline", []), ensure_ascii=False, sort_keys=True),
            "review_status": "pending",
        })
        output.append(row)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        if candidate.get("sample_id") in diagnosis_by_id:
            grouped[candidate["sample_id"]].append(candidate)
    pairs = []
    for sample_id in sorted(grouped):
        choices = sorted(grouped[sample_id], key=lambda item: (item.get("method", ""), item.get("candidate_image", "")))
        pairs.extend((sample_id, a, b) for a, b in itertools.combinations(choices, 2))
    random.Random(seed).shuffle(pairs)
    for index, (sample_id, first, second) in enumerate(pairs[:preference_limit], 1):
        diagnosis = diagnosis_by_id[sample_id]
        row = {field: "" for field in FIELDS}
        row.update({
            "review_id": f"preference_{index:04d}", "review_type": "preference", "sample_id": sample_id,
            "split": diagnosis.get("split", first.get("split", "")), "original_image": diagnosis.get("image_path", first.get("raw_image", "")),
            "image_path": diagnosis.get("image_path", first.get("raw_image", "")),
            "diagnosis_source": "rule_based", "candidate_a_method": first.get("method", ""), "candidate_a_image": first.get("candidate_image", ""),
            "candidate_b_method": second.get("method", ""), "candidate_b_image": second.get("candidate_image", ""),
            "review_status": "pending",
        })
        output.append(row)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnosis", type=Path, default=Path("data/annotations/rule_diagnosis.jsonl"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/annotations/rule_review.csv"))
    parser.add_argument("--diagnosis-limit", type=int, default=50)
    parser.add_argument("--preference-limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); logger = setup_logger("aqua_align.build_review_sheet")
    output = resolve_project_path(args.output)
    try:
        if output.exists() and not args.overwrite: raise FileExistsError(f"output exists; pass --overwrite: {output}")
        diagnoses = _read_jsonl(resolve_project_path(args.diagnosis))
        with resolve_project_path(args.candidates).open("r", newline="", encoding="utf-8-sig") as handle: candidates = list(csv.DictReader(handle))
        rows = build_rows(diagnoses, candidates, diagnosis_limit=args.diagnosis_limit, preference_limit=args.preference_limit, seed=args.seed)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS); writer.writeheader(); writer.writerows(rows)
        summary = {"diagnosis_rows": sum(row["review_type"] == "diagnosis" for row in rows), "preference_rows": sum(row["review_type"] == "preference" for row in rows), "human_fields_initialized_blank": True, "source_is_ground_truth": False}
        output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        logger.exception("Review sheet generation failed"); return 1
    logger.info("Wrote %d review rows to %s", len(rows), output); return 0


if __name__ == "__main__":
    raise SystemExit(main())
