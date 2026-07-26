"""Evaluate pairwise ranking against human choices, keeping reference sources separate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import read_jsonl, read_review, write_report


def evaluate(rows: list[dict], reviews: dict[str, dict[str, str]]) -> dict:
    compared = agreements = ties = 0; calibration_errors = []
    for row in rows:
        review = reviews.get(row.get("sample_id", "")); truth = review.get("human_preference", "").strip() if review else ""
        source = row.get("parsed_response", row.get("parsed", {})); predicted = source.get("preferred_candidate") if isinstance(source, dict) else None
        if not truth or predicted not in {"A", "B", "tie"}: continue
        correct = predicted == truth; compared += 1; agreements += correct; ties += predicted == "tie"
        confidence = source.get("confidence") if isinstance(source, dict) else None
        if isinstance(confidence, (int, float)): calibration_errors.append(abs(float(confidence) - float(correct)))
    report = {"prediction_rows": len(rows), "human_reviewed_pairs": compared, "human_agreement_available": compared > 0, "reference_metric_agreement_available": False, "rule_agreement_available": False}
    if compared: report.update({"human_preference_agreement": agreements / compared, "tie_rate": ties / compared, "confidence_calibration_mae": sum(calibration_errors) / len(calibration_errors) if calibration_errors else None})
    else: report["note"] = "No matching completed human pair reviews; ranking accuracy is intentionally not reported."
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--review", type=Path); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); write_report(evaluate(read_jsonl(args.predictions), read_review(args.review, "preference")), args.output); return 0


if __name__ == "__main__": raise SystemExit(main())
