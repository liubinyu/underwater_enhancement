"""Evaluate diagnosis format/statistics and, only when present, reviewed human labels."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import prf, read_jsonl, read_review, write_report


def _labels(row: dict) -> tuple[set[str], dict[str, float]]:
    source = row.get("parsed_response", row.get("parsed", row))
    degradations = source.get("degradations", {}) if isinstance(source, dict) else {}
    if isinstance(degradations, dict):
        severities = {name: float(value.get("severity", 0)) for name, value in degradations.items() if isinstance(value, dict) and float(value.get("severity", 0)) > 0}
    elif isinstance(degradations, list):
        scale = {"none": 0, "mild": 1, "moderate": 2, "severe": 3}
        severities = {str(item.get("type")): float(scale.get(str(item.get("severity", "none")).lower(), 0)) for item in degradations if isinstance(item, dict) and scale.get(str(item.get("severity", "none")).lower(), 0) > 0}
    else: severities = {}
    return set(severities), severities


def evaluate(rows: list[dict], reviews: dict[str, dict[str, str]]) -> dict:
    successful = [row for row in rows if row.get("success", row.get("parse_success", True))]
    counts = Counter(label for row in successful for label in _labels(row)[0])
    confidence_values = [float(row.get("global_confidence", row.get("parsed_response", {}).get("overall_confidence", 0))) for row in successful if row.get("global_confidence") is not None or isinstance(row.get("parsed_response"), dict)]
    elapsed = [float(row["inference_time_seconds"]) for row in rows if row.get("inference_time_seconds") is not None]
    complete = [row for row in rows if row.get("parse_success") and not row.get("missing_fields", [])]
    report = {"prediction_rows": len(rows), "successful_rows": len(successful), "coverage": len(successful) / len(rows) if rows else 0.0, "json_success_rate": sum(bool(row.get("parse_success")) for row in rows) / len(rows) if rows and any("parse_success" in row for row in rows) else None, "field_complete_rate": len(complete) / len(rows) if rows and any("parse_success" in row for row in rows) else None, "average_inference_time_seconds": sum(elapsed) / len(elapsed) if elapsed else None, "low_confidence_ratio": sum(value < 0.5 for value in confidence_values) / len(confidence_values) if confidence_values else None, "label_frequency": dict(counts), "human_reviewed_rows": 0, "accuracy_metrics_available": False}
    scores, severity_errors, pairs = [], [], []
    for row in successful:
        review = reviews.get(row.get("sample_id", ""))
        if not review or not review.get("human_degradation_labels", "").strip(): continue
        truth = set(json.loads(review["human_degradation_labels"])); predicted, pred_severity = _labels(row)
        scores.append(prf(predicted, truth))
        pairs.append((predicted, truth))
        if review.get("human_severities", "").strip():
            human_severity = json.loads(review["human_severities"])
            severity_errors.extend(abs(float(pred_severity.get(label, 0)) - float(value)) for label, value in human_severity.items())
    if scores:
        labels = sorted(set().union(*(pred | truth for pred, truth in pairs)))
        per_class = {}
        for label in labels:
            tp = sum(label in pred and label in truth for pred, truth in pairs); fp = sum(label in pred and label not in truth for pred, truth in pairs); fn = sum(label not in pred and label in truth for pred, truth in pairs)
            precision = tp / (tp + fp) if tp + fp else 0.0; recall = tp / (tp + fn) if tp + fn else 0.0
            per_class[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}
        report.update({"human_reviewed_rows": len(scores), "accuracy_metrics_available": True, "per_class": per_class, "macro_precision": sum(x["precision"] for x in per_class.values()) / len(per_class) if per_class else 0.0, "macro_recall": sum(x["recall"] for x in per_class.values()) / len(per_class) if per_class else 0.0, "macro_f1": sum(x["f1"] for x in per_class.values()) / len(per_class) if per_class else 0.0, "severity_mae": sum(severity_errors) / len(severity_errors) if severity_errors else None})
    else: report["note"] = "No completed human diagnosis labels were found; accuracy and F1 are intentionally not reported."
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--review", type=Path); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    write_report(evaluate(read_jsonl(args.predictions), read_review(args.review, "diagnosis")), args.output); return 0


if __name__ == "__main__": raise SystemExit(main())
