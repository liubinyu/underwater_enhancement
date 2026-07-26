"""Evaluate strategy agreement only against completed human strategy reviews."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import read_jsonl, read_review, write_report


def _operations(row: dict) -> list[str]:
    source = row.get("parsed_response", row.get("parsed", row))
    values = source.get("recommended_operations", source.get("strategy", {}).get("recommended_pipeline", [])) if isinstance(source, dict) else []
    return [str(item.get("operation")) for item in values if isinstance(item, dict) and item.get("operation")]


def evaluate(rows: list[dict], reviews: dict[str, dict[str, str]]) -> dict:
    frequency = Counter(op for row in rows for op in _operations(row)); top1 = topk = compared = violations = 0
    ordered = 0
    canonical = ["denoise", "red_channel_compensation", "white_balance", "saturation_reduction", "gamma_correction", "retinex", "clahe", "mild_sharpening"]
    for row in rows:
        ops = _operations(row); source = row.get("parsed_response", row.get("parsed", row)); avoid = source.get("avoid", []) if isinstance(source, dict) else []
        violations += bool(set(ops) & set(avoid)); positions = [canonical.index(op) for op in ops if op in canonical]; ordered += positions == sorted(positions)
    for row in rows:
        review = reviews.get(row.get("sample_id", "")); value = review.get("human_preferred_strategy", "") if review else ""
        if not value.strip(): continue
        truth = json.loads(value); truth = truth if isinstance(truth, list) else [truth]; predicted = _operations(row); compared += 1
        top1 += bool(predicted and truth and predicted[0] == truth[0]); topk += bool(set(predicted) & set(truth))
    report = {"prediction_rows": len(rows), "operation_frequency": dict(frequency), "prohibited_strategy_violation_rate": violations / len(rows) if rows else 0.0, "operation_order_reasonable_rate": ordered / len(rows) if rows else 0.0, "human_reviewed_rows": compared, "agreement_metrics_available": compared > 0}
    if compared: report.update({"top1_agreement": top1 / compared, "topk_agreement": topk / compared})
    else: report["note"] = "No completed human strategy labels were found; top-k agreement is intentionally not reported."
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--review", type=Path); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); write_report(evaluate(read_jsonl(args.predictions), read_review(args.review, "diagnosis")), args.output); return 0


if __name__ == "__main__": raise SystemExit(main())
