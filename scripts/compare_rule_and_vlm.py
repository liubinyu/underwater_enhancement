"""Create an evidence-limited comparison of rule and Base VLM baselines."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.config import resolve_project_path
from evaluation.common import read_jsonl, read_review, write_report
from evaluation.evaluate_diagnosis import _labels, evaluate as evaluate_diagnosis
from evaluation.evaluate_hallucination import evaluate as evaluate_hallucination
from evaluation.evaluate_ranking import evaluate as evaluate_ranking
from evaluation.evaluate_strategy import evaluate as evaluate_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-results", "--rule", dest="rule", type=Path, default=Path("data/processed/rule_diagnosis.jsonl"))
    parser.add_argument("--vlm-results", type=Path, default=Path("outputs/base_vlm/diagnose_parsed.jsonl"))
    parser.add_argument("--vlm-dir", type=Path)
    parser.add_argument("--human-review", "--review", dest="review", type=Path, default=Path("data/annotations/rule_review.csv"))
    parser.add_argument("--output", "--output-dir", dest="output_dir", type=Path, default=Path("reports/baseline_comparison"))
    return parser.parse_args()


def main() -> int:
    args = parse_args(); output = resolve_project_path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    rule_rows = read_jsonl(resolve_project_path(args.rule)); vlm_result_path = resolve_project_path(args.vlm_results); vlm_dir = resolve_project_path(args.vlm_dir) if args.vlm_dir else vlm_result_path.parent
    review_path = resolve_project_path(args.review); diagnosis_reviews = read_review(review_path, "diagnosis"); preference_reviews = read_review(review_path, "preference")
    vlm_diagnosis = read_jsonl(vlm_result_path); vlm_strategy = read_jsonl(vlm_dir / "strategy_parsed.jsonl"); vlm_compare = read_jsonl(vlm_dir / "compare_parsed.jsonl"); vlm_rank = read_jsonl(vlm_dir / "rank_parsed.jsonl"); vlm_raw = read_jsonl(vlm_dir / "diagnose_raw.jsonl") + read_jsonl(vlm_dir / "strategy_raw.jsonl") + read_jsonl(vlm_dir / "compare_raw.jsonl") + read_jsonl(vlm_dir / "rank_raw.jsonl")
    reports = {
        "rule_diagnosis": evaluate_diagnosis(rule_rows, diagnosis_reviews),
        "vlm_diagnosis": evaluate_diagnosis(vlm_diagnosis, diagnosis_reviews),
        "rule_strategy": evaluate_strategy(rule_rows, diagnosis_reviews),
        "vlm_strategy": evaluate_strategy(vlm_strategy, diagnosis_reviews),
        "vlm_ranking": evaluate_ranking(vlm_rank, preference_reviews),
        "vlm_hallucination_screen": evaluate_hallucination(vlm_raw),
    }
    run_summary_path = vlm_dir / "run_summary.json"
    run_summary = json.loads(run_summary_path.read_text(encoding="utf-8")) if run_summary_path.is_file() else {"status": "missing"}
    actual_vlm = int(run_summary.get("successful_inferences", 0)) > 0
    comparison = {"rule_rows": len(rule_rows), "base_vlm_run": run_summary, "actual_vlm_outputs_available": actual_vlm, "human_reviewed_diagnoses": len(diagnosis_reviews), "human_reviewed_preferences": len(preference_reviews), "reports": reports}
    rule_lookup = {row.get("sample_id", ""): row for row in rule_rows}
    comparable = [(row, rule_lookup[row.get("sample_id", "")]) for row in vlm_diagnosis if row.get("sample_id", "") in rule_lookup]
    comparison["rule_agreement"] = (sum(_labels(vlm)[0] == _labels(rule)[0] for vlm, rule in comparable) / len(comparable)) if comparable else None
    comparison["rule_agreement_note"] = "Exact active-label agreement with rule-generated labels; this is not real accuracy."
    if not actual_vlm: comparison["conclusion"] = "Base VLM inference results are unavailable; no rule-versus-VLM quality conclusion can be made."
    elif not diagnosis_reviews and not preference_reviews: comparison["conclusion"] = "Model outputs exist, but no completed human review exists; only format, coverage, frequency, and screening statistics are valid."
    else: comparison["conclusion"] = "See separate human-referenced metrics; agreement is not equivalent to perceptual superiority."
    write_report(comparison, output / "comparison.json")
    def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
        with (output / name).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames); writer.writeheader(); writer.writerows(rows)
    write_csv("metrics.csv", ["metric", "value", "interpretation"], [
        {"metric": "rule_rows", "value": len(rule_rows), "interpretation": "descriptive count"},
        {"metric": "successful_base_vlm_inferences", "value": run_summary.get("successful_inferences", 0), "interpretation": "actual saved inference count"},
        {"metric": "human_reviewed_diagnoses", "value": len(diagnosis_reviews), "interpretation": "completed reviews only"},
        {"metric": "human_reviewed_preferences", "value": len(preference_reviews), "interpretation": "completed reviews only"},
    ])
    rule_by_id = {row.get("sample_id", ""): row for row in rule_rows}; vlm_by_id = {row.get("sample_id", ""): row for row in vlm_diagnosis}
    ids = sorted(set(rule_by_id) | set(vlm_by_id))
    write_csv("diagnosis_comparison.csv", ["sample_id", "rule_available", "vlm_available", "human_review_available"], [{"sample_id": key, "rule_available": key in rule_by_id, "vlm_available": key in vlm_by_id, "human_review_available": key in diagnosis_reviews} for key in ids])
    write_csv("strategy_comparison.csv", ["sample_id", "rule_strategy_available", "vlm_strategy_available", "human_strategy_available"], [{"sample_id": key, "rule_strategy_available": bool(rule_by_id.get(key, {}).get("strategy")), "vlm_strategy_available": any(row.get("sample_id") == key for row in vlm_strategy), "human_strategy_available": bool(diagnosis_reviews.get(key, {}).get("human_strategy", ""))} for key in ids])
    write_csv("ranking_comparison.csv", ["sample_id", "vlm_ranking_available", "human_preference_available"], [{"sample_id": row.get("sample_id", ""), "vlm_ranking_available": True, "human_preference_available": row.get("sample_id", "") in preference_reviews} for row in vlm_rank])
    errors = read_jsonl(vlm_dir / "errors.jsonl")
    write_csv("failure_cases.csv", ["sample_id", "task_type", "error"], [{"sample_id": row.get("sample_id", ""), "task_type": row.get("task_type", row.get("scope", "")), "error": row.get("error", "")} for row in errors])
    (output / "figures").mkdir(exist_ok=True)
    markdown = "# Rule and Base VLM baseline report\n\n" + comparison["conclusion"] + "\n\n## Evidence available\n\n" + f"- Rule rows: {len(rule_rows)}\n- Successful Base VLM inferences: {run_summary.get('successful_inferences', 0)}\n- Completed diagnosis reviews: {len(diagnosis_reviews)}\n- Completed preference reviews: {len(preference_reviews)}\n\n## Method scope\n\n- Rule methods are reproducible and auditable for statistics-driven screening, but thresholds are initial engineering values and cannot establish physical depth, backscatter, or true color.\n- Base VLM strengths, hallucinations, and format behavior cannot be concluded when no actual inference output exists.\n- Samples should enter future SFT only after image-visible evidence and labels are human-reviewed; rule and Base VLM outputs are not training truth.\n- Haze, noise, blur versus lost detail, and physical color restoration remain intrinsically ambiguous from a single image.\n"
    (output / "baseline_report.md").write_text(markdown, encoding="utf-8")
    (output / "comparison.md").write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
