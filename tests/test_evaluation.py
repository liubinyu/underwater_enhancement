from evaluation.evaluate_diagnosis import evaluate as evaluate_diagnosis
from evaluation.evaluate_hallucination import evaluate as evaluate_hallucination
from evaluation.evaluate_ranking import evaluate as evaluate_ranking


def test_no_human_labels_does_not_report_accuracy() -> None:
    rows = [{"sample_id": "s1", "degradations": {"blur": {"severity": 2}}, "success": True}]
    report = evaluate_diagnosis(rows, {})
    assert report["accuracy_metrics_available"] is False
    assert "macro_f1" not in report


def test_human_diagnosis_metrics_are_explicit() -> None:
    rows = [{"sample_id": "s1", "degradations": {"blur": {"severity": 2}}, "success": True}]
    reviews = {"s1": {"human_degradation_labels": '["blur"]', "human_severities": '{"blur": 2}'}}
    report = evaluate_diagnosis(rows, reviews)
    assert report["macro_f1"] == 1.0 and report["severity_mae"] == 0.0


def test_no_ranking_reference_stays_unavailable() -> None:
    report = evaluate_ranking([], {})
    assert report["human_agreement_available"] is False
    assert "human_agreement" not in report


def test_hallucination_keyword_screen_is_labeled_as_screening() -> None:
    report = evaluate_hallucination([{"sample_id": "s1", "raw_output": "Captured on a GoPro at 20 meters deep"}])
    assert report["flag_counts"]["unsupported_depth"] == 1
    assert report["flag_counts"]["unsupported_camera"] == 1
    assert "not_semantic_ground_truth" in report["detector_type"]
