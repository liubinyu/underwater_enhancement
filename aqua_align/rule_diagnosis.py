"""Config-driven rule baseline for underwater degradation diagnosis."""

from __future__ import annotations

from typing import Any


def _indicator_severity(value: float, indicator: dict[str, Any]) -> tuple[int, float]:
    direction = indicator.get("direction")
    mild = float(indicator["mild"])
    moderate = float(indicator["moderate"])
    severe = float(indicator["severe"])
    if direction == "high":
        if not mild <= moderate <= severe:
            raise ValueError("high-direction thresholds must satisfy mild <= moderate <= severe")
        if value >= severe:
            return 3, severe
        if value >= moderate:
            return 2, moderate
        if value >= mild:
            return 1, mild
        return 0, mild
    if direction == "low":
        if not mild >= moderate >= severe:
            raise ValueError("low-direction thresholds must satisfy mild >= moderate >= severe")
        if value <= severe:
            return 3, severe
        if value <= moderate:
            return 2, moderate
        if value <= mild:
            return 1, mild
        return 0, mild
    raise ValueError(f"unknown indicator direction: {direction!r}")


def diagnose_features(
    features: dict[str, float | str],
    rules: dict[str, Any],
    *,
    sample_id: str = "",
) -> dict[str, Any]:
    """Evaluate all configured degradations using multiple weighted indicators."""
    degradations_config = rules.get("degradations")
    if not isinstance(degradations_config, dict) or not degradations_config:
        raise ValueError("rules.degradations must be a non-empty mapping")
    confidence_config = rules.get("confidence", {})
    inactive_confidence = float(confidence_config.get("inactive", 0.35))
    base_active = float(confidence_config.get("base_active", 0.55))
    agreement_bonus = float(confidence_config.get("agreement_bonus", 0.30))
    severity_bonus = float(confidence_config.get("severity_bonus", 0.15))

    diagnoses: dict[str, Any] = {}
    active_confidences: list[float] = []
    for degradation, config in degradations_config.items():
        indicators = config.get("indicators", {})
        if not isinstance(indicators, dict) or not indicators:
            raise ValueError(f"{degradation}.indicators must be a non-empty mapping")
        min_indicators = int(config.get("min_indicators", 2))
        observations: list[dict[str, Any]] = []
        weighted_severity = 0.0
        active_weight = 0.0
        total_weight = 0.0
        active_count = 0
        for feature_name, indicator in indicators.items():
            if feature_name not in features:
                raise KeyError(f"feature required by rules is missing: {feature_name}")
            value = float(features[feature_name])
            severity, triggered_threshold = _indicator_severity(value, indicator)
            weight = float(indicator.get("weight", 1.0))
            if weight <= 0.0:
                raise ValueError(f"indicator weight must be positive: {degradation}.{feature_name}")
            total_weight += weight
            if severity > 0:
                active_count += 1
                active_weight += weight
                weighted_severity += severity * weight
            observations.append(
                {
                    "feature": feature_name,
                    "value": value,
                    "direction": indicator["direction"],
                    "thresholds": {
                        "mild": float(indicator["mild"]),
                        "moderate": float(indicator["moderate"]),
                        "severe": float(indicator["severe"]),
                    },
                    "triggered_severity": severity,
                    "triggered_threshold": triggered_threshold if severity > 0 else None,
                }
            )

        if active_count < min_indicators or active_weight <= 0.0:
            severity = 0
            confidence = inactive_confidence
        else:
            severity = int(min(3, max(1, round(weighted_severity / active_weight))))
            agreement = active_weight / max(total_weight, 1e-8)
            confidence = base_active + agreement_bonus * agreement + severity_bonus * (severity / 3.0)
            confidence = float(min(1.0, max(0.0, confidence)))
            active_confidences.append(confidence)
        evidence = [
            (
                f"{item['feature']}={item['value']:.4f}; direction={item['direction']}; "
                f"thresholds={item['thresholds']}; indicator_severity={item['triggered_severity']}"
            )
            for item in observations
        ]
        diagnoses[degradation] = {
            "severity": severity,
            "confidence": confidence,
            "evidence": evidence,
            "feature_checks": observations,
            "reason": (
                f"{active_count}/{len(observations)} indicators triggered; "
                f"minimum required={min_indicators}"
            ),
        }

    global_confidence = (
        float(sum(active_confidences) / len(active_confidences))
        if active_confidences
        else inactive_confidence
    )
    return {
        "sample_id": sample_id,
        "diagnosis_source": "rule_based",
        "annotation_source": "rule_based",
        "needs_review": True,
        "rules_version": str(rules.get("version", "unknown")),
        "threshold_source": str(rules.get("threshold_source", "unknown")),
        "degradations": diagnoses,
        "global_confidence": float(min(1.0, max(0.0, global_confidence))),
        "limitations": list(rules.get("limitations", [])),
    }
