"""Convert structured degradation diagnoses into auditable enhancement strategies."""

from __future__ import annotations

from typing import Any


def _parameters(operation: str, severity: int, defaults: dict[str, Any]) -> dict[str, Any]:
    key = str(max(1, min(3, severity)))
    tables = {
        "gamma_correction": ("gamma_by_severity", "gamma"),
        "clahe": ("clahe_clip_by_severity", "clip_limit"),
        "denoise": ("denoise_strength_by_severity", "strength"),
        "mild_sharpening": ("sharpen_amount_by_severity", "amount"),
        "saturation_reduction": ("saturation_scale_by_severity", "scale"),
    }
    if operation not in tables:
        return {}
    table_name, parameter_name = tables[operation]
    table = defaults.get(table_name, {})
    return {parameter_name: table.get(key)} if key in table else {}


def map_diagnosis_to_strategy(
    diagnosis: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Map active diagnoses to an ordered strategy and explicitly resolve conflicts."""
    degradation_map = config.get("degradations")
    if not isinstance(degradation_map, dict):
        raise ValueError("strategy config must contain a degradations mapping")
    diagnoses = diagnosis.get("degradations")
    if not isinstance(diagnoses, dict):
        raise ValueError("diagnosis must contain a degradations mapping")
    defaults = config.get("defaults", {})
    minimum = int(defaults.get("minimum_severity", 1))
    selected: dict[str, dict[str, Any]] = {}
    rationale: list[str] = []
    active: dict[str, int] = {}

    for name, result in diagnoses.items():
        severity = int(result.get("severity", 0))
        if severity < minimum or name not in degradation_map:
            continue
        active[name] = severity
        mapping = degradation_map[name]
        rationale.append(f"{name}(severity={severity}): {mapping.get('rationale', '')}")
        for operation in mapping.get("preferred", []):
            current = selected.setdefault(operation, {"severity": 0, "reasons": []})
            current["severity"] = max(int(current["severity"]), severity)
            current["reasons"].append(name)

    removed: list[dict[str, str]] = []
    for conflict_name, conflict in config.get("conflicts", {}).items():
        trigger = conflict.get("trigger")
        if trigger not in active:
            continue
        for operation in conflict.get("remove", []):
            if operation in selected:
                selected.pop(operation)
                removed.append({"operation": operation, "reason": f"conflict:{conflict_name}"})

    steps: list[dict[str, Any]] = []
    order = list(config.get("operation_order", []))
    for operation in order:
        if operation not in selected:
            continue
        item = selected[operation]
        params = _parameters(operation, int(item["severity"]), defaults)
        if operation == "clahe" and "possible_noise" in active and "clip_limit" in params:
            cap = float(config.get("conflicts", {}).get("noise", {}).get("max_clahe_clip", 1.8))
            params["clip_limit"] = min(float(params["clip_limit"]), cap)
        steps.append(
            {
                "step": len(steps) + 1,
                "operation": operation,
                "parameters": params,
                "triggered_by": sorted(set(item["reasons"])),
                "severity": int(item["severity"]),
                "reason": ", ".join(sorted(set(item["reasons"]))),
            }
        )
    if not steps:
        steps = [{"step": 1, "operation": "preserve_original", "parameters": {}, "triggered_by": [], "severity": 0, "reason": "no configured degradation reached the minimum severity"}]

    return {
        "mapping_version": str(config.get("version", "unknown")),
        "strategy_source": "rule_mapping",
        "active_degradations": active,
        "recommended_pipeline": steps,
        "removed_due_to_conflicts": removed,
        "warnings": [item["reason"] for item in removed],
        "rationale": rationale,
        "needs_review": True,
        "limitations": list(config.get("limitations", [])),
    }
