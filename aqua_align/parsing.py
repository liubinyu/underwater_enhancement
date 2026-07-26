"""Conservative parsing for VLM JSON output; parsing never invents missing fields."""

from __future__ import annotations

import json
import re
from typing import Any


def _balanced_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0: return None
    depth, quoted, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if quoted:
            if escaped: escaped = False
            elif char == "\\": escaped = True
            elif char == '"': quoted = False
        elif char == '"': quoted = True
        elif char == "{": depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0: return text[start:index + 1]
    return None


def parse_model_output(raw_text: str) -> dict[str, Any]:
    """Extract one JSON object using minimal documented repairs and retain failures."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        return {"success": False, "parsed": None, "error": "empty model output", "repair": None}
    text = raw_text.strip().replace("“", '"').replace("”", '"').replace("’", "'")
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = _balanced_object(fenced.group(1) if fenced else text)
    if candidate is None:
        return {"success": False, "parsed": None, "error": "no complete JSON object found", "repair": None}
    attempts = [(candidate, None)]
    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    if repaired != candidate: attempts.append((repaired, "removed_trailing_commas"))
    for value, repair in attempts:
        try: parsed = json.loads(value)
        except json.JSONDecodeError as exc: last_error = str(exc); continue
        if not isinstance(parsed, dict):
            return {"success": False, "parsed": None, "error": "top-level JSON must be an object", "repair": repair}
        return {"success": True, "parsed": parsed, "error": None, "repair": repair}
    return {"success": False, "parsed": None, "error": f"JSON decode failed: {last_error}", "repair": None}


def validate_task_output(task: str, parsed: dict[str, Any]) -> list[str]:
    """Return missing required top-level fields without modifying the parsed object."""
    required = {
        "diagnose": {"degradations", "overall_confidence", "uncertainties"},
        "strategy": {"degradations", "recommended_operations", "avoid", "confidence", "limitations"},
        "compare": {"overall_better", "color_cast_improved", "contrast_improved", "detail_preserved", "artifacts", "confidence", "reason"},
        "rank": {"preferred_candidate", "confidence", "candidate_a", "candidate_b", "reason", "limitations"},
    }
    if task not in required: raise ValueError(f"unsupported task: {task}")
    return sorted(required[task] - set(parsed))
