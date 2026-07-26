"""Versioned zero-shot prompts and strict response schemas for Base VLM evaluation."""

from __future__ import annotations

import json


PROMPT_VERSION = "0.1"
SYSTEM_PROMPT = """You are evaluating underwater image degradation and enhancement. Use only visible evidence in the supplied image(s). Do not infer water depth, sea area, camera model, physical water parameters, or unavailable quality metrics. Vivid color alone is not proof of quality. Return exactly one JSON object and no markdown."""

SCHEMAS = {
    "diagnose": {"degradations": [{"type": "label", "severity": "none|mild|moderate|severe", "confidence": "0..1", "evidence": "visible evidence"}], "overall_confidence": "0..1", "uncertainties": ["string"]},
    "strategy": {"degradations": [{"type": "label", "severity": "none|mild|moderate|severe", "evidence": "string"}], "recommended_operations": [{"operation": "string", "strength": "mild|moderate|strong", "reason": "string"}], "avoid": ["operation"], "confidence": "0..1", "limitations": ["string"]},
    "compare": {"overall_better": "original|enhanced|tie", "color_cast_improved": "yes|no|uncertain", "contrast_improved": "yes|no|uncertain", "detail_preserved": "yes|no|uncertain", "artifacts": ["over_saturation|over_sharpening|noise|halo|other"], "confidence": "0..1", "reason": "string"},
    "rank": {"preferred_candidate": "A|B|tie", "confidence": "0..1", "candidate_a": {"advantages": ["string"], "disadvantages": ["string"]}, "candidate_b": {"advantages": ["string"], "disadvantages": ["string"]}, "reason": "string", "limitations": ["string"]},
}
IMAGE_COUNTS = {"diagnose": 1, "strategy": 1, "compare": 2, "rank": 3}


def build_prompt(task: str) -> str:
    """Build a zero-shot task prompt containing the exact expected JSON schema."""
    if task not in SCHEMAS:
        raise ValueError(f"unsupported task: {task}")
    instructions = {
        "diagnose": "Diagnose visible color cast, illumination, contrast, blur/detail loss, haze-like appearance, saturation, and noise. Omit unsupported labels.",
        "strategy": "Recommend conservative enhancement operations for the visible degradations. State operations to avoid when they could amplify noise or saturation.",
        "compare": "Image 1 is the original and image 2 is the enhanced candidate. Compare color cast, contrast, preserved detail, over-saturation, over-sharpening, noise, halo, and overall quality. Do not prefer an image only because it is more vivid.",
        "rank": "Image 1 is the original, image 2 is candidate A, and image 3 is candidate B. Rank A and B by overall underwater enhancement quality. A tie is allowed and image order must not determine the answer.",
    }[task]
    return f"Task: {task}. {instructions}\nRequired JSON schema: {json.dumps(SCHEMAS[task], ensure_ascii=False)}"


def expected_image_count(task: str) -> int:
    """Return how many images must be supplied for a task."""
    if task not in IMAGE_COUNTS:
        raise ValueError(f"unsupported task: {task}")
    return IMAGE_COUNTS[task]
