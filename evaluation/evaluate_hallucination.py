"""Flag unsupported claims in raw Base VLM outputs with transparent keyword rules."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.common import read_jsonl, write_report


PATTERNS = {
    "unsupported_depth": re.compile(r"\b(?:depth|meters? deep|metres? deep)\b", re.I),
    "unsupported_location": re.compile(r"\b(?:pacific|atlantic|indian ocean|mediterranean|south china sea)\b", re.I),
    "unsupported_camera": re.compile(r"\b(?:gopro|canon|nikon|sony camera|camera model)\b", re.I),
    "invented_metric": re.compile(r"\b(?:psnr|ssim|uiqm|uciqe)\s*(?:=|is|of)?\s*\d", re.I),
    "vividness_only": re.compile(r"(?:prefer|best|better).{0,40}(?:vivid|saturat).{0,20}(?:alone|only)", re.I),
    "unsupported_object": re.compile(r"\b(?:definitely|clearly) (?:a |an )?(?:shark|whale|shipwreck|coral species)\b", re.I),
    "image_order_confusion": re.compile(r"(?:image|candidate)\s*A\s*(?:is|means)\s*(?:the )?original", re.I),
}


def evaluate(rows: list[dict]) -> dict:
    counts, examples = Counter(), {key: [] for key in PATTERNS}
    for row in rows:
        text = str(row.get("raw_response", row.get("raw_output", "")))
        for name, pattern in PATTERNS.items():
            if pattern.search(text):
                counts[name] += 1
                if len(examples[name]) < 5: examples[name].append(row.get("sample_id", ""))
    total = len(rows)
    rates = {key: counts[key] / total if total else 0.0 for key in PATTERNS}
    any_hallucination = sum(any(pattern.search(str(row.get("raw_response", row.get("raw_output", "")))) for pattern in PATTERNS.values()) for row in rows)
    return {"rows_scanned": total, "detector_type": "keyword_screening_not_semantic_ground_truth", "hallucination_rate": any_hallucination / total if total else 0.0, "unsupported_metric_rate": rates["invented_metric"], "image_order_confusion_rate": rates["image_order_confusion"], "flag_counts": dict(counts), "flag_rates": rates, "example_sample_ids": examples}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--predictions", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args(); write_report(evaluate(read_jsonl(args.predictions)), args.output); return 0


if __name__ == "__main__": raise SystemExit(main())
