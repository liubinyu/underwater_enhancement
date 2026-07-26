"""Shared readers and small, dependency-free evaluation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file(): return []
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip(): continue
        try: rows.append(json.loads(line))
        except json.JSONDecodeError as exc: raise ValueError(f"invalid JSONL {path}:{number}: {exc}") from exc
    return rows


def read_review(path: Path | None, review_type: str) -> dict[str, dict[str, str]]:
    if path is None or not path.is_file(): return {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {row["sample_id"]: row for row in csv.DictReader(handle) if row.get("review_type") == review_type and row.get("review_status", "").casefold() in {"reviewed", "complete", "completed"}}


def prf(predicted: set[str], truth: set[str]) -> dict[str, float]:
    tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0}


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
