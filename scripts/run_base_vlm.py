"""Run Base VLM zero-shot tasks and persist raw, parsed, environment, and error records."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import shlex
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aqua_align.config import load_config, resolve_project_path
from aqua_align.inference import BaseVLMRunner
from aqua_align.parsing import parse_model_output, validate_task_output
from aqua_align.prompts import PROMPT_VERSION
from aqua_align.utils import set_seed, setup_logger


def _version(package: str) -> str | None:
    try: return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: return None


def _environment() -> dict:
    torch_version, cuda_available, cuda_version, devices = _version("torch"), False, None, []
    try:
        import torch
        cuda_available = torch.cuda.is_available(); cuda_version = torch.version.cuda
        if cuda_available: devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except ImportError: pass
    return {"python": platform.python_version(), "platform": platform.platform(), "torch": torch_version, "transformers": _version("transformers"), "accelerate": _version("accelerate"), "cuda_available": cuda_available, "cuda_runtime": cuda_version, "cuda_devices": devices}


def _load_csv(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle: return list(csv.DictReader(handle))


def _jobs(task: str, metadata: list[dict], candidates: list[dict], limit: int | None) -> list[dict]:
    if task in {"diagnose", "strategy"}:
        jobs = [{"sample_id": row["sample_id"], "task_type": task, "image_paths": [row["raw_image"]], "split": row.get("split", "")} for row in metadata]
    else:
        grouped: dict[str, list[dict]] = defaultdict(list)
        for row in candidates: grouped[row["sample_id"]].append(row)
        jobs = []
        for sample_id in sorted(grouped):
            options = sorted(grouped[sample_id], key=lambda row: row.get("method", ""))
            if len(options) < 2: continue
            first, second = options[0], options[1]
            images = [first.get("raw_image", ""), first["candidate_image"]]
            labels = {"image_1": "original", "image_2": first.get("method", "")}
            if task == "rank":
                images.append(second["candidate_image"]); labels["image_2"] = f"candidate_A:{first.get('method', '')}"; labels["image_3"] = f"candidate_B:{second.get('method', '')}"
            jobs.append({"sample_id": sample_id, "task_type": task, "image_paths": images, "image_labels": labels, "split": first.get("split", "")})
    return jobs[:limit] if limit is not None else jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base_vlm.yaml"))
    parser.add_argument("--metadata", type=Path, default=Path("data/processed/uieb/metadata.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("data/processed/candidates.csv"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tasks", nargs="+", choices=["diagnose", "strategy", "compare", "rank"])
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args(); logger = setup_logger("aqua_align.run_base_vlm")
    config = load_config(resolve_project_path(args.config)); set_seed(int(config.get("generation", {}).get("seed", 42)))
    output = resolve_project_path(args.output_dir or Path(config.get("output", {}).get("root", "outputs/base_vlm")))
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        logger.error("Output directory is not empty; pass --overwrite: %s", output); return 1
    output.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks or list(config.get("tasks", [])); env = _environment()
    (output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    (output / "command.txt").write_text(" ".join(shlex.quote(value) for value in sys.argv), encoding="utf-8")
    for task in tasks:
        (output / f"{task}_raw.jsonl").write_text("", encoding="utf-8")
        (output / f"{task}_parsed.jsonl").write_text("", encoding="utf-8")
    errors: list[dict] = []
    blocker = None
    if config.get("model", {}).get("device") == "cuda" and not env["cuda_available"]: blocker = "config requires CUDA but torch.cuda.is_available() is false"
    elif env["transformers"] is None: blocker = "transformers is not installed"
    if blocker:
        errors.append({"scope": "preflight", "error": blocker, "timestamp": datetime.now(timezone.utc).isoformat()})
        (output / "errors.jsonl").write_text("\n".join(json.dumps(x) for x in errors) + "\n", encoding="utf-8")
        summary = {"status": "blocked", "model": config["model"]["name_or_path"], "prompt_version": config.get("prompt", {}).get("version", PROMPT_VERSION), "requested_tasks": tasks, "successful_inferences": 0, "failed_inferences": 0, "blocker": blocker, "fake_outputs_created": False}
        (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.error("Base VLM preflight blocked: %s", blocker); return 2
    metadata, candidates = _load_csv(resolve_project_path(args.metadata)), _load_csv(resolve_project_path(args.candidates))
    if args.split != "all":
        metadata = [row for row in metadata if row.get("split") == args.split]
        candidates = [row for row in candidates if row.get("split") == args.split]
    runner = BaseVLMRunner(config)
    try: runner.load()
    except Exception as exc:
        blocker = f"model load failed: {type(exc).__name__}: {exc}"; errors.append({"scope": "model_load", "error": blocker})
        (output / "errors.jsonl").write_text("\n".join(json.dumps(x) for x in errors) + "\n", encoding="utf-8")
        (output / "run_summary.json").write_text(json.dumps({"status": "blocked", "successful_inferences": 0, "blocker": blocker, "fake_outputs_created": False}, indent=2), encoding="utf-8")
        logger.error(blocker); return 2
    success, failed = 0, 0
    for task in tasks:
        raw_path, parsed_path = output / f"{task}_raw.jsonl", output / f"{task}_parsed.jsonl"
        with raw_path.open("a", encoding="utf-8") as raw_handle, parsed_path.open("a", encoding="utf-8") as parsed_handle:
            for job in _jobs(task, metadata, candidates, args.limit):
                started = time.perf_counter(); timestamp = datetime.now(timezone.utc).isoformat()
                try:
                    images = [resolve_project_path(path) for path in job["image_paths"]]
                    raw_text = runner.generate(task, images); parsed = parse_model_output(raw_text)
                    missing = validate_task_output(task, parsed["parsed"]) if parsed["success"] else []
                    record = {**job, "model_name": config["model"]["name_or_path"], "prompt_version": f"v{config.get('prompt', {}).get('version', PROMPT_VERSION).lstrip('v')}", "generation_config": config.get("generation", {}), "raw_response": raw_text, "parse_success": parsed["success"] and not missing, "inference_time_seconds": time.perf_counter() - started, "timestamp": timestamp, "error": ""}
                    raw_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    parsed_record = {**record, "parsed_response": parsed["parsed"], "parse_error": parsed["error"], "repair": parsed["repair"], "missing_fields": missing}
                    parsed_handle.write(json.dumps(parsed_record, ensure_ascii=False) + "\n"); success += 1
                except Exception as exc:
                    failed += 1; errors.append({**job, "error": f"{type(exc).__name__}: {exc}", "timestamp": timestamp})
    (output / "errors.jsonl").write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in errors) + ("\n" if errors else ""), encoding="utf-8")
    summary = {"status": "completed" if failed == 0 else "completed_with_errors", "model": config["model"]["name_or_path"], "requested_tasks": tasks, "successful_inferences": success, "failed_inferences": failed, "fake_outputs_created": False}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
