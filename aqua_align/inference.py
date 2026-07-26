"""Lazy, local-only Transformers inference adapter for Qwen3-VL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aqua_align.prompts import SYSTEM_PROMPT, build_prompt, expected_image_count


class BaseVLMRunner:
    """Load a configured local Base VLM and generate deterministic zero-shot outputs."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model = None
        self.processor = None

    def load(self) -> None:
        """Load processor and model without allowing an implicit network download."""
        try:
            import torch
            from transformers import AutoProcessor
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                from transformers import AutoModelForMultimodalLM as ModelClass
        except ImportError as exc:
            raise RuntimeError("Transformers Qwen3-VL dependencies are not installed") from exc
        settings = self.config["model"]
        model_name = settings["name_or_path"]
        local_only = bool(settings.get("local_files_only", True))
        common = {"local_files_only": local_only, "trust_remote_code": bool(settings.get("trust_remote_code", True))}
        self.processor = AutoProcessor.from_pretrained(model_name, **common)
        dtype_name = str(settings.get("torch_dtype", "bfloat16"))
        dtype = getattr(torch, dtype_name, None)
        self.model = ModelClass.from_pretrained(model_name, dtype=dtype, device_map="auto", **common)
        self.model.eval()

    def generate(self, task: str, image_paths: list[str | Path]) -> str:
        """Generate one raw response for a task and its ordered image inputs."""
        if self.model is None or self.processor is None:
            raise RuntimeError("runner.load() must complete before generation")
        if len(image_paths) != expected_image_count(task):
            raise ValueError(f"task {task} requires {expected_image_count(task)} images")
        content = [{"type": "image", "image": str(Path(path).resolve())} for path in image_paths]
        content.append({"type": "text", "text": build_prompt(task)})
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}, {"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)
        generation = dict(self.config.get("generation", {}))
        generation.pop("seed", None)
        if not generation.get("do_sample", False): generation.pop("temperature", None)
        outputs = self.model.generate(**inputs, **generation)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs, strict=True)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

