"""Enhance one image or a directory with pretrained FGDPA; save a run manifest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, ImageOps, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aqua_align.modern_enhancement import FGDPAEnhancer, DEFAULT_CHECKPOINT, UPSTREAM_COMMIT
from aqua_align.image_quality import psnr, ssim

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def comparison(images: list[Image.Image], labels: list[str]) -> Image.Image:
    thumbs = []
    for source in images:
        thumb = source.copy()
        thumb.thumbnail((480, 360))
        thumbs.append(thumb)
    canvas = Image.new("RGB", (480 * len(images), 392), "#101b26")
    draw = ImageDraw.Draw(canvas)
    for index, (thumb, label) in enumerate(zip(thumbs, labels)):
        x = index * 480
        draw.text((x + 12, 10), label, fill="white")
        canvas.paste(thumb, (x + (480 - thumb.width) // 2, 32 + (360 - thumb.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/fgdpa")
    parser.add_argument("--reference-dir", type=Path, help="Optional references with identical relative filenames")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit", type=int, default=0, help="0 processes all images")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.limit < 0 or args.threads < 1:
        parser.error("limit must be nonnegative and threads must be positive")
    source = args.input.resolve()
    output = args.output_dir.resolve()
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if output == source or source in output.parents:
        parser.error("Output must be outside the input directory")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is nonempty; choose a new directory to preserve existing results")
    base = source if source.is_dir() else source.parent
    paths = sorted(p for p in (source.rglob("*") if source.is_dir() else [source])
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("No supported images found")
    torch.set_num_threads(args.threads)
    enhancer = FGDPAEnhancer(args.checkpoint, args.device)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"method": "FGDPA", "upstream_commit": UPSTREAM_COMMIT,
                "checkpoint_sha256": enhancer.checkpoint_sha256,
                "torch": torch.__version__, "device": str(enhancer.device),
                "parameters": sum(p.numel() for p in enhancer.model.parameters()),
                "protocol": "Full-resolution RGB [0,1]; FP32; output clamped and rounded to uint8; no tiling",
                "evaluation_note": "Demonstration only: UIEB checkpoint training overlap has not been excluded.",
                "status": "running", "images": []}
    manifest_path = output / "run_summary.json"

    def save_manifest() -> None:
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    save_manifest()
    try:
        for path in paths:
            relative = path.relative_to(base)
            with Image.open(path) as opened:
                original = ImageOps.exif_transpose(opened).convert("RGB")
            start = time.perf_counter()
            enhanced = enhancer.enhance(original)
            seconds = time.perf_counter() - start
            # Retain the original extension in the name to avoid jpg/png stem collisions.
            destination = output / "enhanced" / relative.parent / (relative.name + ".png")
            destination.parent.mkdir(parents=True, exist_ok=True)
            enhanced.save(destination)
            row = {"input": str(path), "output": str(destination), "size": list(original.size), "seconds": seconds}
            images, labels = [original, enhanced], ["Input", "FGDPA (pretrained)"]
            if args.reference_dir:
                reference_path = args.reference_dir / relative
                if not reference_path.is_file():
                    raise FileNotFoundError(f"Missing reference: {reference_path}")
                with Image.open(reference_path) as opened:
                    reference = ImageOps.exif_transpose(opened).convert("RGB")
                for name, img in [("input", original), ("enhanced", enhanced)]:
                    row[f"{name}_psnr"] = psnr(np.asarray(img), np.asarray(reference))
                    row[f"{name}_ssim"] = ssim(np.asarray(img), np.asarray(reference))
                images.append(reference)
                labels.append("Reference")
            preview = output / "comparisons" / relative.parent / (relative.name + ".jpg")
            preview.parent.mkdir(parents=True, exist_ok=True)
            comparison(images, labels).save(preview, quality=92)
            row["comparison"] = str(preview)
            manifest["images"].append(row)
            save_manifest()
            print(f"Saved {relative} ({seconds:.3f}s)", flush=True)
        manifest["status"] = "complete"
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        raise
    finally:
        save_manifest()
    print(f"Run manifest: {manifest_path}")


if __name__ == "__main__":
    main()
