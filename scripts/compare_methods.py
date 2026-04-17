from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.file_scan import scan_images
from utils.image_ops import ensure_dir, make_comparison


def find_result(result_dir: Path, stem: str) -> Image.Image | None:
    for suffix in ["_enhanced.jpg", "_enhanced.png", ".jpg", ".png"]:
        p = result_dir / f"{stem}{suffix}"
        if p.exists():
            return Image.open(p).convert("RGB")
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--baseline_dir", type=str, required=True)
    parser.add_argument("--physics_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results" / "comparisons"))
    args = parser.parse_args()

    out_dir = ensure_dir(args.output_dir)
    baseline_dir = Path(args.baseline_dir)
    physics_dir = Path(args.physics_dir)
    for p in scan_images(args.input_dir):
        raw = Image.open(p).convert("RGB")
        b = find_result(baseline_dir, p.stem)
        g = find_result(physics_dir, p.stem)
        if b is None or g is None:
            print(f"Skip {p.name}: missing result")
            continue
        b = b.resize(raw.size, Image.BICUBIC) if b.size != raw.size else b
        g = g.resize(raw.size, Image.BICUBIC) if g.size != raw.size else g
        make_comparison([raw, b, g], ["input", "baseline", "physics-guided"]).save(out_dir / f"{p.stem}_compare_methods.jpg", quality=95)
        print(f"Saved comparison: {p.name}")


if __name__ == "__main__":
    main()
