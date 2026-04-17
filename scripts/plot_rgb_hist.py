from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.file_scan import scan_images
from utils.image_ops import ensure_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=str(ROOT / "data" / "raw" / "images"))
    parser.add_argument("--num_samples", "-n", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results" / "histograms"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    paths = scan_images(args.input_dir)
    if not paths:
        raise FileNotFoundError(f"No images found under {args.input_dir}")
    random.seed(args.seed)
    samples = random.sample(paths, min(args.num_samples, len(paths)))
    out_dir = ensure_dir(args.output_dir)

    hist = {c: np.zeros(256, dtype=np.float64) for c in "RGB"}
    for p in samples:
        arr = np.asarray(Image.open(p).convert("RGB"))
        for i, c in enumerate("RGB"):
            hist[c] += np.bincount(arr[..., i].ravel(), minlength=256)

    plt.figure(figsize=(9, 5))
    colors = {"R": "red", "G": "green", "B": "blue"}
    for c in "RGB":
        y = hist[c] / max(hist[c].sum(), 1.0)
        plt.plot(y, color=colors[c], label=c)
    plt.title(f"RGB histogram, samples={len(samples)}")
    plt.xlabel("pixel value")
    plt.ylabel("frequency")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path = out_dir / "rgb_histogram.png"
    plt.savefig(out_path, dpi=180)
    plt.close()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
