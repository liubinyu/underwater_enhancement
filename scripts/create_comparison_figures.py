"""Create non-interactive per-sample overview figures for enhancement candidates."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "outputs" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(ROOT))

from aqua_align.config import resolve_project_path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def create_figures(
    candidates_path: str | Path,
    output_dir: str | Path,
    metrics_path: str | Path | None = None,
    limit: int | None = None,
) -> list[Path]:
    """Render raw, optional reference, and candidates in compact sample grids."""
    candidates = _read_csv(resolve_project_path(candidates_path))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        grouped[row["sample_id"]].append(row)
    metric_lookup: dict[tuple[str, str], dict[str, str]] = {}
    if metrics_path is not None:
        for row in _read_csv(resolve_project_path(metrics_path)):
            metric_lookup[(row["sample_id"], row["method"])] = row
    output = resolve_project_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_ids = sorted(grouped)
    if limit is not None:
        sample_ids = sample_ids[:limit]
    saved: list[Path] = []
    for sample_id in sample_ids:
        rows = grouped[sample_id]
        panels: list[tuple[str, Path, str | None]] = [("raw", resolve_project_path(rows[0]["raw_image"]), None)]
        if rows[0].get("reference_image"):
            panels.append(("reference", resolve_project_path(rows[0]["reference_image"]), None))
        for row in rows:
            metric = metric_lookup.get((sample_id, row["method"]))
            suffix = None
            if metric and metric.get("psnr") and metric.get("ssim"):
                suffix = f"PSNR {float(metric['psnr']):.2f} | SSIM {float(metric['ssim']):.3f}"
            panels.append((row["method"], resolve_project_path(row["candidate_image"]), suffix))

        columns = min(4, len(panels))
        rows_count = math.ceil(len(panels) / columns)
        figure, axes = plt.subplots(rows_count, columns, figsize=(4.2 * columns, 3.7 * rows_count), squeeze=False)
        try:
            for axis, (title, image_path, suffix) in zip(axes.flat, panels, strict=False):
                with Image.open(image_path) as image:
                    axis.imshow(image.convert("RGB"))
                axis.set_title(title if suffix is None else f"{title}\n{suffix}", fontsize=10)
                axis.axis("off")
            for axis in list(axes.flat)[len(panels) :]:
                axis.set_visible(False)
            figure.suptitle(sample_id, fontsize=14)
            figure.tight_layout()
            destination = output / f"{sample_id}_overview.jpg"
            figure.savefig(destination, dpi=140, bbox_inches="tight")
            saved.append(destination)
        finally:
            plt.close(figure)
    return saved


def parse_args() -> argparse.Namespace:
    """Parse comparison figure arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> int:
    """Generate overview figures."""
    args = parse_args()
    try:
        saved = create_figures(args.candidates, args.output_dir, args.metrics, args.limit)
    except Exception as exc:
        print(f"Comparison figure generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {len(saved)} figures to {resolve_project_path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
