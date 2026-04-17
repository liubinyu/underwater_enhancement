from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import build_model
from utils.file_scan import scan_images
from utils.image_ops import ensure_dir, load_image, make_comparison, tensor_to_image, tile_inference
from utils.metrics import compute_metrics
from utils.vis import save_background_light, save_tensor_image, save_transmission_map


def load_cfg(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(config: dict, checkpoint: str, device: torch.device):
    model = build_model(config["model"], config).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(ROOT / "configs" / "physics_guided.yaml"))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=str(ROOT / "results" / "inference"))
    parser.add_argument("--tile_size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_metrics", action="store_true")
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.checkpoint, device)
    in_path = Path(args.input)
    paths = scan_images(in_path)
    if not paths:
        raise FileNotFoundError(f"No images found: {args.input}")

    out_root = ensure_dir(args.output_dir)
    metric_rows = []
    for p in paths:
        img = load_image(p)
        out = tile_inference(model, img, device, tile_size=args.tile_size, overlap=args.overlap, amp=bool(cfg.get("train", {}).get("amp", True)))
        stem = p.stem
        enhanced = tensor_to_image(out["final_enhanced"])
        enhanced.save(out_root / f"{stem}_enhanced.jpg", quality=95)
        make_comparison([img, enhanced], ["input", "enhanced"]).save(out_root / f"{stem}_compare.jpg", quality=95)
        if "coarse_restored" in out:
            save_tensor_image(out["coarse_restored"], out_root / f"{stem}_coarse.jpg")
        if "estimated_t" in out:
            save_transmission_map(out["estimated_t"], out_root / f"{stem}_transmission.png")
        if "estimated_A" in out:
            save_background_light(out["estimated_A"], out_root / f"{stem}_A.png")
        if args.save_metrics:
            row = {"file": p.name}
            row.update({f"input_{k}": v for k, v in compute_metrics(img).items()})
            row.update({f"enhanced_{k}": v for k, v in compute_metrics(enhanced).items()})
            metric_rows.append(row)
        print(f"Saved enhanced result for {p.name}")

    if metric_rows:
        import csv

        csv_path = out_root / "metrics.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(metric_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metric_rows)
        print(f"Saved metrics: {csv_path}")


if __name__ == "__main__":
    main()
