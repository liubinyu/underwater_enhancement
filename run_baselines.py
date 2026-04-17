from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from src.baselines.clahe_enhance import apply_clahe_lab
from src.baselines.color_compensation import red_channel_compensation
from src.baselines.retinex import multi_scale_retinex
from src.baselines.white_balance import gray_world_white_balance
from src.utils.config_utils import ensure_dir, load_yaml
from src.utils.io_utils import list_images, read_image_bgr, save_image_bgr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run underwater image enhancement baselines.")
    parser.add_argument(
        "--paths",
        type=str,
        default="configs/paths.yaml",
        help="Path to the path config YAML.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="configs/baseline.yaml",
        help="Path to the baseline config YAML.",
    )
    return parser.parse_args()


def prepare_output_dirs(paths_cfg: Dict[str, str]) -> None:
    ensure_dir(paths_cfg["baseline_outputs"])
    ensure_dir(paths_cfg["wb_outputs"])
    ensure_dir(paths_cfg["cc_outputs"])
    ensure_dir(paths_cfg["clahe_outputs"])
    ensure_dir(paths_cfg["retinex_outputs"])


def main() -> None:
    args = parse_args()
    paths_cfg = load_yaml(args.paths)
    baseline_cfg = load_yaml(args.baseline)

    prepare_output_dirs(paths_cfg)

    raw_images_dir = paths_cfg["raw_images"]
    supported_exts: List[str] = baseline_cfg.get("common", {}).get(
        "supported_exts", [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    )
    image_paths = list_images(raw_images_dir, supported_exts)

    if not image_paths:
        print(f"[WARN] No images found in: {raw_images_dir}")
        print("请先把原始图片放到 data/raw/images 目录下。")
        return

    print(f"[INFO] Found {len(image_paths)} images.")

    overwrite = bool(baseline_cfg.get("common", {}).get("overwrite", True))

    wb_cfg = baseline_cfg.get("white_balance", {})
    cc_cfg = baseline_cfg.get("color_compensation", {})
    clahe_cfg = baseline_cfg.get("clahe", {})
    retinex_cfg = baseline_cfg.get("retinex", {})

    for idx, img_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing: {img_path.name}")

        try:
            image = read_image_bgr(img_path)
        except Exception as exc:
            print(f"  [ERROR] Failed to read image: {img_path.name} | {exc}")
            continue

        # White Balance
        if wb_cfg.get("enabled", True):
            out_path = Path(paths_cfg["wb_outputs"]) / img_path.name
            if overwrite or not out_path.exists():
                try:
                    wb_img = gray_world_white_balance(image)
                    save_image_bgr(out_path, wb_img)
                except Exception as exc:
                    print(f"  [ERROR] White balance failed: {img_path.name} | {exc}")

        # Color Compensation
        if cc_cfg.get("enabled", True):
            out_path = Path(paths_cfg["cc_outputs"]) / img_path.name
            if overwrite or not out_path.exists():
                try:
                    cc_img = red_channel_compensation(
                        image,
                        red_gain=float(cc_cfg.get("red_gain", 1.0)),
                        apply_white_balance_after_compensation=bool(
                            cc_cfg.get("apply_white_balance_after_compensation", True)
                        ),
                    )
                    save_image_bgr(out_path, cc_img)
                except Exception as exc:
                    print(f"  [ERROR] Color compensation failed: {img_path.name} | {exc}")

        # CLAHE
        if clahe_cfg.get("enabled", True):
            out_path = Path(paths_cfg["clahe_outputs"]) / img_path.name
            if overwrite or not out_path.exists():
                try:
                    clahe_img = apply_clahe_lab(
                        image,
                        clip_limit=float(clahe_cfg.get("clip_limit", 2.0)),
                        tile_grid_size=int(clahe_cfg.get("tile_grid_size", 8)),
                    )
                    save_image_bgr(out_path, clahe_img)
                except Exception as exc:
                    print(f"  [ERROR] CLAHE failed: {img_path.name} | {exc}")

        # Retinex
        if retinex_cfg.get("enabled", True):
            out_path = Path(paths_cfg["retinex_outputs"]) / img_path.name
            if overwrite or not out_path.exists():
                try:
                    retinex_img = multi_scale_retinex(
                        image,
                        sigmas=retinex_cfg.get("sigmas", [15, 80, 250]),
                        gain=float(retinex_cfg.get("gain", 1.0)),
                        offset=float(retinex_cfg.get("offset", 0.0)),
                    )
                    save_image_bgr(out_path, retinex_img)
                except Exception as exc:
                    print(f"  [ERROR] Retinex failed: {img_path.name} | {exc}")

    print("[INFO] Done.")
    print(f"[INFO] Results saved under: {paths_cfg['baseline_outputs']}")


if __name__ == "__main__":
    main()
