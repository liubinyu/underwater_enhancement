from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .image_ops import ensure_dir, tensor_to_image


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    tensor_to_image(tensor).save(path)


def save_transmission_map(t: torch.Tensor, path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    arr = t.detach().float().cpu()
    if arr.ndim == 4:
        arr = arr[0, 0]
    plt.figure(figsize=(6, 5))
    plt.imshow(arr.numpy(), cmap="magma", vmin=0, vmax=1)
    plt.axis("off")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def save_background_light(a: torch.Tensor, path: str | Path, size: tuple[int, int] = (240, 120)) -> None:
    ensure_dir(Path(path).parent)
    rgb = a.detach().float().cpu()
    if rgb.ndim == 4:
        rgb = rgb[0, :, 0, 0]
    color = tuple(int(v * 255) for v in rgb.clamp(0, 1).tolist())
    Image.new("RGB", size, color=color).save(path)


def plot_loss_curve(history: list[dict[str, float]], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    if not history:
        return
    plt.figure(figsize=(8, 4))
    xs = list(range(1, len(history) + 1))
    for key in ["train_total", "val_total"]:
        vals = [h.get(key) for h in history]
        if any(v is not None for v in vals):
            plt.plot(xs, vals, label=key)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()

