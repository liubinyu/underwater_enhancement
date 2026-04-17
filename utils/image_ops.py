from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_image(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_tensor(img: Image.Image, device: torch.device | str = "cpu") -> torch.Tensor:
    import torch

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def tensor_to_image(t: torch.Tensor) -> Image.Image:
    t = t.detach().float().cpu().clamp(0, 1)
    if t.ndim == 4:
        t = t[0]
    arr = (t.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr)


def make_comparison(images: list[Image.Image], labels: list[str] | None = None) -> Image.Image:
    widths, heights = zip(*(im.size for im in images))
    label_h = 28 if labels else 0
    canvas = Image.new("RGB", (sum(widths), max(heights) + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for i, im in enumerate(images):
        canvas.paste(im, (x, label_h))
        if labels:
            draw.text((x + 8, 6), labels[i], fill=(20, 20, 20))
        x += im.size[0]
    return canvas


def tile_inference(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    tile_size: int = 1024,
    overlap: int = 64,
    amp: bool = False,
) -> Dict[str, torch.Tensor]:
    """Run tiled inference and blend overlaps with an averaging weight map."""
    import torch

    with torch.no_grad():
        tensor = image_to_tensor(image, device=device)
        _, c, h, w = tensor.shape
        if tile_size <= 0 or (h <= tile_size and w <= tile_size):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                return model(tensor)

        stride = max(1, tile_size - overlap)
        final = torch.zeros((1, 3, h, w), device=device)
        coarse = torch.zeros_like(final)
        t_map = torch.zeros((1, 1, h, w), device=device)
        a_map = torch.zeros_like(final)
        weight = torch.zeros((1, 1, h, w), device=device)

        ys = list(range(0, max(h - tile_size, 0) + 1, stride))
        xs = list(range(0, max(w - tile_size, 0) + 1, stride))
        if ys[-1] != h - tile_size:
            ys.append(max(h - tile_size, 0))
        if xs[-1] != w - tile_size:
            xs.append(max(w - tile_size, 0))

        for y in ys:
            for x in xs:
                patch = tensor[:, :, y : y + tile_size, x : x + tile_size]
                with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                    out = model(patch)
                ph, pw = patch.shape[-2:]
                final[:, :, y : y + ph, x : x + pw] += out["final_enhanced"]
                if "coarse_restored" in out:
                    coarse[:, :, y : y + ph, x : x + pw] += out["coarse_restored"]
                if "estimated_t" in out:
                    t_map[:, :, y : y + ph, x : x + pw] += out["estimated_t"]
                if "estimated_A_map" in out:
                    a_map[:, :, y : y + ph, x : x + pw] += out["estimated_A_map"]
                weight[:, :, y : y + ph, x : x + pw] += 1.0

        weight = weight.clamp_min(1.0)
        outputs: Dict[str, torch.Tensor] = {"final_enhanced": final / weight}
        if coarse.abs().sum() > 0:
            outputs["coarse_restored"] = coarse / weight
        if t_map.abs().sum() > 0:
            outputs["estimated_t"] = t_map / weight
        if a_map.abs().sum() > 0:
            outputs["estimated_A_map"] = a_map / weight
            outputs["estimated_A"] = outputs["estimated_A_map"].mean(dim=(2, 3), keepdim=True)
        return outputs
