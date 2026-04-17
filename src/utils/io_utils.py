from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder: str | Path, allowed_exts: Iterable[str] | None = None) -> List[Path]:
    """List image files in a folder, case-insensitive by suffix."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    allowed = {ext.lower() for ext in (allowed_exts or VALID_IMAGE_EXTS)}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in allowed]
    return sorted(files)


def read_image_bgr(path: str | Path) -> np.ndarray:
    """Read an image in BGR format using OpenCV."""
    path = str(path)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def save_image_bgr(path: str | Path, image: np.ndarray) -> None:
    """Save an image in BGR format using OpenCV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise ValueError(f"Failed to save image: {path}")
