from __future__ import annotations

from pathlib import Path
from typing import List


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan_images(root: str | Path) -> List[Path]:
    root = Path(root)
    if root.is_file() and root.suffix.lower() in IMG_EXTS:
        return [root]
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)

