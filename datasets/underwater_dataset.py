from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, random_split

from utils.file_scan import scan_images


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class UnderwaterImageDataset(Dataset):
    """Patch-based dataset for high-resolution unpaired underwater images."""

    def __init__(
        self,
        input_dir: str | Path,
        target_dir: Optional[str | Path] = None,
        patch_size: int = 256,
        training: bool = True,
        augment: bool = True,
        max_images: Optional[int] = None,
    ) -> None:
        self.input_paths = scan_images(input_dir)
        if max_images:
            self.input_paths = self.input_paths[:max_images]
        if not self.input_paths:
            raise FileNotFoundError(f"No images found under: {input_dir}")

        self.target_dir = Path(target_dir) if target_dir else None
        self.target_paths: Dict[str, Path] = {}
        if self.target_dir:
            for p in scan_images(self.target_dir):
                self.target_paths[p.name.lower()] = p

        self.patch_size = patch_size
        self.training = training
        self.augment = augment and training

    def __len__(self) -> int:
        return len(self.input_paths)

    def _load_pair(self, path: Path) -> Tuple[Image.Image, Optional[Image.Image]]:
        image = Image.open(path).convert("RGB")
        target = None
        if self.target_dir:
            target_path = self.target_paths.get(path.name.lower())
            if target_path is not None:
                target = Image.open(target_path).convert("RGB")
                if target.size != image.size:
                    target = target.resize(image.size, Image.BICUBIC)
        return image, target

    def _crop(self, image: Image.Image, target: Optional[Image.Image]) -> Tuple[Image.Image, Optional[Image.Image]]:
        w, h = image.size
        ps = self.patch_size
        if min(w, h) < ps:
            scale = ps / float(min(w, h))
            nw, nh = max(ps, int(round(w * scale))), max(ps, int(round(h * scale)))
            image = image.resize((nw, nh), Image.BICUBIC)
            if target is not None:
                target = target.resize((nw, nh), Image.BICUBIC)
            w, h = image.size

        if self.training:
            x = random.randint(0, w - ps)
            y = random.randint(0, h - ps)
        else:
            x = max(0, (w - ps) // 2)
            y = max(0, (h - ps) // 2)
        box = (x, y, x + ps, y + ps)
        image = image.crop(box)
        target = target.crop(box) if target is not None else None
        return image, target

    def _augment(self, image: Image.Image, target: Optional[Image.Image]) -> Tuple[Image.Image, Optional[Image.Image]]:
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
            target = ImageOps.mirror(target) if target is not None else None
        if random.random() < 0.2:
            image = ImageOps.flip(image)
            target = ImageOps.flip(target) if target is not None else None
        if random.random() < 0.5:
            k = random.choice([0, 1, 2, 3])
            if k:
                image = image.rotate(90 * k, expand=True)
                target = target.rotate(90 * k, expand=True) if target is not None else None
        return image, target

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        path = self.input_paths[idx]
        image, target = self._load_pair(path)
        image, target = self._crop(image, target)
        if self.augment:
            image, target = self._augment(image, target)
        sample: Dict[str, torch.Tensor | str] = {"input": pil_to_tensor(image), "path": str(path)}
        if target is not None:
            sample["target"] = pil_to_tensor(target)
        return sample


def create_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    data_cfg = cfg.get("data", {})
    train_set = UnderwaterImageDataset(
        input_dir=data_cfg["input_dir"],
        target_dir=data_cfg.get("target_dir"),
        patch_size=int(data_cfg.get("patch_size", 256)),
        training=True,
        augment=bool(data_cfg.get("augment", True)),
        max_images=data_cfg.get("max_images"),
    )
    val_ratio = float(data_cfg.get("val_ratio", 0.1))
    val_len = max(1, int(len(train_set) * val_ratio)) if len(train_set) > 1 else 1
    train_len = max(1, len(train_set) - val_len)
    if train_len + val_len > len(train_set):
        val_len = len(train_set) - train_len
    generator = torch.Generator().manual_seed(int(cfg.get("seed", 42)))
    train_subset, val_subset = random_split(train_set, [train_len, val_len], generator=generator)

    loader_cfg = cfg.get("loader", {})
    train_loader = DataLoader(
        train_subset,
        batch_size=int(loader_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=int(loader_cfg.get("num_workers", 2)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=train_len > 1,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=1,
        shuffle=False,
        num_workers=int(loader_cfg.get("num_workers", 2)),
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
    )
    return train_loader, val_loader
