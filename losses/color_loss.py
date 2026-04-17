from __future__ import annotations

import torch
from torch import nn


class ColorConstancyLoss(nn.Module):
    """Gray-world channel balance regularizer."""

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        mean_rgb = image.mean(dim=(2, 3))
        mr, mg, mb = mean_rgb[:, 0], mean_rgb[:, 1], mean_rgb[:, 2]
        return ((mr - mg).pow(2) + (mr - mb).pow(2) + (mg - mb).pow(2)).mean()


class ExposureLoss(nn.Module):
    """Patch-wise exposure target, similar to Zero-DCE's exposure control."""

    def __init__(self, target: float = 0.55, patch_size: int = 16) -> None:
        super().__init__()
        self.target = target
        self.pool = nn.AvgPool2d(patch_size, stride=patch_size)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        luminance = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        return (self.pool(luminance) - self.target).abs().mean()

