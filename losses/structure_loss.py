from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def gradient_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return gx, gy


class GradientConsistencyLoss(nn.Module):
    def forward(self, pred: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        pgx, pgy = gradient_xy(pred)
        sgx, sgy = gradient_xy(src)
        return F.l1_loss(pgx, sgx) + F.l1_loss(pgy, sgy)


class EdgeDetailLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def _edges(self, x: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        gx = F.conv2d(gray, self.kx, padding=1)
        gy = F.conv2d(gray, self.ky, padding=1)
        return torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-6)

    def forward(self, pred: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(self._edges(pred), self._edges(src))


class TotalVariationLoss(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx, gy = gradient_xy(x)
        return gx.abs().mean() + gy.abs().mean()


class SSIMLoss(nn.Module):
    """Small differentiable SSIM loss for optional supervised training."""

    def __init__(self, window: int = 7) -> None:
        super().__init__()
        self.window = window

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        pad = self.window // 2
        mu_x = F.avg_pool2d(pred, self.window, 1, pad)
        mu_y = F.avg_pool2d(target, self.window, 1, pad)
        sigma_x = F.avg_pool2d(pred * pred, self.window, 1, pad) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(target * target, self.window, 1, pad) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(pred * target, self.window, 1, pad) - mu_x * mu_y
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + 1e-8
        )
        return torch.clamp((1 - ssim.mean()) * 0.5, 0, 1)

