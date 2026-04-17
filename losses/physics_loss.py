from __future__ import annotations

import torch
from torch import nn

from .structure_loss import TotalVariationLoss


class PhysicsConsistencyLoss(nn.Module):
    """Regularize physical estimates: smooth t and plausible global A."""

    def __init__(self, min_t: float = 0.05, max_t: float = 1.0) -> None:
        super().__init__()
        self.tv = TotalVariationLoss()
        self.min_t = min_t
        self.max_t = max_t

    def forward(self, outputs: dict[str, torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        if "estimated_t" in outputs:
            t = outputs["estimated_t"]
            loss = loss + self.tv(t)
            loss = loss + torch.relu(self.min_t - t).mean() + torch.relu(t - self.max_t).mean()
        if "estimated_A" in outputs:
            a = outputs["estimated_A"]
            loss = loss + 0.05 * ((a - a.mean(dim=1, keepdim=True)).abs().mean())
        return loss

