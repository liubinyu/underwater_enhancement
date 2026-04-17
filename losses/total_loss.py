from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .color_loss import ColorConstancyLoss, ExposureLoss
from .physics_loss import PhysicsConsistencyLoss
from .structure_loss import EdgeDetailLoss, GradientConsistencyLoss, SSIMLoss, TotalVariationLoss


def charbonnier(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return torch.sqrt((pred - target).pow(2) + eps * eps).mean()


class EnhancementLoss(nn.Module):
    """Weighted total loss for unpaired and optional paired training."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        weights = cfg.get("loss", {}).get("weights", cfg.get("loss_weights", {}))
        self.weights = {
            "color": float(weights.get("color", 1.0)),
            "exposure": float(weights.get("exposure", 1.0)),
            "structure": float(weights.get("structure", 1.0)),
            "edge": float(weights.get("edge", 0.5)),
            "tv": float(weights.get("tv", 0.05)),
            "physics": float(weights.get("physics", 0.2)),
            "supervised_l1": float(weights.get("supervised_l1", 0.0)),
            "supervised_charbonnier": float(weights.get("supervised_charbonnier", 0.0)),
            "supervised_ssim": float(weights.get("supervised_ssim", 0.0)),
        }
        loss_cfg = cfg.get("loss", {})
        self.color = ColorConstancyLoss()
        self.exposure = ExposureLoss(target=float(loss_cfg.get("exposure_target", 0.55)))
        self.structure = GradientConsistencyLoss()
        self.edge = EdgeDetailLoss()
        self.tv = TotalVariationLoss()
        self.physics = PhysicsConsistencyLoss()
        self.ssim = SSIMLoss()

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        source: torch.Tensor,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred = outputs["final_enhanced"]
        terms: dict[str, torch.Tensor] = {
            "color": self.color(pred),
            "exposure": self.exposure(pred),
            "structure": self.structure(pred, source),
            "edge": self.edge(pred, source),
            "tv": self.tv(pred),
            "physics": self.physics(outputs),
        }
        if target is not None:
            terms["supervised_l1"] = F.l1_loss(pred, target)
            terms["supervised_charbonnier"] = charbonnier(pred, target)
            terms["supervised_ssim"] = self.ssim(pred, target)

        total = torch.tensor(0.0, device=pred.device)
        logs: dict[str, float] = {}
        for name, value in terms.items():
            weight = self.weights.get(name, 0.0)
            if weight > 0:
                total = total + weight * value
            logs[name] = float(value.detach().cpu())
        logs["total"] = float(total.detach().cpu())
        return total, logs

