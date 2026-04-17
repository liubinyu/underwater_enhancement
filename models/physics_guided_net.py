from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .modules import LightEncoderDecoder, ResidualDSBlock


class BackgroundLightEstimator(nn.Module):
    """Estimate low-frequency/global background light A in RGB."""

    def __init__(self, base_ch: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            ResidualDSBlock(base_ch),
            nn.Conv2d(base_ch, base_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            ResidualDSBlock(base_ch),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(base_ch, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class TransmissionEstimator(nn.Module):
    """Estimate per-pixel transmission map t(x)."""

    def __init__(self, base_ch: int = 16, min_t: float = 0.08) -> None:
        super().__init__()
        self.min_t = min_t
        self.net = nn.Sequential(
            nn.Conv2d(3, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            ResidualDSBlock(base_ch),
            nn.Conv2d(base_ch, base_ch, 3, padding=1, groups=base_ch, bias=False),
            nn.Conv2d(base_ch, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.min_t + (1.0 - self.min_t) * self.net(x)


def coarse_restore(x: torch.Tensor, a: torch.Tensor, t: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    j = (x - a) / torch.clamp(t, min=eps) + a
    return torch.clamp(j, 0.0, 1.0)


class PhysicsGuidedEnhancementNet(nn.Module):
    """Physical prior coarse restoration followed by lightweight refinement."""

    def __init__(self, base_ch: int = 32, prior_ch: int = 16, min_t: float = 0.08, residual_scale: float = 0.15) -> None:
        super().__init__()
        self.a_estimator = BackgroundLightEstimator(base_ch=prior_ch)
        self.t_estimator = TransmissionEstimator(base_ch=prior_ch, min_t=min_t)
        self.refinement = LightEncoderDecoder(in_ch=7, out_ch=3, base_ch=base_ch)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        a = self.a_estimator(x)
        t = self.t_estimator(x)
        coarse = coarse_restore(x, a, t)
        a_map = a.expand(-1, -1, x.shape[-2], x.shape[-1])
        refine_in = torch.cat([x, coarse, t], dim=1)
        residual = torch.tanh(self.refinement(refine_in)) * self.residual_scale
        final = torch.clamp(coarse + residual, 0.0, 1.0)
        return {
            "estimated_A": a,
            "estimated_A_map": a_map,
            "estimated_t": t,
            "coarse_restored": coarse,
            "final_enhanced": final,
            "enhanced": final,
        }

