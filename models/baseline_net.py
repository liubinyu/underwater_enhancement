from __future__ import annotations

import torch
from torch import nn

from .modules import LightEncoderDecoder


class BaselineEnhancementNet(nn.Module):
    """Pure learning-based lightweight enhancement baseline."""

    def __init__(self, base_ch: int = 32, residual_scale: float = 0.2) -> None:
        super().__init__()
        self.net = LightEncoderDecoder(in_ch=3, out_ch=3, base_ch=base_ch)
        self.residual_scale = residual_scale

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        residual = torch.tanh(self.net(x)) * self.residual_scale
        enhanced = torch.clamp(x + residual, 0.0, 1.0)
        return {
            "final_enhanced": enhanced,
            "enhanced": enhanced,
        }

