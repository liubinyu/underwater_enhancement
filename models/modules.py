from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Mobile-friendly depthwise + pointwise convolution block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualDSBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSeparableConv(ch, ch),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class ChannelAttention(nn.Module):
    def __init__(self, ch: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, ch // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class LightEncoderDecoder(nn.Module):
    """Small encoder-decoder used by baseline and refinement branches."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base_ch: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_ch),
            nn.SiLU(inplace=True),
            ResidualDSBlock(base_ch),
        )
        self.down1 = DepthwiseSeparableConv(base_ch, base_ch * 2, stride=2)
        self.enc1 = ResidualDSBlock(base_ch * 2)
        self.down2 = DepthwiseSeparableConv(base_ch * 2, base_ch * 4, stride=2)
        self.bottleneck = nn.Sequential(ResidualDSBlock(base_ch * 4), ChannelAttention(base_ch * 4))
        self.up1 = nn.Conv2d(base_ch * 4 + base_ch * 2, base_ch * 2, 1)
        self.dec1 = ResidualDSBlock(base_ch * 2)
        self.up2 = nn.Conv2d(base_ch * 2 + base_ch, base_ch, 1)
        self.dec2 = ResidualDSBlock(base_ch)
        self.head = nn.Conv2d(base_ch, out_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s0 = self.stem(x)
        s1 = self.enc1(self.down1(s0))
        b = self.bottleneck(self.down2(s1))
        u1 = F.interpolate(b, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.dec1(self.up1(torch.cat([u1, s1], dim=1)))
        u2 = F.interpolate(u1, size=s0.shape[-2:], mode="bilinear", align_corners=False)
        u2 = self.dec2(self.up2(torch.cat([u2, s0], dim=1)))
        return self.head(u2)

