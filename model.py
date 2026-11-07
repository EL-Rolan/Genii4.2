"""PyTorch U-Net that maps four target planes to one phase-only DOE."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        while out_channels % groups != 0:
            groups -= 1
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class PhaseDOEUNet(nn.Module):
    """Input [B,4,H,W], output phase [B,H,W] in [-pi, pi].

    The network predicts sin(phi) and cos(phi), avoiding a discontinuity at
    the phase wrap boundary (-pi and +pi represent the same physical phase).
    """

    def __init__(self, in_channels: int = 4, base_channels: int = 32) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 6
        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(2)
        self.dec3 = ConvBlock(c4 + c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)
        self.phase_head = nn.Conv2d(c1, 2, kernel_size=1)

        # Start close to a flat phase while retaining small trainable gradients.
        nn.init.normal_(self.phase_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.phase_head.bias)
        with torch.no_grad():
            self.phase_head.bias[1] = 1.0  # channel 0=sin, channel 1=cos

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, target_amplitudes: torch.Tensor) -> torch.Tensor:
        if target_amplitudes.ndim != 4 or target_amplitudes.shape[1] != 4:
            raise ValueError("Expected input with shape [B, 4, H, W]")
        if min(target_amplitudes.shape[-2:]) < 8:
            raise ValueError("Input height and width must be at least 8 pixels")

        e1 = self.enc1(target_amplitudes)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self._up(b, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))

        sin_cos = self.phase_head(d1)
        sin_cos = F.normalize(sin_cos, p=2, dim=1, eps=1e-6)
        phase = torch.atan2(sin_cos[:, 0], sin_cos[:, 1])
        return phase
