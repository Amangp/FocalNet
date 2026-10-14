
"""Three-scale decoder for FocalNet."""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resblock import ResBlock
from .mresblock import MResBlock


class Decoder(nn.Module):
    """Reconstruct full-resolution features using skip connections."""

    def __init__(
        self,
        channels: Sequence[int] = (32, 64, 128),
        num_blocks: int = 4,
    ):
        super().__init__()

        if len(channels) != 3:
            raise ValueError("channels must contain exactly three values")

        if min(channels) <= 0 or num_blocks < 1:
            raise ValueError("channels and num_blocks must be positive")

        c1, c2, c3 = channels

        # Decode from scale 3 to scale 2.
        self.decode2 = nn.Sequential(
            nn.Conv2d(c3 + c2, c2, kernel_size=1),
            *[ResBlock(c2) for _ in range(num_blocks)],
        )

        # Decode from scale 2 to scale 1.
        self.decode1 = nn.Sequential(
            nn.Conv2d(c2 + c1, c1, kernel_size=1),
            MResBlock(c1, c1, num_blocks),
        )

    def forward(
        self,
        f1: torch.Tensor,
        f2: torch.Tensor,
        f3: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-resolution features with shape (B, c1, H, W)."""

        # Upsample bottleneck features to the second scale.
        x = F.interpolate(
            f3,
            size=f2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, f2], dim=1)
        x = self.decode2(x)

        # Upsample to the first scale and merge the skip connection.
        x = F.interpolate(
            x,
            size=f1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        x = torch.cat([x, f1], dim=1)
        x = self.decode1(x)

        return x
