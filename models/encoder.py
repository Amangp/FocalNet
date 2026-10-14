"""Three-scale encoder for FocalNet."""

from typing import Sequence, Tuple

import torch
import torch.nn as nn

from .resblock import ResBlock
from .mresblock import MResBlock


class Encoder(nn.Module):
    """Extract features at three spatial resolutions."""

    def __init__(
        self,
        in_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128),
        num_blocks: int = 4,
    ):
        super().__init__()

        if len(channels) != 3:
            raise ValueError("channels must contain exactly three values")

        if min(channels) <= 0 or num_blocks < 1:
            raise ValueError("channels and num_blocks must be positive")

        c1, c2, c3 = channels

        # Scale 1: highest resolution, using MResBlock.
        self.scale1 = MResBlock(in_channels, c1, num_blocks)

        # Scale 2: half resolution.
        self.down1 = nn.Conv2d(
            c1, c2, kernel_size=3, stride=2, padding=1
        )
        self.scale2 = nn.Sequential(
            *[ResBlock(c2) for _ in range(num_blocks)]
        )

        # Scale 3: quarter resolution.
        self.down2 = nn.Conv2d(
            c2, c3, kernel_size=3, stride=2, padding=1
        )
        self.scale3 = nn.Sequential(
            *[ResBlock(c3) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        f1 = self.scale1(x)
        f2 = self.scale2(self.down1(f1))
        f3 = self.scale3(self.down2(f2))

        return f1, f2, f3
```
