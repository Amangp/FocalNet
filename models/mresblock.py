
"""Multi-scale residual block for FocalNet."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resblock import ResBlock


class MResBlock(nn.Module):
    """Process features at two spatial scales."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 4,
    ):
        super().__init__()

        if in_channels <= 0 or out_channels < 2:
            raise ValueError("Invalid channel count")

        if out_channels % 2 != 0:
            raise ValueError("out_channels must be even")

        if num_blocks < 1:
            raise ValueError("num_blocks must be positive")

        self.out_channels = out_channels
        branch_channels = out_channels // 2

        # Match the input channels to the requested output channels.
        self.input_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=1
        )

        # Branch 1: process features at reduced resolution.
        self.down = nn.Conv2d(
            branch_channels,
            branch_channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.low_blocks = nn.Sequential(
            *[ResBlock(branch_channels) for _ in range(num_blocks)]
        )

        # Branch 2: preserve the original spatial resolution.
        self.high_blocks = nn.Sequential(
            *[ResBlock(branch_channels) for _ in range(num_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_conv(x)

        # Split channels into two equal branches.
        x1, x2 = torch.chunk(x, 2, dim=1)

        # Process the first branch at lower resolution.
        x1 = self.down(x1)
        x1 = self.low_blocks(x1)

        # Restore the original spatial dimensions.
        x1 = F.interpolate(
            x1,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        # Process the second branch at full resolution.
        x2 = self.high_blocks(x2)

        # Join the two branches.
        out = torch.cat([x1, x2], dim=1)

        return out
