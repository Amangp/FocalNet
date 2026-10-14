
"""Basic residual block for FocalNet."""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Two 3x3 convolutions with a residual connection."""

    def __init__(self, channels: int):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        self.conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + residual

        return out
