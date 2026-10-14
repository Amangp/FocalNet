
"""Frequency Selection Module (FSM) for FocalNet."""

import torch
import torch.nn as nn


class FSM(nn.Module):
    """Emphasize high-frequency feature information."""

    def __init__(self, channels: int):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (B, C, H, W).

        # Estimate the lowest-frequency component per channel.
        low_frequency = torch.mean(
            x, dim=(2, 3), keepdim=True
        )

        # Remove the low-frequency component.
        high_frequency = x - low_frequency

        # Frequency selection with a residual connection.
        out = high_frequency * x + x

        return out
