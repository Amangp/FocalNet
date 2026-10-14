
"""Dual-Domain Selection Mechanism (DSM) for FocalNet."""

import torch
import torch.nn as nn

from .ssm import SSM
from .fsm import FSM


class DSM(nn.Module):
    """Apply spatial selection followed by frequency selection."""

    def __init__(self, channels: int):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        self.ssm = SSM(channels)
        self.fsm = FSM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ssm(x)
        x = self.fsm(x)
        return x
