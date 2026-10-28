"""FocalNet building blocks.

Implements the paper's core design ideas: residual blocks, a high-resolution
multi-scale residual block, and the dual-domain selection mechanism (SSM + FSM).
This is a clean-room implementation draft; compare exact layer widths/operation
ordering against the authors' repository before claiming bit-for-bit reproduction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels: int, res_scale: float = 1.0):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.body(x) * self.res_scale


class ResBlockGroup(nn.Module):
    def __init__(self, channels: int, num_blocks: int = 4):
        super().__init__()
        self.blocks = nn.Sequential(*(ResBlock(channels) for _ in range(num_blocks)))

    def forward(self, x):
        return self.blocks(x)


class MResBlock(nn.Module):
    """Multi-scale residual block used at high-resolution encoder/decoder stages.

    Splits channels into two groups. One group keeps full resolution; the other
    is downsampled, processed, then resized back before fusion.
    """
    def __init__(self, channels: int):
        super().__init__()
        if channels < 4 or channels % 2:
            raise ValueError("MResBlock channels must be an even number >= 4")
        half = channels // 2
        self.local = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, padding=1),
        )
        self.down = nn.Conv2d(half, half, 3, stride=2, padding=1)
        self.coarse = nn.Sequential(
            nn.Conv2d(half, half, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(half, half, 3, padding=1),
        )
        self.fuse = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=1)
        a = a + self.local(a)
        size = b.shape[-2:]
        b = self.down(b)
        b = b + self.coarse(b)
        b = F.interpolate(b, size=size, mode="bilinear", align_corners=False)
        return x + self.fuse(torch.cat((a, b), dim=1))


class SSM(nn.Module):
    """Spatial Selection Module: predicts a spatial degradation-selection map."""
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.dw5 = nn.Conv2d(channels, channels, 5, padding=4, dilation=2,
                             groups=channels)
        self.dw7 = nn.Conv2d(channels, channels, 7, padding=9, dilation=3,
                             groups=channels)
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        mean_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        mask = torch.sigmoid(self.spatial(torch.cat((mean_map, max_map), dim=1)))
        a = self.dw5(x * mask)
        b = self.dw7(x * mask)
        c = self.dw3(x)
        return x + self.project(a + b + c)


class FSM(nn.Module):
    """Frequency Selection Module: emphasizes high-frequency feature content."""
    def forward(self, x):
        low_frequency = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        high_frequency = x - low_frequency
        # Residual modulation keeps the module numerically stable at initialization.
        return x + high_frequency * high_frequency


class DSM(nn.Module):
    """Dual-domain Selection Mechanism: spatial selection followed by frequency selection."""
    def __init__(self, channels: int):
        super().__init__()
        self.ssm = SSM(channels)
        self.fsm = FSM()

    def forward(self, x):
        return self.fsm(self.ssm(x))
