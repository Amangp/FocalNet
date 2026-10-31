"""Paper-aligned FocalNet building blocks for image restoration."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class ResBlockGroup(nn.Module):
    def __init__(self, channels, num_res=4):
        super().__init__()
        self.blocks = nn.Sequential(
            *[ResBlock(channels) for _ in range(num_res)]
        )

    def forward(self, x):
        return self.blocks(x)


class MResBlock(nn.Module):
    """Two-branch multi-scale residual block."""

    def __init__(self, channels):
        super().__init__()

        if channels % 2 != 0:
            raise ValueError("MResBlock channels must be divisible by 2.")

        half = channels // 2

        # Branch 1: original spatial resolution
        self.local = ResBlock(half)

        # Branch 2: reduce spatial resolution to one-quarter
        self.down1 = nn.Conv2d(half, half, 3, stride=2, padding=1)
        self.down2 = nn.Conv2d(half, half, 3, stride=2, padding=1)
        self.multi_scale = ResBlock(half)

    def forward(self, x):
        first, second = torch.chunk(x, 2, dim=1)

        local_features = self.local(first)

        multi_features = self.down1(second)
        multi_features = self.down2(multi_features)
        multi_features = self.multi_scale(multi_features)

        multi_features = F.interpolate(
            multi_features,
            size=second.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return torch.cat([local_features, multi_features], dim=1)


class SSM(nn.Module):
    """Spatial Selection Module."""

    def __init__(self, channels):
        super().__init__()

        # Average/max channel pooling followed by a 3x3 convolution
        self.spatial_map = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        # Main branch: cascaded depth-wise convolutions
        self.dw5 = nn.Conv2d(
            channels, channels, kernel_size=5,
            padding=2, groups=channels
        )
        self.dw7 = nn.Conv2d(
            channels, channels, kernel_size=7,
            padding=3, groups=channels
        )

        # Separate residual branch
        self.dw3 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, groups=channels
        )

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        maximum = x.amax(dim=1, keepdim=True)

        spatial_attention = torch.sigmoid(
            self.spatial_map(torch.cat([avg, maximum], dim=1))
        )

        main = self.dw5(x)
        main = self.dw7(main)

        selected = main * spatial_attention

        residual = self.dw3(x)

        return selected + residual


class FSM(nn.Module):
    """Frequency Selection Module."""

    def forward(self, x):
        low_frequency = x.mean(dim=(-2, -1), keepdim=True)
        high_frequency = x - low_frequency

        return high_frequency * x + x


class DSM(nn.Module):
    """Sequential dual-domain selection: FSM(SSM(x))."""

    def __init__(self, channels):
        super().__init__()
        self.ssm = SSM(channels)
        self.fsm = FSM()

    def forward(self, x):
        spatial_features = self.ssm(x)
        return self.fsm(spatial_features)
