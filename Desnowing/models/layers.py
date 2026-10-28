"""Paper-oriented building blocks for FocalNet image restoration.

This module follows the design described in:
Cui et al., "Focal Network for Image Restoration", ICCV 2023.

It intentionally implements the paper's described blocks rather than copying
helper modules from the authors' public code repository.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ConvLayer(nn.Module):
    """Convolution layer used for encoder/decoder sampling and feature mapping."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        *,
        transpose: bool = False,
        activation: bool = False,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size <= 0 or stride <= 0:
            raise ValueError("kernel_size and stride must be positive")

        if transpose:
            # For kernel_size=4, stride=2, padding=1, spatial size doubles.
            padding = (kernel_size - stride) // 2
            conv: nn.Module = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        else:
            padding = kernel_size // 2
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )

        modules: list[nn.Module] = [conv]
        if activation:
            modules.append(nn.GELU())
        self.block = nn.Sequential(*modules)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualUnit(nn.Module):
    """One residual unit: 3x3 conv -> GELU -> 3x3 conv -> identity addition."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.activation(self.conv1(x)))


class ResBlock(nn.Module):
    """A ResBlock containing n residual units, as described in the paper."""

    def __init__(self, channels: int, num_residual_units: int = 4) -> None:
        super().__init__()
        if num_residual_units < 1:
            raise ValueError("num_residual_units must be at least 1")
        self.units = nn.Sequential(
            *(ResidualUnit(channels) for _ in range(num_residual_units))
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.units(x)


class MResBlock(nn.Module):
    """Multi-scale ResBlock from the paper.

    The channels are split equally. One half is processed at the original
    resolution; the other half is downsampled by stride-2 convolution, processed
    by a ResBlock, and bilinearly upsampled back to the original H x W. Outputs
    of the branches are concatenated along the channel dimension.
    """

    def __init__(self, channels: int, num_residual_units: int = 4) -> None:
        super().__init__()
        if channels % 2 != 0:
            raise ValueError("MResBlock requires an even channel count")

        branch_channels = channels // 2
        self.full_resolution = ResBlock(branch_channels, num_residual_units)
        self.downsample = nn.Conv2d(
            branch_channels, branch_channels, kernel_size=3, stride=2, padding=1
        )
        self.low_resolution = ResBlock(branch_channels, num_residual_units)

    def forward(self, x: Tensor) -> Tensor:
        x_full, x_low = torch.chunk(x, chunks=2, dim=1)
        y_full = self.full_resolution(x_full)
        y_low = self.downsample(x_low)
        y_low = self.low_resolution(y_low)
        y_low = F.interpolate(
            y_low, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return torch.cat((y_full, y_low), dim=1)


class DualDomainSelectionMechanism(nn.Module):
    """Paper's DSM: spatial selection followed by frequency selection.

    SSM:
      F' = Conv3([AvgPool_channel(F), MaxPool_channel(F)])
      Fs = DWConv7(DWConv5(F)) * Tile(F', C) + DWConv3(F)

    FSM:
      Fh = Fs - spatial_global_mean(Fs)
      Fhat = Fh * Fs + Fs

    The depth-wise kernels here use the kernel sizes stated in the paper without
    adding dilation factors that are not specified in the paper description.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.spatial_map = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.depthwise_5 = nn.Conv2d(
            channels, channels, kernel_size=5, padding=2, groups=channels
        )
        self.depthwise_7 = nn.Conv2d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.depthwise_3 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels
        )

    def forward(self, x: Tensor) -> Tensor:
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)
        spatial_map = self.spatial_map(torch.cat((avg_map, max_map), dim=1))

        channel_features = self.depthwise_7(self.depthwise_5(x))
        spatially_selected = channel_features * spatial_map + self.depthwise_3(x)

        # Spatial global average pooled features are broadcast back to H x W.
        low_frequency = spatially_selected.mean(dim=(-2, -1), keepdim=True)
        high_frequency = spatially_selected - low_frequency
        return high_frequency * spatially_selected + spatially_selected
