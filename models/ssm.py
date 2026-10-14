
"""Spatial Selection Module (SSM) for FocalNet."""

import torch
import torch.nn as nn


class SSM(nn.Module):
    """Emphasize spatial regions that need restoration."""

    def __init__(self, channels: int):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive")

        # Generate a one-channel spatial selection map.
        self.spatial_conv = nn.Conv2d(
            2, 1, kernel_size=3, padding=1
        )

        # Channel-wise spatial feature transformations.
        self.depthwise5 = nn.Conv2d(
            channels, channels, kernel_size=5,
            padding=2, groups=channels
        )
        self.depthwise7 = nn.Conv2d(
            channels, channels, kernel_size=7,
            padding=3, groups=channels
        )
        self.depthwise3 = nn.Conv2d(
            channels, channels, kernel_size=3,
            padding=1, groups=channels
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x has shape (B, C, H, W).

        # Pool across channels to locate important spatial regions.
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map = torch.amax(x, dim=1, keepdim=True)

        spatial_map = torch.cat([avg_map, max_map], dim=1)
        spatial_map = self.spatial_conv(spatial_map)

        # Apply cascaded depthwise convolutions.
        selected = self.depthwise5(x)
        selected = self.depthwise7(selected)

        # Modulate with the spatial map and add local features.
        selected = selected * spatial_map + self.depthwise3(x)

        return selected
