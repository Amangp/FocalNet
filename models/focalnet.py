"""Main FocalNet model for image restoration.

Architecture follows the main design in:
    Cui et al., "Focal Network for Image Restoration", ICCV 2023.

This file orchestrates the encoder, bottleneck DSM, and decoder. The component
modules live in separate files so team members can implement and test them
independently.

Required component contracts
----------------------------
Encoder(in_channels: int, channels: Sequence[int], num_blocks: int)
    forward(x) -> (f1, f2, f3)
    f1: (B, channels[0], H, W)
    f2: (B, channels[1], H/2, W/2)
    f3: (B, channels[2], H/4, W/4)

DSM(channels: int)
    forward(f3) -> tensor with the same shape as f3
    DSM must apply SSM first, then FSM.

Decoder(channels: Sequence[int], num_blocks: int)
    forward(f1, f2, f3) -> full-resolution features
    output shape: (B, channels[0], H, W)

The encoder/decoder component implementations should use MResBlock at the
first (highest-resolution) scale and ResBlock at the other scales, as described
in the paper. Exact channel widths and block counts are configurable here.
"""

from typing import Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

# These modules are intentionally separate. Their public constructors and
# forward shapes must match the contracts documented above.
from .encoder import Encoder
from .decoder import Decoder
from .dsm import DSM


class FocalNet(nn.Module):
    """Three-scale FocalNet for image restoration.

    Args:
        in_channels: Number of channels in the degraded input image.
        out_channels: Number of channels in the restored image.
        channels: Feature channels at the three encoder/decoder scales.
        num_blocks: Number of residual blocks per ordinary ResBlock stage.

    Notes:
        The paper uses an image-level residual connection, so the default
        restoration task expects in_channels == out_channels.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        channels: Sequence[int] = (32, 64, 128),
        num_blocks: int = 4,
    ) -> None:
        super().__init__()

        if len(channels) != 3:
            raise ValueError("channels must contain exactly three scale widths")
        if any((not isinstance(c, int)) or c <= 0 for c in channels):
            raise ValueError("all channel widths must be positive integers")
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be a positive integer")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)

        # Extract shallow features at the original image resolution.
        self.shallow = nn.Conv2d(
            in_channels, self.channels[0], kernel_size=3, stride=1, padding=1
        )

        # Encoder returns three feature maps for the symmetric decoder.
        self.encoder = Encoder(
            in_channels=self.channels[0],
            channels=self.channels,
            num_blocks=num_blocks,
        )

        # DSM is placed at the bottleneck, and its order is SSM -> FSM.
        self.dsm = DSM(channels=self.channels[2])

        # Decoder merges encoder features through skip connections.
        self.decoder = Decoder(channels=self.channels, num_blocks=num_blocks)

        # Predict the residual image that is added to the degraded input.
        self.reconstruction = nn.Conv2d(
            self.channels[0], out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x: Tensor) -> Tensor:
        """Restore an image tensor of shape (B, C, H, W)."""
        if x.ndim != 4:
            raise ValueError(
                f"Expected input with shape (B, C, H, W), got {tuple(x.shape)}"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        original_h, original_w = x.shape[-2:]

        # Two downsampling stages require dimensions divisible by 4. Pad only
        # on the right/bottom and crop the final prediction back to input size.
        pad_h = (4 - original_h % 4) % 4
        pad_w = (4 - original_w % 4) % 4
        if pad_h or pad_w:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_pad = x

        shallow = self.shallow(x_pad)
        f1, f2, f3 = self.encoder(shallow)

        f3 = self.dsm(f3)
        decoded = self.decoder(f1, f2, f3)
        residual = self.reconstruction(decoded)

        # Image-level residual connection used by the paper for restoration.
        if self.in_channels == self.out_channels:
            restored = x_pad + residual
        else:
            restored = residual

        return restored[..., :original_h, :original_w]


if __name__ == "__main__":
    # Smoke test becomes runnable once encoder.py, decoder.py and dsm.py exist.
    model = FocalNet()
    sample = torch.randn(2, 3, 256, 256)
    output = model(sample)
    print("Input shape: ", tuple(sample.shape))
    print("Output shape:", tuple(output.shape))

