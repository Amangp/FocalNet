"""FocalNet desnowing architecture based on the ICCV 2023 paper description.

Reference:
Y. Cui, W. Ren, X. Cao, and A. Knoll,
"Focal Network for Image Restoration," ICCV 2023.

This is an independently written, paper-oriented implementation. It is not a
port of the authors' released FocalNet.py/layers.py code. It contains the
paper's three-scale symmetric encoder-decoder, MResBlock at the first encoder
and decoder scales, and DSM at the bottleneck.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .layers import (
    ConvLayer,
    DualDomainSelectionMechanism,
    MResBlock,
    ResBlock,
)


class FocalNet(nn.Module):
    """Three-scale FocalNet for single-image desnowing.

    Args:
        in_channels: input image channels; RGB images use 3.
        out_channels: restored image channels; RGB images use 3.
        base_channels: C in the paper's notation. The paper describes this as C
            but does not mandate a numeric value in the main architecture text;
            32 is used as a practical default and can be changed.
        num_residual_units: number n of residual units within each ResBlock.
            The paper's implementation details specify n=4 for the denoising
            setting, which includes desnowing.

    Input and output tensors use NCHW layout and are expected to have spatial
    dimensions divisible by four for the two encoder downsampling stages.
    The network predicts a residual image and adds it to the degraded input,
    following the paper's image-level residual connection.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        num_residual_units: int = 4,
    ) -> None:
        super().__init__()
        if base_channels <= 0 or base_channels % 2:
            raise ValueError("base_channels must be a positive even number")
        if in_channels != out_channels:
            raise ValueError(
                "Image-level residual addition requires in_channels == out_channels"
            )

        c = base_channels
        n = num_residual_units

        # Shallow feature extraction: H x W x 3 -> H x W x C.
        self.shallow = ConvLayer(in_channels, c, kernel_size=3, activation=True)

        # Encoder, three feature scales: (H,W,C), (H/2,W/2,2C), (H/4,W/4,4C).
        self.enc_scale1 = MResBlock(c, n)
        self.down1 = ConvLayer(c, 2 * c, kernel_size=3, stride=2, activation=True)
        self.enc_scale2 = ResBlock(2 * c, n)
        self.down2 = ConvLayer(2 * c, 4 * c, kernel_size=3, stride=2, activation=True)
        self.enc_scale3 = ResBlock(4 * c, n)

        # The paper inserts the dual-domain selection mechanism at the bottleneck.
        self.dsm = DualDomainSelectionMechanism(4 * c)
        self.bottleneck_resblock = ResBlock(4 * c, n)

        # Symmetric decoder. Each skip concatenation is followed by 1x1 channel
        # adjustment, then the residual block at that scale.
        self.up2 = ConvLayer(
            4 * c, 2 * c, kernel_size=4, stride=2, transpose=True, activation=True
        )
        self.skip2_adjust = ConvLayer(4 * c, 2 * c, kernel_size=1, activation=True)
        self.dec_scale2 = ResBlock(2 * c, n)

        self.up1 = ConvLayer(
            2 * c, c, kernel_size=4, stride=2, transpose=True, activation=True
        )
        self.skip1_adjust = ConvLayer(2 * c, c, kernel_size=1, activation=True)
        self.dec_scale1 = MResBlock(c, n)

        # Image reconstruction layer; no activation before image-level residual.
        self.reconstruction = ConvLayer(
            c, out_channels, kernel_size=3, activation=False
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected NCHW input with 4 dimensions; got {tuple(x.shape)}")
        if x.shape[1] != self.shallow.block[0].in_channels:
            raise ValueError(
                f"Expected {self.shallow.block[0].in_channels} input channels, "
                f"got {x.shape[1]}"
            )
        if (x.shape[-2] % 4) or (x.shape[-1] % 4):
            raise ValueError(
                "Input height and width must be divisible by 4 for the two "
                "downsampling stages; pad the image before inference if needed."
            )

        shallow = self.shallow(x)

        enc1 = self.enc_scale1(shallow)
        enc2 = self.enc_scale2(self.down1(enc1))
        enc3 = self.enc_scale3(self.down2(enc2))

        bottleneck = self.dsm(enc3)
        bottleneck = self.bottleneck_resblock(bottleneck)

        dec2 = self.up2(bottleneck)
        if dec2.shape[-2:] != enc2.shape[-2:]:
            dec2 = torch.nn.functional.interpolate(
                dec2, size=enc2.shape[-2:], mode="bilinear", align_corners=False
            )
        dec2 = self.skip2_adjust(torch.cat((dec2, enc2), dim=1))
        dec2 = self.dec_scale2(dec2)

        dec1 = self.up1(dec2)
        if dec1.shape[-2:] != enc1.shape[-2:]:
            dec1 = torch.nn.functional.interpolate(
                dec1, size=enc1.shape[-2:], mode="bilinear", align_corners=False
            )
        dec1 = self.skip1_adjust(torch.cat((dec1, enc1), dim=1))
        dec1 = self.dec_scale1(dec1)

        residual = self.reconstruction(dec1)
        return x + residual


def build_net(**kwargs) -> FocalNet:
    """Construct a paper-oriented FocalNet model."""
    return FocalNet(**kwargs)
