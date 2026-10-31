"""Three-scale FocalNet-style dehazing network draft."""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .layers import MResBlock, ResBlockGroup, DSM
except ImportError:
    from layers import MResBlock, ResBlockGroup, DSM


class FocalNet(nn.Module):
    """Three-scale encoder-decoder for single-image dehazing.

    This is an independent implementation draft. Verify every block and connection
    against the paper before claiming a strict reproduction.
    """
    def __init__(self, width=32, num_res=4):
        super().__init__()
        if width % 2:
            raise ValueError("width must be divisible by 2.")

        self.shallow = nn.Conv2d(3, width, 3, padding=1)

        self.enc1 = MResBlock(width)
        self.down1 = nn.Conv2d(width, width * 2, 3, stride=2, padding=1)
        self.enc2 = ResBlockGroup(width * 2, num_res)
        self.down2 = nn.Conv2d(width * 2, width * 4, 3, stride=2, padding=1)
        self.enc3 = ResBlockGroup(width * 4, num_res)

        self.dsm = DSM(width * 4)

        self.dec3 = ResBlockGroup(width * 4, num_res)
        self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
        self.skip2 = nn.Conv2d(width * 4, width * 2, 1)
        self.dec2 = ResBlockGroup(width * 2, num_res)
        self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
        self.skip1 = nn.Conv2d(width * 2, width, 1)
        self.dec1 = MResBlock(width)

        self.output = nn.Conv2d(width, 3, 3, padding=1)

    @staticmethod
    def _match_size(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:],
                              mode="bilinear", align_corners=False)
        return x

    def forward(self, image):
        shallow = self.shallow(image)
        e1 = self.enc1(shallow)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        z = self.dsm(e3)
        d3 = self.dec3(z)

        d2 = self._match_size(self.up2(d3), e2)
        d2 = self.dec2(self.skip2(torch.cat([d2, e2], dim=1)))

        d1 = self._match_size(self.up1(d2), e1)
        d1 = self.dec1(self.skip1(torch.cat([d1, e1], dim=1)))

        return self.output(d1) + image


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FocalNet().to(device).eval()
    image = torch.rand(1, 3, 257, 319, device=device)
    with torch.no_grad():
        output = model(image)
    print("Input:", tuple(image.shape))
    print("Output:", tuple(output.shape))
    print("Shape test:", "PASSED" if output.shape == image.shape else "FAILED")
    print("Parameters:", sum(p.numel() for p in model.parameters()))
