"""FocalNet image-restoration network (PyTorch)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .layers import MResBlock, ResBlockGroup, DSM
except ImportError:
    from layers import MResBlock, ResBlockGroup, DSM


class FocalNet(nn.Module):
    """Three-scale U-shaped image-restoration network.

    Input/output: RGB tensors in [0, 1], shape [N, 3, H, W].
    """
    def __init__(self, width: int = 32, num_res: int = 4):
        super().__init__()
        if width % 2:
            raise ValueError("width must be even for the multi-scale residual block")
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
    def _fit(x, reference):
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear",
                              align_corners=False)
        return x

    def forward(self, image):
        shallow = self.shallow(image)
        e1 = self.enc1(shallow)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        z = self.dec3(self.dsm(e3))

        d2 = self._fit(self.up2(z), e2)
        d2 = self.dec2(self.skip2(torch.cat((d2, e2), dim=1)))
        d1 = self._fit(self.up1(d2), e1)
        d1 = self.dec1(self.skip1(torch.cat((d1, e1), dim=1)))
        return torch.clamp(image + self.output(d1), 0.0, 1.0)


if __name__ == "__main__":
    model = FocalNet()
    x = torch.rand(1, 3, 257, 319)
    with torch.no_grad():
        y = model(x)
    print("Input:", tuple(x.shape), "Output:", tuple(y.shape))
    print("Parameters:", sum(p.numel() for p in model.parameters()))
