
"""Basic shape and gradient tests for FocalNet."""

import torch

from models.focalnet import FocalNet


def test_output_shape():
    model = FocalNet(
        in_channels=3,
        out_channels=3,
        channels=(32, 64, 128),
        num_blocks=1,
    )

    x = torch.randn(1, 3, 64, 64)
    y = model(x)

    assert y.shape == x.shape


def test_odd_image_shape():
    model = FocalNet(
        channels=(32, 64, 128),
        num_blocks=1,
    )

    x = torch.randn(1, 3, 65, 67)
    y = model(x)

    assert y.shape == x.shape


def test_gradient_flow():
    model = FocalNet(
        channels=(32, 64, 128),
        num_blocks=1,
    )

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    loss = y.mean()
    loss.backward()

    assert model.shallow.weight.grad is not None
    assert model.reconstruction.weight.grad is not None
