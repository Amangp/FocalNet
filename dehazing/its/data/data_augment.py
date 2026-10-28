"""Paired geometric augmentations; same transform is applied to hazy and clear images."""
import random
import torch


def augment_pair(hazy: torch.Tensor, clear: torch.Tensor):
    if random.random() < 0.5:
        hazy, clear = hazy.flip(-1), clear.flip(-1)
    if random.random() < 0.5:
        hazy, clear = hazy.flip(-2), clear.flip(-2)
    k = random.randint(0, 3)
    if k:
        hazy, clear = torch.rot90(hazy, k, (-2, -1)), torch.rot90(clear, k, (-2, -1))
    return hazy.contiguous(), clear.contiguous()
