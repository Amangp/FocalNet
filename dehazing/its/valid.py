"""Validation loop."""
import torch
from utils import psnr


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    scores = []
    for hazy, clear, _ in loader:
        hazy, clear = hazy.to(device), clear.to(device)
        pred = model(hazy)
        scores.append(psnr(pred, clear))
    model.train()
    return sum(scores) / max(1, len(scores))
