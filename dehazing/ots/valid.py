"""Validation loop for FocalNet dehazing."""
import torch
import torch.nn.functional as F
from utils import image_metrics


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_l1, total_psnr, count = 0.0, 0.0, 0
    for hazy, clear, _names in loader:
        hazy = hazy.to(device, non_blocking=True)
        clear = clear.to(device, non_blocking=True)
        pred = model(hazy)
        if pred.shape != clear.shape:
            raise RuntimeError(f"Prediction/target mismatch: {pred.shape} vs {clear.shape}")
        metrics = image_metrics(pred, clear)
        n = hazy.size(0)
        total_l1 += metrics["l1"] * n
        total_psnr += metrics["psnr"] * n
        count += n
    if count == 0:
        raise RuntimeError("Validation dataset is empty.")
    return total_l1 / count, total_psnr / count
