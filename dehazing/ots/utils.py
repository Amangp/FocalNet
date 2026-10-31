"""Training and evaluation helpers for FocalNet OTS."""
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def image_metrics(pred, target):
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    psnr = 99.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)
    return {"l1": F.l1_loss(pred, target).item(), "psnr": psnr}


def dual_domain_loss(pred, target, frequency_weight=0.1):
    """Spatial L1 + weighted frequency-domain L1."""
    spatial_loss = F.l1_loss(pred, target)

    pred_fft = torch.fft.fft2(
        pred, dim=(-2, -1), norm="ortho"
    )
    target_fft = torch.fft.fft2(
        target, dim=(-2, -1), norm="ortho"
    )

    frequency_loss = torch.mean(torch.abs(pred_fft - target_fft))
    total_loss = spatial_loss + frequency_weight * frequency_loss

    return total_loss, spatial_loss.detach(), frequency_loss.detach()


def save_checkpoint(
    path, model, optimizer, epoch, best_psnr, config, scheduler=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_psnr": float(best_psnr),
        "config": config,
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path, model, optimizer=None, device="cpu", scheduler=None
):
    checkpoint = torch.load(
        path, map_location=device, weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
