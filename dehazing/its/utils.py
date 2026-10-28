"""Checkpoint, image, and metric helpers."""
from pathlib import Path
import math
import torch
from PIL import Image


def save_checkpoint(path, model, optimizer, epoch, best_psnr, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_psnr": float(best_psnr),
        "extra": extra or {},
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    return int(state.get("epoch", 0)), float(state.get("best_psnr", -math.inf))


def psnr(pred, target):
    mse = torch.mean((pred.clamp(0, 1) - target.clamp(0, 1)) ** 2).item()
    if mse == 0:
        return float("inf")
    return -10.0 * math.log10(mse)


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (tensor.detach().cpu().clamp(0, 1) * 255.0).round().byte()
    arr = arr.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)
