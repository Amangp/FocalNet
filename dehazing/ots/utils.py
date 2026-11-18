"""Training and evaluation helpers for FocalNet OTS."""
import hashlib
import math
import os
import tempfile
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
    spatial_loss = F.l1_loss(pred, target)
    pred_fft = torch.fft.fft2(pred, dim=(-2, -1), norm="ortho")
    target_fft = torch.fft.fft2(target, dim=(-2, -1), norm="ortho")
    frequency_loss = torch.mean(torch.abs(pred_fft - target_fft))
    total_loss = spatial_loss + frequency_weight * frequency_loss
    return total_loss, spatial_loss.detach(), frequency_loss.detach()


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a dictionary.")

    required = {
        "epoch", "model_state_dict", "optimizer_state_dict",
        "scheduler_state_dict", "best_psnr", "config"
    }

    missing = required - checkpoint.keys()
    if missing:
        raise RuntimeError(f"Missing checkpoint keys: {sorted(missing)}")

    if not isinstance(checkpoint["epoch"], int) or checkpoint["epoch"] < 0:
        raise RuntimeError("Invalid checkpoint epoch.")

    if not isinstance(checkpoint["model_state_dict"], dict):
        raise RuntimeError("Invalid model state dictionary.")

    if not isinstance(checkpoint["config"], dict):
        raise RuntimeError("Invalid checkpoint configuration.")

    best_psnr = float(checkpoint["best_psnr"])
    if not math.isfinite(best_psnr) and best_psnr != float("-inf"):
        raise RuntimeError("Invalid best PSNR.")

    for name, value in checkpoint["model_state_dict"].items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite model tensor: {name}")

    optimizer_state = checkpoint["optimizer_state_dict"]
    if optimizer_state is not None:
        for state in optimizer_state.get("state", {}).values():
            for name, value in state.items():
                if torch.is_tensor(value) and not torch.isfinite(value).all():
                    raise RuntimeError(f"Non-finite optimizer state: {name}")

    return True


def verify_checkpoint(path):
    path = Path(path)
    checksum_path = Path(str(path) + ".sha256")

    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if not checksum_path.is_file():
        raise RuntimeError(f"Checkpoint checksum is missing: {checksum_path}")

    expected = checksum_path.read_text(encoding="utf-8").strip()
    actual = _sha256(path)

    if len(expected) != 64 or actual != expected:
        raise RuntimeError(f"Checkpoint checksum mismatch: {path}")

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    _validate_checkpoint(checkpoint)
    return checkpoint


def save_checkpoint(
    path, model, optimizer, epoch, best_psnr, config,
    scheduler=None, training_state=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "best_psnr": float(best_psnr),
        "config": dict(config),
        "training_state": training_state,
    }

    _validate_checkpoint(checkpoint)

    temp_path = None
    temp_checksum_path = None
    checksum_path = Path(str(path) + ".sha256")

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent,
            prefix=path.name + ".", suffix=".tmp", delete=False
        ) as temp_file:
            temp_path = Path(temp_file.name)
            torch.save(checkpoint, temp_file)
            temp_file.flush()
            os.fsync(temp_file.fileno())

        test_checkpoint = torch.load(
            temp_path, map_location="cpu", weights_only=False
        )
        _validate_checkpoint(test_checkpoint)
        checksum = _sha256(temp_path)

        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=checksum_path.name + ".", suffix=".tmp", delete=False
        ) as checksum_file:
            temp_checksum_path = Path(checksum_file.name)
            checksum_file.write(checksum + "\n")
            checksum_file.flush()
            os.fsync(checksum_file.fileno())

        os.replace(temp_path, path)
        temp_path = None
        os.replace(temp_checksum_path, checksum_path)
        temp_checksum_path = None

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        if temp_checksum_path is not None:
            temp_checksum_path.unlink(missing_ok=True)


def load_checkpoint(
    path, model, optimizer=None, device="cpu", scheduler=None
):
    checkpoint = verify_checkpoint(path)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if optimizer is not None and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
