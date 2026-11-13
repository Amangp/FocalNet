"""Training loop for the paper-oriented FocalNet desnowing model.

Updates:
- Validation reports both PSNR and SSIM.
- Appends per-epoch train/validation metrics to training_metrics.csv.
- Uses CUDA automatic mixed precision (AMP) for the model forward pass.
- Saves/restores AMP GradScaler state when available, alongside the existing
  model, optimizer, epoch, and best-PSNR checkpoint state.

This file preserves the existing model, L1 + 0.1 * FFT loss, gradient clipping,
and best-checkpoint selection by PSNR. Validation runs every 10 epochs.
"""

import csv
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from data.data_load import train_dataloader, valid_dataloader


# Run validation and print PSNR + SSIM every 10 completed epochs.
VALIDATION_FREQUENCY = 10


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def _atomic_save(payload, path: Path):
    """Safely overwrite a checkpoint without leaving a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _make_grad_scaler(enabled: bool):
    """Create a GradScaler compatible with both newer and older PyTorch APIs."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _autocast_context(device):
    """Use FP16 autocast on CUDA; leave CPU execution in full precision."""
    if device.type == "cuda":
        try:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        except AttributeError:  # Compatibility with older PyTorch versions.
            return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


def _pad_to_multiple(image, factor=4):
    h, w = image.shape[-2:]
    pad_h, pad_w = (-h) % factor, (-w) % factor
    if pad_h or pad_w:
        mode = "reflect" if h > pad_h and w > pad_w else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    return image, h, w


def _ssim(pred, target, window_size=11, sigma=1.5):
    """Return mean SSIM for NCHW images in the [0, 1] range.

    Uses a Gaussian-weighted local window and reflect padding (replicate
    padding for very small images). Inputs are evaluated in float32.
    """
    pred = pred.float()
    target = target.float()

    if pred.shape != target.shape:
        raise ValueError(
            f"SSIM inputs must have matching shapes; got {pred.shape} and {target.shape}."
        )

    channels = pred.shape[1]
    height, width = pred.shape[-2:]
    # Keep an odd window that fits even if a small image is supplied.
    max_window = min(window_size, height if height % 2 else height - 1,
                     width if width % 2 else width - 1)
    max_window = max(1, max_window)
    if max_window % 2 == 0:
        max_window -= 1
    radius = max_window // 2

    coords = torch.arange(max_window, device=pred.device, dtype=pred.dtype) - radius
    gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gaussian = gaussian / gaussian.sum()
    window_2d = gaussian[:, None] * gaussian[None, :]
    window = window_2d.expand(channels, 1, max_window, max_window).contiguous()

    if radius:
        pad_mode = "reflect" if height > radius and width > radius else "replicate"
        pred_pad = F.pad(pred, (radius, radius, radius, radius), mode=pad_mode)
        target_pad = F.pad(target, (radius, radius, radius, radius), mode=pad_mode)
    else:
        pred_pad = pred
        target_pad = target

    mu_x = F.conv2d(pred_pad, window, groups=channels)
    mu_y = F.conv2d(target_pad, window, groups=channels)
    mu_x_sq = mu_x.square()
    mu_y_sq = mu_y.square()
    mu_xy = mu_x * mu_y

    sigma_x = F.conv2d(pred_pad * pred_pad, window, groups=channels) - mu_x_sq
    sigma_y = F.conv2d(target_pad * target_pad, window, groups=channels) - mu_y_sq
    sigma_xy = F.conv2d(pred_pad * target_pad, window, groups=channels) - mu_xy

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x + sigma_y + c2)
    score = numerator / denominator.clamp_min(1e-12)
    return score.mean()


def _validate(model, args, device):
    validation_source = (
        getattr(args, "validation_roots", None)
        or getattr(args, "dataset_roots", None)
        or args.data_dir
    )
    loader = valid_dataloader(validation_source, batch_size=1, num_workers=0)
    model.eval()
    psnr_values = []
    ssim_values = []

    with torch.inference_mode():
        for input_img, label_img in loader:
            input_img = input_img.to(device, non_blocking=True).float()
            label_img = label_img.to(device, non_blocking=True).float()
            padded, h, w = _pad_to_multiple(input_img)
            pred = model(padded)[..., :h, :w].float().clamp(0.0, 1.0)

            mse = F.mse_loss(pred, label_img).item()
            psnr_values.append(float("inf") if mse == 0 else -10.0 * math.log10(mse))
            ssim_values.append(_ssim(pred, label_img).item())

    if not psnr_values:
        raise RuntimeError("Validation set is empty; check data_dir/test2000/Snow.")

    return sum(psnr_values) / len(psnr_values), sum(ssim_values) / len(ssim_values)


def _set_learning_rate(optimizer, epoch, args):
    """Three-epoch linear warm-up followed by cosine decay."""
    warmup_epochs = min(3, max(1, args.num_epoch - 1))
    min_lr = 1e-6
    base_lr = args.learning_rate

    if epoch <= warmup_epochs:
        lr = base_lr * epoch / warmup_epochs
    else:
        duration = max(1, args.num_epoch - warmup_epochs)
        progress = min(1.0, (epoch - warmup_epochs) / duration)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def _append_metrics(metrics_path, row):
    """Append a row to the per-epoch CSV log, creating the header if needed."""
    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "epoch_seconds", "learning_rate", "train_loss", "train_l1",
        "train_fft", "validation_psnr_db", "validation_ssim",
    ]
    write_header = not metrics_path.exists() or metrics_path.stat().st_size == 0
    with metrics_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _train(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    save_dir = Path(args.model_save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_dir / "training_metrics.csv"
    amp_enabled = device.type == "cuda"

    # cuDNN autotuning can help when the training crops have fixed dimensions.
    # Set this to False if your batches frequently have varying image sizes.
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    scaler = _make_grad_scaler(enabled=amp_enabled)

    training_source = getattr(args, "dataset_roots", None) or args.data_dir
    loader = train_dataloader(
        training_source, args.batch_size, args.num_worker, balanced=True
    )
    if len(loader) == 0:
        raise RuntimeError(
            "Training set is empty; check each dataset root has a training split "
            "with paired input and ground-truth images."
        )

    start_epoch = 1
    best_psnr = float("-inf")
    if args.resume:
        checkpoint = _load_checkpoint(args.resume, device)
        weights = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(weights, strict=True)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_psnr = float(checkpoint.get("best_psnr", best_psnr))
            print(
                f"Resuming from epoch {start_epoch}; latest completed epoch was "
                f"{start_epoch - 1}."
            )
        else:
            print(
                "Loaded model weights only; optimizer and epoch state are missing, "
                "so training starts at epoch 1."
            )

    l1_loss = torch.nn.L1Loss()
    model.train()

    for epoch in range(start_epoch, args.num_epoch + 1):
        epoch_start = time.perf_counter()
        lr = _set_learning_rate(optimizer, epoch, args)
        total_loss_sum = 0.0
        content_loss_sum = 0.0
        fft_loss_sum = 0.0

        for step, (input_img, label_img) in enumerate(loader, start=1):
            input_img = input_img.to(device, non_blocking=True).float()
            label_img = label_img.to(device, non_blocking=True).float()
            optimizer.zero_grad(set_to_none=True)

            # Run the model forward pass under AMP on CUDA. Keep FFT and losses
            # in FP32 for greater numerical stability.
            with _autocast_context(device):
                pred = model(input_img)

            pred_fp32 = pred.float()
            label_fp32 = label_img.float()
            content_loss = l1_loss(pred_fp32, label_fp32)

            pred_fft = torch.fft.fft2(pred_fp32, dim=(-2, -1))
            label_fft = torch.fft.fft2(label_fp32, dim=(-2, -1))
            pred_fft_pair = torch.stack((pred_fft.real, pred_fft.imag), dim=-1)
            label_fft_pair = torch.stack((label_fft.real, label_fft.imag), dim=-1)
            fft_loss = l1_loss(pred_fft_pair, label_fft_pair)
            loss = content_loss + 0.1 * fft_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}.")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += loss.detach().item()
            content_loss_sum += content_loss.detach().item()
            fft_loss_sum += fft_loss.detach().item()

            if step % max(1, args.print_freq) == 0 or step == len(loader):
                print(
                    f"Epoch {epoch}/{args.num_epoch} | Step {step}/{len(loader)} "
                    f"| Loss {loss.detach().item():.4f} | LR {lr:.2e}"
                )

        steps = len(loader)
        train_loss = total_loss_sum / steps
        train_l1 = content_loss_sum / steps
        train_fft = fft_loss_sum / steps
        print(
            f"Epoch {epoch} complete | Loss {train_loss:.4f} "
            f"| L1 {train_l1:.4f} | FFT {train_fft:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
        }
        _atomic_save(checkpoint, save_dir / "model.pkl")
        print(f"Updated checkpoint: {save_dir / 'model.pkl'}")

        val_psnr = ""
        val_ssim = ""
        if epoch % VALIDATION_FREQUENCY == 0 or epoch == args.num_epoch:
            val_psnr_value, val_ssim_value = _validate(model, args, device)
            val_psnr = f"{val_psnr_value:.6f}"
            val_ssim = f"{val_ssim_value:.6f}"
            print(
                f"Validation | PSNR: {val_psnr_value:.2f} dB "
                f"| SSIM: {val_ssim_value:.4f}"
            )

            if val_psnr_value >= best_psnr:
                best_psnr = val_psnr_value
                _atomic_save({"model": model.state_dict()}, save_dir / "Best.pkl")
                print(f"Updated best checkpoint: {save_dir / 'Best.pkl'}")
            model.train()

        # Refresh latest checkpoint with the best score after validation.
        checkpoint["best_psnr"] = best_psnr
        _atomic_save(checkpoint, save_dir / "model.pkl")

        epoch_seconds = time.perf_counter() - epoch_start
        _append_metrics(
            metrics_path,
            {
                "epoch": epoch,
                "epoch_seconds": f"{epoch_seconds:.3f}",
                "learning_rate": f"{lr:.10g}",
                "train_loss": f"{train_loss:.6f}",
                "train_l1": f"{train_l1:.6f}",
                "train_fft": f"{train_fft:.6f}",
                "validation_psnr_db": val_psnr,
                "validation_ssim": val_ssim,
            },
        )

    _atomic_save({"model": model.state_dict()}, save_dir / "Final.pkl")
