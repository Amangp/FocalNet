"""Training loop for the paper-oriented FocalNet desnowing model."""

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data.data_load import train_dataloader, valid_dataloader


def _load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location=device)


def _pad_to_multiple(image, factor=4):
    h, w = image.shape[-2:]
    pad_h, pad_w = (-h) % factor, (-w) % factor
    if pad_h or pad_w:
        mode = "reflect" if h > pad_h and w > pad_w else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    return image, h, w


def _validate(model, args, device):
    validation_source = getattr(args, "validation_roots", None) or getattr(args, "dataset_roots", None) or args.data_dir
    loader = valid_dataloader(validation_source, batch_size=1, num_workers=0)
    model.eval()
    psnr_values = []

    with torch.inference_mode():
        for input_img, label_img in loader:
            input_img = input_img.to(device, non_blocking=True).float()
            label_img = label_img.to(device, non_blocking=True).float()
            padded, h, w = _pad_to_multiple(input_img)
            pred = model(padded)[..., :h, :w].clamp(0.0, 1.0)
            mse = F.mse_loss(pred, label_img).item()
            psnr_values.append(float("inf") if mse == 0 else -10.0 * math.log10(mse))

    if not psnr_values:
        raise RuntimeError("Validation set is empty; check data_dir/test2000/Snow.")
    return sum(psnr_values) / len(psnr_values)


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


def _train(model, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    training_source = getattr(args, "dataset_roots", None) or args.data_dir
    loader = train_dataloader(training_source, args.batch_size, args.num_worker, balanced=True)
    if len(loader) == 0:
        raise RuntimeError("Training set is empty; check each dataset root has a training split with paired input and ground-truth images.")

    start_epoch = 1
    best_psnr = float("-inf")
    if args.resume:
        checkpoint = _load_checkpoint(args.resume, device)
        weights = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(weights, strict=True)
        if isinstance(checkpoint, dict) and "optimizer" in checkpoint and "epoch" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_psnr = float(checkpoint.get("best_psnr", best_psnr))
            print(f"Resuming from epoch {start_epoch}; latest completed epoch was {start_epoch - 1}.")
        else:
            print("Loaded model weights only; optimizer and epoch state are missing, so training starts at epoch 1.")

    l1_loss = torch.nn.L1Loss()
    save_dir = Path(args.model_save_dir)
    model.train()

    for epoch in range(start_epoch, args.num_epoch + 1):
        lr = _set_learning_rate(optimizer, epoch, args)
        total_loss_sum = 0.0
        content_loss_sum = 0.0
        fft_loss_sum = 0.0

        for step, (input_img, label_img) in enumerate(loader, start=1):
            input_img = input_img.to(device, non_blocking=True).float()
            label_img = label_img.to(device, non_blocking=True).float()

            optimizer.zero_grad(set_to_none=True)
            pred = model(input_img)

            content_loss = l1_loss(pred, label_img)
            pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
            label_fft = torch.fft.fft2(label_img, dim=(-2, -1))
            pred_fft_pair = torch.stack((pred_fft.real, pred_fft.imag), dim=-1)
            label_fft_pair = torch.stack((label_fft.real, label_fft.imag), dim=-1)
            fft_loss = l1_loss(pred_fft_pair, label_fft_pair)
            loss = content_loss + 0.1 * fft_loss

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, step {step}.")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.001)
            optimizer.step()

            total_loss_sum += loss.item()
            content_loss_sum += content_loss.item()
            fft_loss_sum += fft_loss.item()

            if step % max(1, args.print_freq) == 0 or step == len(loader):
                print(
                    f"Epoch {epoch}/{args.num_epoch} | Step {step}/{len(loader)} "
                    f"| Loss {loss.item():.4f} | LR {lr:.2e}"
                )

        steps = len(loader)
        print(
            f"Epoch {epoch} complete | Loss {total_loss_sum / steps:.4f} "
            f"| L1 {content_loss_sum / steps:.4f} | FFT {fft_loss_sum / steps:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
        }
        torch.save(checkpoint, save_dir / "model.pkl")

        if epoch % max(1, args.save_freq) == 0:
            torch.save({"model": model.state_dict()}, save_dir / f"model_{epoch}.pkl")

        if epoch % max(1, args.valid_freq) == 0 or epoch == args.num_epoch:
            val_psnr = _validate(model, args, device)
            print(f"Validation PSNR: {val_psnr:.2f} dB")
            if val_psnr >= best_psnr:
                best_psnr = val_psnr
                torch.save({"model": model.state_dict()}, save_dir / "Best.pkl")
            model.train()

        # Update the resumable checkpoint with the latest best PSNR.
        checkpoint["best_psnr"] = best_psnr
        torch.save(checkpoint, save_dir / "model.pkl")

    torch.save({"model": model.state_dict()}, save_dir / "Final.pkl")
