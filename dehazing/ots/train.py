"""Train FocalNet on the RESIDE-Outdoor dataset."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.focalnet import FocalNet
from data.data_load import PairedHazeDataset
from valid import validate
from utils import (
    dual_domain_loss,
    save_checkpoint,
    load_checkpoint,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hazy-dir", required=True)
    parser.add_argument("--clear-dir", required=True)
    parser.add_argument("--val-hazy-dir", default="")
    parser.add_argument("--val-clear-dir", default="")
    parser.add_argument("--output-dir", default="checkpoints_ots")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stop-after", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-res", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.stop_after <= args.epochs:
        parser.error("--stop-after must be between 1 and --epochs")

    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    model = FocalNet(width=args.width, num_res=args.num_res).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    start_epoch = 0
    best_psnr = float("-inf")

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume,
            model,
            optimizer,
            device,
            scheduler,
        )
        start_epoch = int(checkpoint["epoch"])
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))
        print(f"Resumed after epoch {start_epoch}")
    else:
        print("Starting new OTS training run.")

    train_ds = PairedHazeDataset(
        args.hazy_dir,
        args.clear_dir,
        patch_size=args.patch_size,
        training=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        persistent_workers=(args.workers > 0),
    )

    val_loader = None

    if args.val_hazy_dir and args.val_clear_dir:
        val_ds = PairedHazeDataset(
            args.val_hazy_dir,
            args.val_clear_dir,
            patch_size=None,
            training=False,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.workers > 0),
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Training pairs:", len(train_ds))
    print("Training batches per epoch:", len(train_loader))

    if val_loader:
        print("Validation pairs:", len(val_loader.dataset))

    print("Batch size:", args.batch_size)
    print("Workers:", args.workers)
    print("Initial learning rate:", optimizer.param_groups[0]["lr"])

    for epoch in range(start_epoch, args.stop_after):
        model.train()

        total_loss = 0.0
        total_spatial = 0.0
        total_frequency = 0.0
        seen = 0

        for hazy, clear, _names in train_loader:
            hazy = hazy.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(hazy)

            if prediction.shape != clear.shape:
                raise RuntimeError(
                    f"Prediction/target mismatch: "
                    f"{prediction.shape} vs {clear.shape}"
                )

            loss, spatial_loss, frequency_loss = dual_domain_loss(
                prediction,
                clear,
                frequency_weight=0.1,
            )

            loss.backward()
            optimizer.step()

            batch_count = hazy.size(0)
            total_loss += loss.item() * batch_count
            total_spatial += spatial_loss.item() * batch_count
            total_frequency += frequency_loss.item() * batch_count
            seen += batch_count

        scheduler.step()

        avg_loss = total_loss / max(1, seen)
        avg_spatial = total_spatial / max(1, seen)
        avg_frequency = total_frequency / max(1, seen)

        message = (
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss={avg_loss:.6f} | "
            f"Spatial L1={avg_spatial:.6f} | "
            f"Frequency L1={avg_frequency:.6f} | "
            f"LR={optimizer.param_groups[0]['lr']:.8f}"
        )

        if val_loader:
            val_l1, val_psnr = validate(model, val_loader, device)

            message += (
                f" | Val L1={val_l1:.6f}"
                f" | PSNR={val_psnr:.3f} dB"
            )

            if val_psnr > best_psnr:
                best_psnr = val_psnr

                save_checkpoint(
                    output_dir / "best_model.pkl",
                    model,
                    optimizer,
                    epoch + 1,
                    best_psnr,
                    vars(args),
                    scheduler,
                )

        save_checkpoint(
            output_dir / "last_checkpoint.pkl",
            model,
            optimizer,
            epoch + 1,
            best_psnr,
            vars(args),
            scheduler,
        )

        print(message, flush=True)

        if device.type == "cuda":
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak GPU memory: {peak:.2f} GB", flush=True)

    print("Training finished.")
    print("Best validation PSNR:", best_psnr)
    print("Checkpoints:", output_dir.resolve())
    print("Total schedule epochs:", args.epochs)
    print("Training stop epoch:", args.stop_after)


if __name__ == "__main__":
    main()
