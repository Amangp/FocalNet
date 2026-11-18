import argparse
import time
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
    verify_checkpoint,
)


def check_finite(tensor, name):
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"Non-finite values detected in {name}")


def check_optimizer(optimizer):
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value):
                check_finite(value, f"Adam state: {name}")


def make_sample_order(dataset_size, seed, epoch):
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    return torch.randperm(dataset_size, generator=generator).tolist()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hazy-dir", required=True)
    parser.add_argument("--clear-dir", required=True)
    parser.add_argument("--val-hazy-dir", required=True)
    parser.add_argument("--val-clear-dir", required=True)
    parser.add_argument("--init-weights", required=True)
    parser.add_argument("--output-dir", default="./checkpoints/ots_recovery")
    parser.add_argument("--resume", default="")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-res", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--min-lr", type=float, default=0.000001)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-limit-hours", type=float, default=0)

    args = parser.parse_args()

    if args.save_every < 1:
        parser.error("--save-every must be at least 1")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    model = FocalNet(
        width=args.width,
        num_res=args.num_res
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    start_batch = 0
    best_psnr = float("-inf")
    save_slot = 0
    running = {"loss": 0.0, "spatial": 0.0, "frequency": 0.0, "seen": 0}

    if args.resume:
        checkpoint = load_checkpoint(
            args.resume, model, optimizer, device, scheduler
        )

        state = checkpoint.get("training_state")
        if not isinstance(state, dict):
            raise RuntimeError("Resume checkpoint has no training state.")

        start_epoch = int(state["epoch_index"])
        start_batch = int(state["next_batch"])
        best_psnr = float(checkpoint["best_psnr"])
        save_slot = 1 - int(state.get("save_slot", 0))

        running = state.get("running", running)

        if state.get("torch_rng_state") is not None:
            torch.set_rng_state(state["torch_rng_state"].cpu())

        if device.type == "cuda" and state.get("cuda_rng_states") is not None:
            torch.cuda.set_rng_state_all(
                [rng_state.cpu() for rng_state in state["cuda_rng_states"]]
            )

        print(
            f"Resumed at epoch {start_epoch + 1}, "
            f"next batch {start_batch + 1}",
            flush=True
        )
    else:
        initial = verify_checkpoint(args.init_weights)
        model.load_state_dict(initial["model_state_dict"], strict=True)
        print(
            f"Initialized weights from epoch {initial['epoch']}. "
            "Using a fresh Adam optimizer.",
            flush=True
        )

    train_ds = PairedHazeDataset(
        args.hazy_dir,
        args.clear_dir,
        patch_size=args.patch_size,
        training=True,
        seed=args.seed
    )

    val_ds = PairedHazeDataset(
        args.val_hazy_dir,
        args.val_clear_dir,
        patch_size=None,
        training=False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    batches_per_epoch = (
        len(train_ds) + args.batch_size - 1
    ) // args.batch_size

    print("Training pairs:", len(train_ds), flush=True)
    print("Validation pairs:", len(val_ds), flush=True)
    print("Batches per epoch:", batches_per_epoch, flush=True)
    print("Batch size:", args.batch_size, flush=True)
    print("Learning rate:", args.lr, flush=True)
    print("Emergency save interval:", args.save_every, "batches", flush=True)

    started = time.monotonic()
    processed_batches = 0
    stop_requested = False

    def save_emergency(epoch_index, next_batch):
        nonlocal save_slot

        state = {
            "epoch_index": epoch_index,
            "next_batch": next_batch,
            "seed": args.seed,
            "save_slot": save_slot,
            "running": running.copy(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda" else None
            ),
        }

        filename = (
            "emergency_a.pkl" if save_slot == 0
            else "emergency_b.pkl"
        )

        save_checkpoint(
            output_dir / filename,
            model,
            optimizer,
            epoch_index + 1,
            best_psnr,
            vars(args),
            scheduler,
            training_state=state
        )

        print(
            f"Emergency checkpoint saved: {filename} | "
            f"epoch={epoch_index + 1} | next_batch={next_batch + 1}",
            flush=True
        )

        save_slot = 1 - save_slot

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_ds.set_epoch(epoch)

        sample_order = make_sample_order(
            len(train_ds), args.seed, epoch
        )

        batch_start = start_batch if epoch == start_epoch else 0

        if batch_start < 0 or batch_start > batches_per_epoch:
            raise RuntimeError("Invalid resume batch position.")

        first_sample = batch_start * args.batch_size
        remaining_indices = sample_order[first_sample:]

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=remaining_indices,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False
        )

        print(
            f"Starting epoch {epoch + 1}/{args.epochs} "
            f"from batch {batch_start + 1}",
            flush=True
        )

        for batch_idx, (hazy, clear, _names) in enumerate(
            train_loader, start=batch_start
        ):
            hazy = hazy.to(device, non_blocking=True)
            clear = clear.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            prediction = model(hazy)

            if prediction.shape != clear.shape:
                raise RuntimeError("Prediction/target shape mismatch.")

            loss, spatial, frequency = dual_domain_loss(
                prediction,
                clear,
                frequency_weight=0.1
            )

            check_finite(loss, "training loss")
            loss.backward()

            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    check_finite(parameter.grad, f"gradient: {name}")

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
                error_if_nonfinite=True
            )

            optimizer.step()

            for name, parameter in model.named_parameters():
                check_finite(parameter, f"parameter: {name}")

            check_optimizer(optimizer)

            count = hazy.size(0)
            running["loss"] += loss.item() * count
            running["spatial"] += spatial.item() * count
            running["frequency"] += frequency.item() * count
            running["seen"] += count

            processed_batches += 1

            next_batch = batch_idx + 1

            if next_batch % args.save_every == 0:
                save_emergency(epoch, next_batch)

            if next_batch == 1 or next_batch % 10 == 0:
                print(
                    f"Epoch {epoch + 1} | "
                    f"Batch {next_batch}/{batches_per_epoch} | "
                    f"Loss={loss.item():.6f} | "
                    f"GradNorm={grad_norm.item():.4f} | "
                    f"LR={optimizer.param_groups[0]['lr']:.8f}",
                    flush=True
                )

            if (
                args.time_limit_hours > 0
                and time.monotonic() - started
                >= args.time_limit_hours * 3600
            ):
                save_emergency(epoch, next_batch)
                print("Time limit reached. Training stopped safely.", flush=True)
                stop_requested = True
                break

        if stop_requested:
            break

        scheduler.step()

        if running["seen"] == 0:
            raise RuntimeError("No training samples processed.")

        avg_loss = running["loss"] / running["seen"]
        avg_spatial = running["spatial"] / running["seen"]
        avg_frequency = running["frequency"] / running["seen"]

        val_l1, val_psnr = validate(model, val_loader, device)

        check_finite(
            torch.tensor([val_l1, val_psnr]),
            "validation metrics"
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
                scheduler
            )

        print(
            f"Epoch {epoch + 1}/{args.epochs} COMPLETE | "
            f"Loss={avg_loss:.6f} | "
            f"Spatial={avg_spatial:.6f} | "
            f"Frequency={avg_frequency:.6f} | "
            f"Val L1={val_l1:.6f} | "
            f"PSNR={val_psnr:.3f} dB | "
            f"LR={optimizer.param_groups[0]['lr']:.8f}",
            flush=True
        )

        running = {
            "loss": 0.0,
            "spatial": 0.0,
            "frequency": 0.0,
            "seen": 0
        }

        start_batch = 0
        save_emergency(epoch + 1, 0)

    print("Training process ended.", flush=True)
    print("Best PSNR:", best_psnr, flush=True)
    print("Output:", output_dir.resolve(), flush=True)


if __name__ == "__main__":
    main()
