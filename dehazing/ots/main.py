
"""Command-line entry point for training or evaluating OTS dehazing."""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="FocalNet OTS dehazing entry point"
    )

    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument("--hazy-dir", required=True)
    parser.add_argument("--clear-dir", default="")
    parser.add_argument("--val-hazy-dir", default="")
    parser.add_argument("--val-clear-dir", default="")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--output-dir", default="checkpoints_ots")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stop-after", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-res", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", default="")

    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.mode == "train":

        if not args.clear_dir:
            parser.error("--clear-dir is required in train mode")

        if bool(args.val_hazy_dir) != bool(args.val_clear_dir):
            parser.error(
                "--val-hazy-dir and --val-clear-dir must be provided together"
            )

        if not 1 <= args.stop_after <= args.epochs:
            parser.error("--stop-after must be between 1 and --epochs")

        command = [
            sys.executable,
            str(script_dir / "train.py"),
            "--hazy-dir", args.hazy_dir,
            "--clear-dir", args.clear_dir,
            "--output-dir", args.output_dir,
            "--epochs", str(args.epochs),
            "--stop-after", str(args.stop_after),
            "--batch-size", str(args.batch_size),
            "--patch-size", str(args.patch_size),
            "--width", str(args.width),
            "--num-res", str(args.num_res),
            "--lr", str(args.lr),
            "--min-lr", str(args.min_lr),
            "--workers", str(args.workers),
        ]

        if args.val_hazy_dir:
            command += [
                "--val-hazy-dir", args.val_hazy_dir,
                "--val-clear-dir", args.val_clear_dir,
            ]

        if args.resume:
            command += ["--resume", args.resume]

    else:

        if not args.checkpoint:
            parser.error("--checkpoint is required in eval mode")

        command = [
            sys.executable,
            str(script_dir / "eval.py"),
            "--checkpoint", args.checkpoint,
            "--hazy-dir", args.hazy_dir,
            "--output-dir", args.output_dir,
            "--width", str(args.width),
            "--num-res", str(args.num_res),
            "--workers", str(args.workers),
        ]

        if args.clear_dir:
            command += ["--clear-dir", args.clear_dir]

    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
