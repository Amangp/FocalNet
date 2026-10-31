
"""Evaluate a trained FocalNet checkpoint and optionally save restored images."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from models.focalnet import FocalNet
from data.data_load import PairedHazeDataset
from utils import load_checkpoint
from valid import validate


def save_rgb(tensor, path):
    array = (
        tensor.detach()
        .cpu()
        .clamp(0, 1)[0]
        .permute(1, 2, 0)
        .numpy()
    )
    array = (array * 255.0).round().astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FocalNet OTS dehazing model"
    )

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hazy-dir", required=True)
    parser.add_argument("--clear-dir", default="")
    parser.add_argument("--output-dir", default="results_ots")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--num-res", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = FocalNet(
        width=args.width,
        num_res=args.num_res
    ).to(device)

    checkpoint = load_checkpoint(
        args.checkpoint,
        model,
        optimizer=None,
        device=device
    )

    model.eval()

    print(
        f"Loaded checkpoint from epoch "
        f"{checkpoint.get('epoch', '?')} on {device}"
    )

    hazy_root = Path(args.hazy_dir)
    output_root = Path(args.output_dir)

    if args.clear_dir:

        dataset = PairedHazeDataset(
            args.hazy_dir,
            args.clear_dir,
            patch_size=None,
            training=False
        )

        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=args.workers
        )

        l1, psnr = validate(model, loader, device)

        print(
            f"Evaluation: L1={l1:.6f}, "
            f"PSNR={psnr:.3f} dB"
        )

        pairs = dataset.pairs

        for hazy_path, _clear_path in pairs:

            with Image.open(hazy_path) as image:
                array = np.asarray(
                    image.convert("RGB"),
                    dtype=np.float32
                ) / 255.0

            x = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            prediction = model(x)

            save_rgb(
                prediction,
                output_root / hazy_path.name
            )

    else:

        files = sorted(
            path for path in hazy_root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {
                ".png", ".jpg", ".jpeg",
                ".bmp", ".tif", ".tiff"
            }
        )

        if not files:
            raise RuntimeError(
                f"No images found in {hazy_root}"
            )

        for path in files:

            with Image.open(path) as image:
                array = np.asarray(
                    image.convert("RGB"),
                    dtype=np.float32
                ) / 255.0

            x = (
                torch.from_numpy(array)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device)
            )

            prediction = model(x)

            save_rgb(
                prediction,
                output_root / path.name
            )

        print(
            f"Saved {len(files)} restored images to "
            f"{output_root.resolve()}"
        )


if __name__ == "__main__":
    main()
