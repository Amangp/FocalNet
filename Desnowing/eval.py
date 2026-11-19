
import argparse
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ssim

from data.data_load import test_dataloader
from models.focalnet import build_net


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluation device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = build_net().to(device)

    checkpoint = torch.load(
        args.test_model,
        map_location=device,
        weights_only=False
    )

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    loader = test_dataloader(
        args.data_dir,
        batch_size=1,
        num_workers=0
    )

    results = defaultdict(lambda: {"psnr": [], "ssim": []})

    if args.save_images:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for index, (input_img, label_img, name) in enumerate(loader, 1):
            input_img = input_img.to(device).float()
            label_img = label_img.to(device).float()

            # Dataset name is prefixed by the data loader.
            image_name = str(name[0])
            dataset_name = image_name.split("__", 1)[0]

            h, w = input_img.shape[-2:]

            pad_h, pad_w = (-h) % 4, (-w) % 4

            if pad_h or pad_w:
                mode = "reflect" if h > pad_h and w > pad_w else "replicate"
                input_img = F.pad(
                    input_img,
                    (0, pad_w, 0, pad_h),
                    mode=mode
                )

            pred = model(input_img)[..., :h, :w].clamp(0, 1)

            mse = F.mse_loss(pred, label_img).item()
            psnr_value = float("inf") if mse == 0 else -10 * np.log10(mse)

            down_ratio = max(1, round(min(h, w) / 256))
            size = (
                min(max(11, h // down_ratio), h),
                min(max(11, w // down_ratio), w)
            )

            pred_s = F.adaptive_avg_pool2d(pred, size)
            label_s = F.adaptive_avg_pool2d(label_img, size)

            ssim_value = ssim(
                pred_s,
                label_s,
                data_range=1,
                size_average=True
            ).item()

            results[dataset_name]["psnr"].append(psnr_value)
            results[dataset_name]["ssim"].append(ssim_value)

            if args.save_images:
                output = pred[0].cpu().permute(1, 2, 0).numpy()
                output = np.round(output * 255).astype(np.uint8)

                filename = image_name.split("__", 1)[-1]
                Image.fromarray(output).save(
                    Path(args.result_dir) / dataset_name / filename
                )

            if index % 100 == 0:
                print(f"Processed {index}/{len(loader)} images")

    print("\n" + "=" * 65)
    print("FOCALNET DESNOWING - TEST RESULTS")
    print("=" * 65)

    print(f"{'Dataset':<18}{'Images':>10}{'PSNR (dB)':>17}{'SSIM':>14}")
    print("-" * 65)

    all_psnr = []
    all_ssim = []

    for dataset_name in sorted(results):
        psnr_values = results[dataset_name]["psnr"]
        ssim_values = results[dataset_name]["ssim"]

        avg_psnr = np.mean(psnr_values)
        avg_ssim = np.mean(ssim_values)

        all_psnr.extend(psnr_values)
        all_ssim.extend(ssim_values)

        print(
            f"{dataset_name:<18}"
            f"{len(psnr_values):>10}"
            f"{avg_psnr:>17.2f}"
            f"{avg_ssim:>14.5f}"
        )

    print("-" * 65)
    print(
        f"{'Overall':<18}"
        f"{len(all_psnr):>10}"
        f"{np.mean(all_psnr):>17.2f}"
        f"{np.mean(all_ssim):>14.5f}"
    )
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FocalNet desnowing evaluation"
    )

    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_DATA_DIR)
    )

    parser.add_argument(
        "--test-model",
        required=True
    )

    parser.add_argument(
        "--save-images",
        action="store_true"
    )

    parser.add_argument(
        "--result-dir",
        default="results/desnowing"
    )

    main(parser.parse_args())
