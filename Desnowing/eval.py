import argparse
import time
from pathlib import Path

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
    model = build_net().to(device)
    checkpoint = torch.load(args.test_model, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    loader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    psnr_scores, ssim_scores = [], []

    if args.save_images:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for input_img, label_img, name in loader:
            input_img = input_img.to(device).float()
            label_img = label_img.to(device).float()
            h, w = input_img.shape[-2:]

            pad_h, pad_w = (-h) % 4, (-w) % 4
            if pad_h or pad_w:
                mode = "reflect" if h > pad_h and w > pad_w else "replicate"
                input_img = F.pad(input_img, (0, pad_w, 0, pad_h), mode=mode)

            start = time.perf_counter()
            pred = model(input_img)[..., :h, :w].clamp(0, 1)
            if device.type == "cuda":
                torch.cuda.synchronize()
            _ = time.perf_counter() - start

            mse = F.mse_loss(pred, label_img).item()
            psnr_scores.append(float("inf") if mse == 0 else -10 * np.log10(mse))

            # Match the adaptive-pooling SSIM approach in the supplied sample.
            down_ratio = max(1, round(min(h, w) / 256))
            size = (max(11, h // down_ratio), max(11, w // down_ratio))
            size = (min(size[0], h), min(size[1], w))
            pred_s = F.adaptive_avg_pool2d(pred, size)
            label_s = F.adaptive_avg_pool2d(label_img, size)
            ssim_scores.append(ssim(pred_s, label_s, data_range=1, size_average=True).item())

            if args.save_images:
                out = pred[0].cpu().permute(1, 2, 0).numpy()
                out = np.round(out * 255).astype(np.uint8)
                Image.fromarray(out).save(Path(args.result_dir) / Path(str(name[0])).name)

    if not psnr_scores:
        raise RuntimeError("No test images were loaded.")

    print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_scores):.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                        help=f"Dataset root (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--test-model", required=True)
    parser.add_argument("--save-images", action="store_true")
    parser.add_argument("--result-dir", default="results/desnowing")
    main(parser.parse_args())
