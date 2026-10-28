import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.backends import cudnn
from models.focalnet import build_net
from data.data_load import test_dataloader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "dataset"

def evaluate(model, args, device):
    from pytorch_msssim import ssim

    checkpoint = torch.load(args.test_model, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    loader = test_dataloader(args.data_dir, batch_size=1, num_workers=args.num_worker)
    if args.save_image:
        Path(args.result_dir).mkdir(parents=True, exist_ok=True)

    psnr_scores, ssim_scores = [], []
    with torch.inference_mode():
        for input_img, label_img, name in loader:
            input_img = input_img.to(device).float()
            label_img = label_img.to(device).float()
            h, w = input_img.shape[-2:]
            pad_h, pad_w = (-h) % 4, (-w) % 4
            if pad_h or pad_w:
                mode = "reflect" if h > pad_h and w > pad_w else "replicate"
                input_img = F.pad(input_img, (0, pad_w, 0, pad_h), mode=mode)

            pred = model(input_img)[..., :h, :w].clamp(0, 1)
            mse = F.mse_loss(pred, label_img).item()
            psnr_scores.append(float("inf") if mse == 0 else -10 * np.log10(mse))

            down_ratio = max(1, round(min(h, w) / 256))
            size = (max(11, h // down_ratio), max(11, w // down_ratio))
            size = (min(size[0], h), min(size[1], w))
            pred_s = F.adaptive_avg_pool2d(pred, size)
            label_s = F.adaptive_avg_pool2d(label_img, size)
            ssim_scores.append(
                ssim(pred_s, label_s, data_range=1, size_average=True).item()
            )

            if args.save_image:
                out = pred[0].cpu().permute(1, 2, 0).numpy()
                out = np.round(out * 255).astype(np.uint8)
                Image.fromarray(out).save(Path(args.result_dir) / Path(str(name[0])).name)

    if not psnr_scores:
        raise RuntimeError("No test images found.")
    print(f"Average PSNR: {np.mean(psnr_scores):.2f} dB")
    print(f"Average SSIM: {np.mean(ssim_scores):.5f}")


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate paper-oriented FocalNet desnowing model")
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", default=str(DEFAULT_DATA_DIR),
                        help=f"Dataset parent folder (default: {DEFAULT_DATA_DIR})")
    parser.add_argument("--dataset-roots", nargs="+", default=None,
                        help="Explicit training dataset roots, e.g. path/CSD path/SRRS path/Snow100K")
    parser.add_argument("--validation-roots", nargs="+", default=None,
                        help="Optional separate validation roots; each should contain val/valid/validation")
    parser.add_argument("--mode", choices=("train", "test"), default="test")
    parser.add_argument("--model-name", "--model_name", dest="model_name", default="FocalNet")
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    parser.add_argument("--learning-rate", "--learning_rate", dest="learning_rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", "--weight_decay", dest="weight_decay", type=float, default=0.0)
    parser.add_argument("--num-epoch", "--num_epoch", dest="num_epoch", type=int, default=2000)
    parser.add_argument("--print-freq", "--print_freq", dest="print_freq", type=int, default=100)
    parser.add_argument("--num-worker", "--num_worker", dest="num_worker", type=int, default=4)
    parser.add_argument("--save-freq", "--save_freq", dest="save_freq", type=int, default=50)
    parser.add_argument("--valid-freq", "--valid_freq", dest="valid_freq", type=int, default=50)
    parser.add_argument("--resume", default="")
    parser.add_argument("--test-model", "--test_model", dest="test_model", default="")
    parser.add_argument("--save-image", "--save_image", dest="save_image", action="store_true")
    parser.add_argument("--model-save-dir", default="results/FocalNet/Desnowing")
    parser.add_argument("--result-dir", default="results/FocalNet/test")
    args = parser.parse_args()

    if args.dataset_roots:
        args.dataset_roots = [str(Path(p).expanduser().resolve()) for p in args.dataset_roots]
    args.data_dir = str(Path(args.data_dir).expanduser().resolve())

    if args.mode == "train" and not args.dataset_roots and not Path(args.data_dir).is_dir():
        parser.error(f"Dataset folder not found: {args.data_dir}. Create FocalNet/dataset or pass --data-dir.")
    if args.mode == "test" and not Path(args.data_dir).is_dir():
        parser.error(f"Dataset folder not found: {args.data_dir}. Pass --data-dir to its location.")
    if args.mode == "test" and not args.test_model:
        parser.error("--test-model is required when --mode test")

    Path(args.model_save_dir).mkdir(parents=True, exist_ok=True)
    print(f"Dataset root: {args.data_dir if not args.dataset_roots else args.dataset_roots}")

    cudnn.benchmark = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_net().to(device)

    if args.mode == "train":
        from train import _train
        _train(model, args)
    else:
        evaluate(model, args, device)


if __name__ == "__main__":
    main()
