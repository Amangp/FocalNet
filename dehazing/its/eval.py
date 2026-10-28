"""Evaluate a trained checkpoint on paired images or restore an input folder."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype
from models import FocalNet
from data.data_load import PairedHazeDataset
from utils import load_checkpoint, save_tensor_image, psnr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hazy-dir", default=None)
    p.add_argument("--clear-dir", default=None)
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default="restored")
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--num-res", type=int, default=4)
    a = p.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FocalNet(a.width, a.num_res).to(device)
    load_checkpoint(a.checkpoint, model, device=device)
    model.eval()
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if a.hazy_dir and a.clear_dir:
        ds = PairedHazeDataset(a.hazy_dir, a.clear_dir, None, augment=False)
        loader = DataLoader(ds, batch_size=1, shuffle=False)
        scores = []
        with torch.no_grad():
            for hazy, clear, name in loader:
                pred = model(hazy.to(device)).cpu()[0]
                scores.append(psnr(pred.unsqueeze(0), clear))
                save_tensor_image(pred, out / name[0])
        print(f"Mean PSNR: {sum(scores)/len(scores):.4f} dB")
    elif a.input_dir:
        from data.data_load import EXTS
        paths = sorted(q for q in Path(a.input_dir).rglob("*") if q.is_file() and q.suffix.lower() in EXTS)
        if not paths:
            raise RuntimeError("No supported images found in --input-dir")
        with torch.no_grad():
            for path in paths:
                with Image.open(path) as im:
                    x = convert_image_dtype(pil_to_tensor(im.convert("RGB")), torch.float32).unsqueeze(0).to(device)
                y = model(x)[0]
                save_tensor_image(y, out / (path.stem + ".png"))
        print(f"Restored {len(paths)} images to {out.resolve()}")
    else:
        p.error("Provide either --hazy-dir and --clear-dir, or --input-dir")


if __name__ == "__main__":
    main()
