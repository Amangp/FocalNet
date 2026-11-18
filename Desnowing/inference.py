import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from models.focalnet import build_net


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint must be a .pth/.pt file, not a directory: {checkpoint_path}. "
            "Run repack_checkpoints.py from the FocalNet root to rebuild extracted checkpoint folders."
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint does not contain a model state_dict.")
    if checkpoint and all(k.startswith("module.") for k in checkpoint):
        checkpoint = {k[len("module."):]: v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint, strict=True)
    return model


def pad_to_multiple(image, factor=4):
    h, w = image.shape[-2:]
    pad_h, pad_w = (-h) % factor, (-w) % factor
    if pad_h or pad_w:
        mode = "reflect" if h > pad_h and w > pad_w else "replicate"
        image = F.pad(image, (0, pad_w, 0, pad_h), mode=mode)
    return image, h, w


def infer_full(model, image):
    padded, h, w = pad_to_multiple(image)
    return model(padded)[..., :h, :w]


def infer_tiled(model, image, tile_size=512, overlap=32):
    if tile_size <= 0 or tile_size % 4:
        raise ValueError("tile_size must be a positive multiple of 4")
    if overlap < 0 or overlap >= tile_size // 2 or overlap % 4:
        raise ValueError("overlap must be a non-negative multiple of 4 and < tile_size/2")
    padded, original_h, original_w = pad_to_multiple(image)
    _, _, height, width = padded.shape
    stride = tile_size - 2 * overlap
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(height - tile_size, 0))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(width - tile_size, 0))
    output = torch.zeros_like(padded)
    weight_sum = torch.zeros_like(padded[:, :1])
    for y in ys:
        for x in xs:
            patch = padded[..., y:min(y + tile_size, height), x:min(x + tile_size, width)]
            ph, pw = patch.shape[-2:]
            patch, _, _ = pad_to_multiple(patch)
            restored = model(patch)[..., :ph, :pw]
            wy = torch.ones(ph, device=image.device, dtype=image.dtype)
            wx = torch.ones(pw, device=image.device, dtype=image.dtype)
            ramp = min(overlap, ph // 2, pw // 2)
            if ramp:
                rv = torch.linspace(1.0 / (ramp + 1), 1.0, ramp, device=image.device, dtype=image.dtype)
                if y > 0: wy[:ramp] = rv
                if y + ph < height: wy[-ramp:] = rv.flip(0)
                if x > 0: wx[:ramp] = rv
                if x + pw < width: wx[-ramp:] = rv.flip(0)
            weight = (wy[:, None] * wx[None, :])[None, None]
            output[..., y:y + ph, x:x + pw] += restored * weight
            weight_sum[..., y:y + ph, x:x + pw] += weight
    return (output / weight_sum.clamp_min(1e-8))[..., :original_h, :original_w]


def process_image(model, image_path, output_path, device, tile_size, overlap):
    with Image.open(image_path) as im:
        array = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.inference_mode():
        restored = infer_tiled(model, tensor, tile_size, overlap) if tile_size else infer_full(model, tensor)
    restored = restored.clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(restored * 255).astype(np.uint8), mode="RGB").save(output_path)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="FocalNet desnowing inference")
    parser.add_argument("--input", required=True, help="Input image or folder")
    parser.add_argument("--checkpoint", required=True, help="Repacked checkpoint file, e.g. ../checkpoints/Best.pth")
    parser.add_argument("--output", default="results/inference", help="Output file or folder")
    parser.add_argument("--tile-size", type=int, default=512, help="Tile size (multiple of 4); 0 = full image")
    parser.add_argument("--overlap", type=int, default=32)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = build_net().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    src, dst = Path(args.input), Path(args.output)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    if src.is_file():
        out = dst if dst.suffix.lower() in exts else dst / f"{src.stem}_desnowed.png"
        process_image(model, src, out, device, args.tile_size, args.overlap)
    elif src.is_dir():
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(p for p in src.iterdir() if p.is_file() and p.suffix.lower() in exts)
        if not files: raise RuntimeError(f"No supported images found in {src}")
        for p in files: process_image(model, p, dst / f"{p.stem}_desnowed.png", device, args.tile_size, args.overlap)
    else:
        raise FileNotFoundError(src)

if __name__ == "__main__":
    main()
