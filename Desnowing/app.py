from pathlib import Path

import gradio as gr
import numpy as np
import torch
from PIL import Image

from models.focalnet import build_net
from inference import load_checkpoint, infer_full, infer_tiled

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CANDIDATES = [
    CHECKPOINT_DIR / "Best.pth",
    CHECKPOINT_DIR / "model.pth",
    CHECKPOINT_DIR / "Final.pth",
]
DEFAULT_CHECKPOINT = next((p for p in DEFAULT_CANDIDATES if p.is_file()), DEFAULT_CANDIDATES[0])
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL = None
_LOADED_CHECKPOINT = None


def restore_image(image, checkpoint_path, tile_size):
    global _MODEL, _LOADED_CHECKPOINT
    if image is None:
        raise gr.Error("Upload a snowy image first.")

    checkpoint = Path(checkpoint_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = PROJECT_ROOT / checkpoint
    if not checkpoint.is_file():
        raise gr.Error(
            f"Checkpoint file not found: {checkpoint}\n"
            "If your checkpoint is still an extracted folder, run repack_checkpoints.py from the FocalNet root first."
        )

    resolved = str(checkpoint.resolve())
    if _MODEL is None or _LOADED_CHECKPOINT != resolved:
        try:
            model = build_net().to(DEVICE)
            load_checkpoint(model, resolved, DEVICE)
            model.eval()
            _MODEL = model
            _LOADED_CHECKPOINT = resolved
        except Exception as exc:
            _MODEL = None
            _LOADED_CHECKPOINT = None
            raise gr.Error(f"Could not load checkpoint into this model: {exc}")

    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    try:
        with torch.inference_mode():
            size = int(tile_size)
            restored = infer_tiled(_MODEL, tensor, tile_size=size, overlap=32) if size else infer_full(_MODEL, tensor)
            restored = restored.clamp(0, 1)[0].cpu().permute(1, 2, 0).numpy()
    except torch.cuda.OutOfMemoryError:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        raise gr.Error("GPU out of memory. Try tile size 256.")

    output = np.round(restored * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(output, mode="RGB")


with gr.Blocks(title="FocalNet Desnowing") as demo:
    gr.Markdown("# FocalNet Desnowing\nUpload a snowy image and restore it with your trained checkpoint.")
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input snowy image")
        output_image = gr.Image(type="pil", label="Desnowed result")
    checkpoint_box = gr.Textbox(
        label="Checkpoint file",
        value=str(DEFAULT_CHECKPOINT),
        placeholder=r"D:\FocalNet\checkpoints\Best.pth",
    )
    tile_size = gr.Dropdown(choices=[256, 512, 0], value=512, label="Tile size (0 = full image)")
    run_button = gr.Button("Desnow image", variant="primary")
    run_button.click(restore_image, inputs=[input_image, checkpoint_box, tile_size], outputs=output_image)
    gr.Markdown("The checkpoint is loaded on first use and reused until its path changes.")

if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    demo.launch()
