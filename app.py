"""Unified FocalNet inference app for desnowing and dehazing.

Expected project layout:
    FocalNet/
      app.py
      checkpoints/
        Best.pth              # Desnowing checkpoint (or model.pth / Final.pth)
        its_model.pkl         # ITS dehazing checkpoint
        ots_model.pkl         # OTS dehazing checkpoint
      Desnowing/
      Dehazing/its/
      Dehazing/ots/

Run from the project root with: python app.py
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any
import uuid

import gradio as gr
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Project paths and model imports
# -----------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results" / "unified_inference"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Import the three independent architectures as packages. Using package imports
# keeps each model connected to its own `layers.py` implementation.
import sys
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from Desnowing.models.focalnet import build_net as build_desnowing_net
    from Dehazing.its.models.FocalNet import FocalNet as ITSDehazingNet
    from Dehazing.ots.models.focalnet import FocalNet as OTSDehazingNet
except Exception as exc:
    raise ImportError(
        "Could not import the FocalNet architectures. Make sure app.py is in "
        "the repository root and the Desnowing/ and Dehazing/ folders are present."
    ) from exc


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DESNOW_CHECKPOINT_CANDIDATES = (
    CHECKPOINT_DIR / "Best.pth",
    CHECKPOINT_DIR / "model.pth",
    CHECKPOINT_DIR / "Final.pth",
)
DEFAULT_DESNOW_CHECKPOINT = next(
    (path for path in DESNOW_CHECKPOINT_CANDIDATES if path.is_file()),
    DESNOW_CHECKPOINT_CANDIDATES[0],
)
DEFAULT_ITS_CHECKPOINT = CHECKPOINT_DIR / "its_model.pkl"
DEFAULT_OTS_CHECKPOINT = CHECKPOINT_DIR / "ots_model.pkl"

# Only the most recently selected network is retained in memory. This avoids
# holding three large networks on the GPU at the same time.
_LOADED_MODEL: torch.nn.Module | None = None
_LOADED_MODEL_KEY: tuple[str, str, str, int, int] | None = None


# -----------------------------------------------------------------------------
# Checkpoint loading
# -----------------------------------------------------------------------------

def _torch_load(path: Path) -> Any:
    """Load a PyTorch checkpoint on CPU before moving the model to the device."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # Compatibility with older PyTorch versions.
        return torch.load(path, map_location="cpu")


def _extract_state_dict(checkpoint: Any) -> tuple[dict[str, torch.Tensor], dict]:
    """Accept the checkpoint formats used by the three project implementations."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a PyTorch state dictionary or checkpoint dictionary.")

    metadata = checkpoint
    state = None
    for key in ("model", "model_state_dict", "state_dict", "net", "weights"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, dict):
            state = candidate
            break

    # A raw state_dict has tensor values and no wrapper key.
    if state is None and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state = checkpoint

    if not isinstance(state, dict) or not state:
        available = ", ".join(map(str, checkpoint.keys()))
        raise ValueError(
            "Could not find model weights in checkpoint. Expected one of "
            "'model', 'model_state_dict', 'state_dict', or a raw state_dict. "
            f"Found keys: {available or '(none)'}"
        )

    # Handle DataParallel checkpoints.
    if all(str(key).startswith("module.") for key in state):
        state = {str(key)[7:]: value for key, value in state.items()}

    return state, metadata


def _model_config(metadata: dict) -> tuple[int, int]:
    """Read width/num_res from training metadata, falling back to project defaults."""
    config = metadata.get("config", {})
    if not isinstance(config, dict):
        config = {}
    extra = metadata.get("extra", {})
    if not isinstance(extra, dict):
        extra = {}

    width = config.get("width", extra.get("width", 32))
    num_res = config.get("num_res", extra.get("num_res", 4))
    try:
        width, num_res = int(width), int(num_res)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid width/num_res values in checkpoint metadata.") from exc
    return width, num_res


def _load_network(task: str, variant: str, checkpoint_path: str) -> torch.nn.Module:
    global _LOADED_MODEL, _LOADED_MODEL_KEY

    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = ROOT_DIR / path
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Check that the file is inside the root-level checkpoints folder and "
            "that its name matches the path shown in the app."
        )

    stat = path.stat()
    cache_key = (task, variant if task == "Dehazing" else "", str(path), stat.st_mtime_ns, stat.st_size)
    if _LOADED_MODEL is not None and _LOADED_MODEL_KEY == cache_key:
        return _LOADED_MODEL

    checkpoint = _torch_load(path)
    state_dict, metadata = _extract_state_dict(checkpoint)

    if task == "Desnowing":
        model = build_desnowing_net()
    elif task == "Dehazing" and variant == "ITS (Indoor)":
        width, num_res = _model_config(metadata)
        model = ITSDehazingNet(width=width, num_res=num_res)
    elif task == "Dehazing" and variant == "OTS (Outdoor)":
        width, num_res = _model_config(metadata)
        model = OTSDehazingNet(width=width, num_res=num_res)
    else:
        raise ValueError(f"Unsupported task/model selection: {task} / {variant}")

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"The checkpoint '{path.name}' does not match the selected {task} model"
            + (f" ({variant})" if task == "Dehazing" else "")
            + ". Check that the correct trained checkpoint is selected.\n"
            f"Details: {exc}"
        ) from exc

    # Release the previous network before moving the new one onto the GPU.
    # This keeps peak GPU memory lower when switching between the three models.
    if _LOADED_MODEL is not None:
        _LOADED_MODEL.cpu()
        del _LOADED_MODEL
        _LOADED_MODEL = None
        _LOADED_MODEL_KEY = None
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    model.to(DEVICE)
    model.eval()

    _LOADED_MODEL = model
    _LOADED_MODEL_KEY = cache_key
    return model


# -----------------------------------------------------------------------------
# Image inference
# -----------------------------------------------------------------------------

def _pad_to_multiple(tensor: torch.Tensor, factor: int = 4):
    height, width = tensor.shape[-2:]
    pad_h = (-height) % factor
    pad_w = (-width) % factor
    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return tensor, height, width


def _infer_full(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    padded, height, width = _pad_to_multiple(image)
    restored = model(padded)
    return restored[..., :height, :width]


def _infer_tiled(
    model: torch.nn.Module,
    image: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 32,
) -> torch.Tensor:
    if tile_size <= 0 or tile_size % 4:
        raise ValueError("Tile size must be 0 (full image) or a positive multiple of 4.")
    if overlap < 0 or overlap >= tile_size // 2 or overlap % 4:
        raise ValueError("Overlap must be a non-negative multiple of 4 and less than half the tile size.")

    padded, original_h, original_w = _pad_to_multiple(image)
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
            patch_h, patch_w = patch.shape[-2:]
            patch, _, _ = _pad_to_multiple(patch)
            restored = model(patch)[..., :patch_h, :patch_w]

            weight_y = torch.ones(patch_h, device=image.device, dtype=image.dtype)
            weight_x = torch.ones(patch_w, device=image.device, dtype=image.dtype)
            ramp = min(overlap, patch_h // 2, patch_w // 2)
            if ramp:
                ramp_values = torch.linspace(
                    1.0 / (ramp + 1), 1.0, ramp,
                    device=image.device, dtype=image.dtype,
                )
                if y > 0:
                    weight_y[:ramp] = ramp_values
                if y + patch_h < height:
                    weight_y[-ramp:] = ramp_values.flip(0)
                if x > 0:
                    weight_x[:ramp] = ramp_values
                if x + patch_w < width:
                    weight_x[-ramp:] = ramp_values.flip(0)

            weight = (weight_y[:, None] * weight_x[None, :])[None, None]
            output[..., y:y + patch_h, x:x + patch_w] += restored * weight
            weight_sum[..., y:y + patch_h, x:x + patch_w] += weight

    return (output / weight_sum.clamp_min(1e-8))[..., :original_h, :original_w]


def _image_to_tensor(image: Image.Image | np.ndarray) -> torch.Tensor:
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image))
    image = image.convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)


def _run_model_and_save(
    image_tensor: torch.Tensor,
    task: str,
    variant: str,
    checkpoint_path: Path,
    tile_size: int,
    output_tag: str,
) -> tuple[Image.Image, str]:
    model = _load_network(task, variant, str(checkpoint_path))
    with torch.inference_mode():
        if tile_size == 0:
            restored = _infer_full(model, image_tensor)
        else:
            restored = _infer_tiled(model, image_tensor, tile_size=tile_size, overlap=32)

    restored_array = (
        restored.clamp(0.0, 1.0)[0]
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )
    output_image = Image.fromarray(
        np.round(restored_array * 255.0).clip(0, 255).astype(np.uint8),
        mode="RGB",
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = RESULTS_DIR / f"{timestamp}_{uuid.uuid4().hex[:8]}_{output_tag}.png"
    output_image.save(output_path)
    return output_image, str(output_path)


def restore_image(
    image: Image.Image | np.ndarray | None,
    task: str,
    tile_size: int,
):
    """Run desnowing once, or run both ITS and OTS dehazing models."""
    if image is None:
        raise gr.Error("Please upload an image first.")

    tile_size = int(tile_size)
    if tile_size != 0 and (tile_size < 64 or tile_size % 4):
        raise gr.Error("Tile size must be 0 or a positive multiple of 4.")

    image_tensor = _image_to_tensor(image)

    try:
        if task == "Desnowing":
            checkpoint_path = DEFAULT_DESNOW_CHECKPOINT
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    "Desnowing checkpoint not found. Expected one of: "
                    + ", ".join(str(p.relative_to(ROOT_DIR)) for p in DESNOW_CHECKPOINT_CANDIDATES)
                )
            desnow_image, desnow_file = _run_model_and_save(
                image_tensor, "Desnowing", "", checkpoint_path, tile_size, "desnowed"
            )
            return desnow_image, desnow_file, None, None, None, None

        if task == "Dehazing":
            missing = [
                path for path in (DEFAULT_ITS_CHECKPOINT, DEFAULT_OTS_CHECKPOINT)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    "Both dehazing checkpoints are required. Missing: "
                    + ", ".join(str(path.relative_to(ROOT_DIR)) for path in missing)
                )

            # Run both separately trained networks on the same input. The loader
            # keeps only one network in memory at a time to limit GPU usage.
            its_image, its_file = _run_model_and_save(
                image_tensor, "Dehazing", "ITS (Indoor)",
                DEFAULT_ITS_CHECKPOINT, tile_size, "dehazed_its",
            )
            ots_image, ots_file = _run_model_and_save(
                image_tensor, "Dehazing", "OTS (Outdoor)",
                DEFAULT_OTS_CHECKPOINT, tile_size, "dehazed_ots",
            )
            return None, None, its_image, its_file, ots_image, ots_file

        raise gr.Error("Choose either Desnowing or Dehazing.")

    except torch.cuda.OutOfMemoryError as exc:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        raise gr.Error(
            "GPU memory is insufficient for this image. Try a smaller tile size, "
            "such as 256, or run on CPU."
        ) from exc
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Image restoration failed: {exc}") from exc


# -----------------------------------------------------------------------------
# Gradio interface
# -----------------------------------------------------------------------------

def _toggle_result_sections(selected_task: str):
    is_desnowing = selected_task == "Desnowing"
    return gr.update(visible=is_desnowing), gr.update(visible=not is_desnowing)


with gr.Blocks(title="FocalNet Image Restoration", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# FocalNet Image Restoration\n"
        "Upload an image and choose the restoration task. Desnowing uses its own "
        "checkpoint. Dehazing automatically runs both ITS and OTS checkpoints and "
        "shows both restored results. No classifier or checkpoint selection is used."
    )

    task = gr.Radio(
        choices=["Desnowing", "Dehazing"],
        value="Desnowing",
        label="Choose restoration task",
    )

    tile_size = gr.Dropdown(
        choices=[256, 512, 768, 0],
        value=512,
        label="Inference tile size (0 = process full image)",
    )

    input_image = gr.Image(type="pil", label="Input image")
    restore_button = gr.Button("Restore image", variant="primary")

    with gr.Row(visible=True) as desnow_results:
        with gr.Column():
            gr.Markdown("### Desnowing result")
            desnow_output = gr.Image(type="pil", label="Clean image")
            desnow_download = gr.File(label="Download desnowed image")

    with gr.Row(visible=False) as dehaze_results:
        with gr.Column():
            gr.Markdown("### Dehazing – ITS (Indoor)")
            its_output = gr.Image(type="pil", label="ITS restored image")
            its_download = gr.File(label="Download ITS result")
        with gr.Column():
            gr.Markdown("### Dehazing – OTS (Outdoor)")
            ots_output = gr.Image(type="pil", label="OTS restored image")
            ots_download = gr.File(label="Download OTS result")

    task.change(
        fn=_toggle_result_sections,
        inputs=task,
        outputs=[desnow_results, dehaze_results],
    )

    restore_button.click(
        fn=restore_image,
        inputs=[input_image, task, tile_size],
        outputs=[
            desnow_output, desnow_download,
            its_output, its_download,
            ots_output, ots_download,
        ],
    )

    gr.Markdown(
        "**Checkpoint files are loaded automatically from the project root:**\n"
        "- Desnowing: `checkpoints/Best.pth` (falls back to `model.pth` or `Final.pth`)\n"
        "- Dehazing ITS: `checkpoints/its_model.pkl`\n"
        "- Dehazing OTS: `checkpoints/ots_model.pkl`\n\n"
        "Dehazing runs both models on the same input. Results are saved in "
        "`results/unified_inference/`."
    )


if __name__ == "__main__":
    print(f"FocalNet unified inference device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    demo.launch()
