"""Unified Gradio app for FocalNet desnowing and dehazing."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_MODEL_CACHE = {}


def default_checkpoint(*parts: str) -> str:
    """Return the first existing likely checkpoint, otherwise a useful default."""
    candidates = [ROOT.joinpath(*parts)]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return str(candidates[0])


DEFAULT_DESNOW = str(ROOT / "checkpoints" / "desnowing_best.pth")
DEFAULT_ITS = str(ROOT / "Dehazing" / "its" / "checkpoints" / "best_model.pkl")
DEFAULT_OTS = str(ROOT / "Dehazing" / "ots" / "checkpoints_ots" / "best_model.pkl")


def _load_torch_file(path: Path):
    try:
        return torch.load(path, map_location=DEVICE, weights_only=False)
    except TypeError:  # PyTorch versions before weights_only
        return torch.load(path, map_location=DEVICE)


def _extract_state(checkpoint, task: str):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a PyTorch state dictionary.")

    keys = ("model", "state_dict", "model_state_dict")
    for key in keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            state = value
            break
    else:
        state = checkpoint

    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    if not isinstance(state, dict) or not state:
        raise ValueError(f"No model weights were found in the {task} checkpoint.")
    return state


def _import_model(task: str, checkpoint):
    """Import each repository's local `models` package without name collisions."""
    # These projects each use a package named `models`; remove the previous
    # package before importing the selected implementation from its own folder.
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]

    if task == "Desnowing":
        model_dir = ROOT / "Desnowing"
        sys.path.insert(0, str(model_dir))
        try:
            module = importlib.import_module("models.focalnet")
            model = module.build_net()
        finally:
            sys.path.remove(str(model_dir))
    elif task == "Dehazing (ITS)":
        model_dir = ROOT / "Dehazing" / "its"
        sys.path.insert(0, str(model_dir))
        try:
            module = importlib.import_module("models.FocalNet")
            config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            model = module.FocalNet(
                width=int(config.get("width", 32)),
                num_res=int(config.get("num_res", 4)),
            )
        finally:
            sys.path.remove(str(model_dir))
    elif task == "Dehazing (OTS)":
        model_dir = ROOT / "Dehazing" / "ots"
        sys.path.insert(0, str(model_dir))
        try:
            module = importlib.import_module("models.focalnet")
            config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
            model = module.FocalNet(
                width=int(config.get("width", 32)),
                num_res=int(config.get("num_res", 4)),
            )
        finally:
            sys.path.remove(str(model_dir))
    else:
        raise ValueError(f"Unknown restoration task: {task}")
    return model


def get_model(task: str, checkpoint_path: str):
    path = Path(checkpoint_path).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Choose the trained checkpoint for the selected task. The ZIP you provided "
            "contains model code but does not include trained checkpoint files."
        )

    cache_key = (task, str(path), path.stat().st_mtime_ns)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    checkpoint = _load_torch_file(path)
    model = _import_model(task, checkpoint)
    state = _extract_state(checkpoint, task)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"The selected checkpoint is not compatible with {task}. "
            "Please select the checkpoint trained with this exact model implementation.\n"
            f"Details: {exc}"
        ) from exc

    model = model.to(DEVICE).eval()
    # Keep only the most recently used model in memory to avoid holding three
    # large networks in RAM/VRAM at once.
    _MODEL_CACHE.clear()
    _MODEL_CACHE[cache_key] = model
    return model


def _pad_to_multiple(tensor: torch.Tensor, factor: int = 4):
    height, width = tensor.shape[-2:]
    pad_h, pad_w = (-height) % factor, (-width) % factor
    if pad_h or pad_w:
        mode = "reflect" if height > pad_h and width > pad_w else "replicate"
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode=mode)
    return tensor, height, width


def restore_image(image, task, haze_variant, checkpoint_desnow, checkpoint_its, checkpoint_ots):
    if image is None:
        raise gr.Error("Please upload an image first.")
    if task == "Desnowing":
        model_task = "Desnowing"
        checkpoint_path = checkpoint_desnow
    elif task == "Dehazing":
        if haze_variant == "ITS (Indoor)":
            model_task = "Dehazing (ITS)"
            checkpoint_path = checkpoint_its
        else:
            model_task = "Dehazing (OTS)"
            checkpoint_path = checkpoint_ots
    else:
        raise gr.Error("Choose Desnowing or Dehazing.")

    try:
        model = get_model(model_task, checkpoint_path)
        rgb = image.convert("RGB") if isinstance(image, Image.Image) else Image.fromarray(np.asarray(image)).convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        padded, height, width = _pad_to_multiple(tensor)
        with torch.inference_mode():
            restored = model(padded)[..., :height, :width]
        restored = restored.clamp(0.0, 1.0)[0].cpu().permute(1, 2, 0).numpy()
        result = Image.fromarray(np.round(restored * 255.0).clip(0, 255).astype(np.uint8), mode="RGB")
        return rgb, result, f"Completed: {model_task} | Device: {DEVICE}"
    except torch.cuda.OutOfMemoryError as exc:
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        raise gr.Error("GPU memory is insufficient for this image. Try a smaller image or run on CPU.") from exc
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


def update_dehazing_visibility(task):
    return gr.update(visible=(task == "Dehazing"))


with gr.Blocks(title="Unified FocalNet Image Restoration") as demo:
    gr.Markdown(
        "# Unified Image Restoration\n"
        "Run your trained FocalNet models from one interface. Select the restoration task manually; "
        "the app loads only the checkpoint that belongs to that task."
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Input image")
            task = gr.Radio(
                choices=["Desnowing", "Dehazing"],
                value="Desnowing",
                label="Choose restoration task",
            )
            with gr.Group():
                checkpoint_desnow = gr.Textbox(
                    label="Desnowing checkpoint", value=DEFAULT_DESNOW,
                    info="Used only when Desnowing is selected.",
                )
            with gr.Group(visible=False) as dehaze_group:
                haze_variant = gr.Radio(
                    choices=["ITS (Indoor)", "OTS (Outdoor)"],
                    value="ITS (Indoor)",
                    label="Choose dehazing model",
                )
                checkpoint_its = gr.Textbox(
                    label="Dehazing ITS checkpoint", value=DEFAULT_ITS,
                    info="Indoor Training Set model.",
                )
                checkpoint_ots = gr.Textbox(
                    label="Dehazing OTS checkpoint", value=DEFAULT_OTS,
                    info="Outdoor Training Set model.",
                )
            run_button = gr.Button("Restore image", variant="primary")
            status = gr.Textbox(label="Status", interactive=False)
        with gr.Column():
            original_output = gr.Image(label="Original image", interactive=False)
            restored_output = gr.Image(label="Clean / restored image", interactive=False)

    task.change(update_dehazing_visibility, inputs=task, outputs=dehaze_group)
    run_button.click(
        restore_image,
        inputs=[input_image, task, haze_variant, checkpoint_desnow, checkpoint_its, checkpoint_ots],
        outputs=[original_output, restored_output, status],
    )
    gr.Markdown(
        "**Checkpoint note:** The project archive contains the model source code, not trained weights. "
        "Add your separate trained checkpoint files and set their paths above. "
        "When you choose Dehazing, select ITS (indoor) or OTS (outdoor); the app loads that variant's own checkpoint."
    )


if __name__ == "__main__":
    print(f"Running on: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    demo.launch(theme=gr.themes.Soft())
