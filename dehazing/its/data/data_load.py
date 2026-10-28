
"""Paired hazy/clear dataset loader for RESIDE-Indoor (ITS)."""
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import pil_to_tensor, convert_image_dtype

from .data_augment import augment_pair


EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(folder):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {folder}")

    return {
        p.name: p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTS
    }


def _scene_id(hazy_path):
    """1000_1_0.99837.png -> 1000.png scene ID."""
    return hazy_path.name.split("_", 1)[0]


class PairedHazeDataset(Dataset):
    def __init__(self, hazy_dir, clear_dir, patch_size=256, augment=False):
        self.hazy_dir = Path(hazy_dir)
        self.clear_dir = Path(clear_dir)
        hazy_images = _images(self.hazy_dir)
        clear_images = _images(self.clear_dir)

        if not hazy_images:
            raise RuntimeError(f"No images found in hazy folder: {self.hazy_dir}")
        if not clear_images:
            raise RuntimeError(f"No images found in clear folder: {self.clear_dir}")

        self.pairs = []
        missing = []

        for hazy_name, hazy_path in sorted(hazy_images.items()):
            scene_id = _scene_id(hazy_path)
            clear_name = f"{scene_id}{hazy_path.suffix.lower()}"

            clear_path = clear_images.get(clear_name)
            if clear_path is None:
                # Permit matching a scene ID when extensions differ.
                candidates = [
                    p for name, p in clear_images.items()
                    if Path(name).stem == scene_id
                ]
                clear_path = candidates[0] if candidates else None

            if clear_path is None:
                missing.append(hazy_name)
            else:
                self.pairs.append((hazy_path, clear_path))

        if not self.pairs:
            example = ", ".join(list(hazy_images)[:5])
            raise RuntimeError(
                "No hazy/clear pairs could be created. "
                f"Example hazy filenames: {example}"
            )

        if missing:
            print(
                f"Warning: {len(missing)} hazy images have no matching clear scene "
                f"and will be skipped. Example: {missing[:3]}"
            )

        self.patch_size = patch_size
        self.augment = augment
        print(
            f"Loaded {len(self.pairs)} pairs from "
            f"{self.hazy_dir} and {self.clear_dir}"
        )

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def _load(path):
        with Image.open(path) as im:
            im = im.convert("RGB")
            return convert_image_dtype(pil_to_tensor(im), torch.float32)

    def __getitem__(self, index):
        hazy_path, clear_path = self.pairs[index]
        hazy = self._load(hazy_path)
        clear = self._load(clear_path)

        if hazy.shape != clear.shape:
            raise ValueError(
                f"Image sizes differ: {hazy_path.name} {tuple(hazy.shape)} vs "
                f"{clear_path.name} {tuple(clear.shape)}"
            )

        size = self.patch_size
        if size and hazy.shape[-2] >= size and hazy.shape[-1] >= size:
            if self.augment:
                top = random.randint(0, hazy.shape[-2] - size)
                left = random.randint(0, hazy.shape[-1] - size)
            else:
                top = (hazy.shape[-2] - size) // 2
                left = (hazy.shape[-1] - size) // 2

            hazy = hazy[:, top:top + size, left:left + size]
            clear = clear[:, top:top + size, left:left + size]

        if self.augment:
            hazy, clear = augment_pair(hazy, clear)

        return hazy, clear, hazy_path.name
