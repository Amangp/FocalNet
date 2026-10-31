"""Paired RESIDE-OTS dataset loader for FocalNet."""

from pathlib import Path
import random

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _collect_images(folder):
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Directory not found: {folder}")

    return {
        path.name: path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    }


class PairedHazeDataset(Dataset):
    """
    Match RESIDE-OTS hazy images to clear images by image ID.

    Example:
        Hazy: 0001_0.85_0.04.jpg
        Clear: 0001.jpg
    """

    def __init__(
        self,
        hazy_dir,
        clear_dir,
        patch_size=256,
        training=True
    ):
        hazy = _collect_images(hazy_dir)
        clear = _collect_images(clear_dir)

        self.pairs = []

        for hazy_name, hazy_path in sorted(hazy.items()):
            image_id = hazy_path.stem.split("_")[0]

            clear_path = clear.get(image_id + ".jpg")

            if clear_path is None:
                clear_path = clear.get(image_id + ".png")

            if clear_path is None:
                clear_path = clear.get(image_id + ".jpeg")

            if clear_path is not None:
                self.pairs.append((hazy_path, clear_path))

        if not self.pairs:
            raise RuntimeError("No matching hazy/clear image pairs found.")

        self.patch_size = patch_size
        self.training = training

        print(f"Loaded {len(self.pairs)} paired images.")

    @staticmethod
    def _read(path):
        with Image.open(path) as image:
            array = np.asarray(
                image.convert("RGB"),
                dtype=np.float32
            ) / 255.0

        return torch.from_numpy(array).permute(2, 0, 1).contiguous()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        hazy_path, clear_path = self.pairs[index]

        hazy = self._read(hazy_path)
        clear = self._read(clear_path)

        if hazy.shape != clear.shape:
            raise ValueError(
                f"Image dimensions differ: "
                f"{hazy_path.name} {tuple(hazy.shape)} vs "
                f"{clear_path.name} {tuple(clear.shape)}"
            )

        if self.training:
            height, width = hazy.shape[-2:]
            patch = self.patch_size

            if height < patch or width < patch:
                raise ValueError(
                    f"Image too small for {patch}x{patch} crop: "
                    f"{hazy_path.name}"
                )

            top = random.randint(0, height - patch)
            left = random.randint(0, width - patch)

            hazy = hazy[:, top:top + patch, left:left + patch]
            clear = clear[:, top:top + patch, left:left + patch]

            # Paper-specified random horizontal flip (p = 0.5)
            if random.random() < 0.5:
                hazy = hazy.flip(-1)
                clear = clear.flip(-1)

        return hazy, clear, hazy_path.name