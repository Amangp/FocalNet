"""Paired image loaders for one or several desnowing datasets.

A combined dataset root may contain CSD, SRRS and Snow100K subfolders. Each
individual dataset uses paired input/ground-truth folders, for example:

    datasets/Snow100K/train2500/Snow/*
    datasets/Snow100K/train2500/Gt/*
    datasets/Snow100K/test2000/Snow/*  # final evaluation only
    datasets/Snow100K/test2000/Gt/*

If a dedicated val/valid/validation split exists, it is used. Otherwise, a
reproducible 10% holdout is carved out of train2500 (or the equivalent train
split), with no overlap between training and validation. test2000 remains
reserved for final evaluation.

Snow/Gt filenames are matched by exact filename first, then by filename stem
(to support pairs such as image.png and image.jpg).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision.transforms import functional as F

from .data_augment import PairCompose, PairRandomCrop, PairRandomHorizontalFlip, PairToTensor

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_INPUT_DIR_NAMES = ("snow", "snowy", "snow_images", "snow_image", "input", "inputs", "degraded", "snow100k-l", "snow100k-s")
_GT_DIR_NAMES = ("gt", "groundtruth", "ground_truth", "ground truth", "gt_images", "clean", "clean_images", "clear")
_TRAIN_SPLITS = ("train2500", "train", "training", "trainset", "train_set", "train_data")
_VALID_SPLITS = ("val", "valid", "validation", "val_set", "valid_set", "validation_set", "val_data")
_TEST_SPLITS = ("test2000", "test", "testing", "testset", "test_set", "test_data")


def _child_dir(root: Path, aliases):
    if not root.is_dir():
        return None
    children = {p.name.casefold(): p for p in root.iterdir() if p.is_dir()}
    for name in aliases:
        found = children.get(name.casefold())
        if found is not None:
            return found
    return None


def _pair_dirs(split_dir: Path):
    return _child_dir(split_dir, _INPUT_DIR_NAMES), _child_dir(split_dir, _GT_DIR_NAMES)


def _has_pairs(folder: Path) -> bool:
    inputs, labels = _pair_dirs(folder)
    return inputs is not None and labels is not None


def _select_split(root: Path, split: str):
    """Return (split directory, holdout_from_train).

    When no explicit validation split exists, validation is carved out of the
    training split; the held-out test split is never used for model selection.
    """
    if _has_pairs(root):
        return root, False

    if split == "train":
        candidates = _TRAIN_SPLITS
    elif split == "test":
        candidates = _TEST_SPLITS
    elif split == "valid":
        candidate = _child_dir(root, _VALID_SPLITS)
        if candidate is not None and _has_pairs(candidate):
            return candidate, False
        candidates = _TRAIN_SPLITS
    else:
        raise ValueError(f"Unknown split: {split}")

    candidate = _child_dir(root, candidates)
    if candidate is not None and _has_pairs(candidate):
        return candidate, split == "valid"

    expected = ", ".join(candidates)
    raise FileNotFoundError(
        f"Could not find a valid {split} split under {root}. Expected one of [{expected}], "
        "containing paired input and ground-truth image folders."
    )


def _has_explicit_valid(root: Path) -> bool:
    """True only when root has its own dedicated validation split."""
    candidate = _child_dir(root, _VALID_SPLITS)
    return candidate is not None and _has_pairs(candidate)


def _train_valid_indices(length: int, root: Path, validation_fraction: float = 0.1):
    """Stable 90/10 split based on sorted dataset indices."""
    if length < 2:
        raise ValueError(
            f"At least two paired training images are needed to create a validation holdout: {root}"
        )
    # Stable seed derived from the dataset folder name, not Python's randomized hash().
    stable_seed = 2026 + sum(root.name.casefold().encode("utf-8"))
    generator = torch.Generator().manual_seed(stable_seed)
    indices = torch.randperm(length, generator=generator).tolist()
    n_valid = max(1, int(round(length * validation_fraction)))
    n_valid = min(n_valid, length - 1)
    valid_indices = sorted(indices[:n_valid])
    train_indices = sorted(indices[n_valid:])
    return train_indices, valid_indices


def _has_split(root: Path, split: str) -> bool:
    try:
        _select_split(root, split)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def _normalise_roots(source, split: str):
    """Accept a dataset root, a parent containing dataset folders, or a list of roots."""
    if isinstance(source, (list, tuple)):
        supplied = [Path(p).expanduser() for p in source]
    elif source is None or str(source).strip() == "":
        raise ValueError("Provide --data-dir or --dataset-roots.")
    else:
        supplied = [Path(source).expanduser()]

    roots = []
    for path in supplied:
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")

        # A path that is itself a dataset (or split) takes precedence.
        if _has_split(path, split):
            roots.append(path)
            continue

        # Otherwise auto-discover dataset subdirectories, e.g. parent/CSD,
        # parent/SRRS and parent/Snow100K.
        if path.is_dir():
            children = sorted(
                child for child in path.iterdir()
                if child.is_dir() and _has_split(child, split)
            )
            if children:
                roots.extend(children)
                continue

        # Keep the original path so the split resolver can provide a useful error.
        roots.append(path)

    # Deduplicate while preserving order.
    unique = []
    seen = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def _image_files(folder: Path):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


class SnowDataset(Dataset):
    """Dataset of aligned snowy inputs and clean target images."""

    def __init__(self, split_dir, transform=None, is_test=False, dataset_name=""):
        self.split_dir = Path(split_dir)
        self.snow_dir, self.gt_dir = _pair_dirs(self.split_dir)
        if self.snow_dir is None or self.gt_dir is None:
            raise FileNotFoundError(
                f"Expected paired input and ground-truth folders under {self.split_dir}. "
                "Common folder names are Snow/ and Gt/."
            )

        input_files = _image_files(self.snow_dir)
        gt_files = _image_files(self.gt_dir)
        if not input_files:
            raise RuntimeError(f"No input images found in {self.snow_dir}")
        if not gt_files:
            raise RuntimeError(f"No ground-truth images found in {self.gt_dir}")

        gt_by_name = {p.name.casefold(): p for p in gt_files}
        gt_by_stem = {}
        for path in gt_files:
            gt_by_stem.setdefault(path.stem.casefold(), path)

        self.pairs = []
        missing = []
        for input_path in input_files:
            target_path = gt_by_name.get(input_path.name.casefold())
            if target_path is None:
                target_path = gt_by_stem.get(input_path.stem.casefold())
            if target_path is None:
                missing.append(input_path.name)
            else:
                self.pairs.append((input_path, target_path))

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} input image(s) in {self.snow_dir} have no matching ground truth "
                f"in {self.gt_dir}; examples: {', '.join(missing[:5])}"
            )
        if not self.pairs:
            raise RuntimeError(f"No paired images found in {self.split_dir}")

        self.transform = transform
        self.is_test = is_test
        self.dataset_name = dataset_name or self.split_dir.parent.name or self.split_dir.name

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        from PIL import Image

        with Image.open(input_path) as img:
            image = img.convert("RGB")
        with Image.open(target_path) as img:
            label = img.convert("RGB")

        if image.size != label.size:
            raise ValueError(
                f"Input/GT size mismatch for {input_path.name}: "
                f"{image.size} vs {label.size}"
            )

        if self.transform is not None:
            image, label = self.transform(image, label)
        else:
            image, label = F.to_tensor(image), F.to_tensor(label)

        if self.is_test:
            return image, label, f"{self.dataset_name}__{input_path.name}"
        return image, label


# Backward-compatible class name from the initial loader.
DeblurDataset = SnowDataset


def _make_datasets(source, split, transform=None, is_test=False):
    roots = _normalise_roots(source, split)
    datasets = []
    multiple = len(roots) > 1

    for root in roots:
        split_dir, holdout_from_train = _select_split(root, split)
        name = root.name if multiple else ""
        dataset_transform = transform if split == "train" else None
        dataset = SnowDataset(
            split_dir,
            transform=dataset_transform,
            is_test=is_test,
            dataset_name=name,
        )

        # If a dedicated validation folder is unavailable, split the training
        # set deterministically into disjoint train and validation subsets.
        if split == "train" and not _has_explicit_valid(root):
            train_indices, _ = _train_valid_indices(len(dataset), root)
            dataset = Subset(dataset, train_indices)
        elif split == "valid" and holdout_from_train:
            _, valid_indices = _train_valid_indices(len(dataset), root)
            dataset = Subset(dataset, valid_indices)

        datasets.append(dataset)

    for root, dataset in zip(roots, datasets):
        print(f"{split.title()} data: {root.name} | {len(dataset)} paired images")
    return roots, datasets


def train_dataloader(path, batch_size=64, num_workers=0, use_transform=True, balanced=True):
    transform = None
    if use_transform:
        transform = PairCompose([
            PairRandomCrop(256, pad_if_needed=True, padding_mode="reflect"),
            PairRandomHorizontalFlip(),
            PairToTensor(),
        ])

    roots, datasets = _make_datasets(path, "train", transform=transform)
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    sampler = None
    shuffle = True

    if balanced and len(datasets) > 1:
        # Give each dataset equal total sampling weight, so the largest dataset
        # does not dominate training just because it contains more images.
        sample_weights = []
        for dataset in datasets:
            sample_weights.extend([1.0 / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(
            torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(combined),
            replacement=True,
        )
        shuffle = False
        print(f"Combined training set: {len(datasets)} datasets | {len(combined)} total images | balanced sampling enabled")
    else:
        print(f"Training set: {len(combined)} total images")

    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def test_dataloader(path, batch_size=1, num_workers=0):
    _, datasets = _make_datasets(path, "test", is_test=True)
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def valid_dataloader(path, batch_size=1, num_workers=0):
    _, datasets = _make_datasets(path, "valid")
    combined = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
    return DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
