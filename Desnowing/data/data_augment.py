import random

import torchvision.transforms as transforms
import torchvision.transforms.functional as F


class PairRandomCrop(transforms.RandomCrop):
    """Apply the same random crop to a degraded image and its ground truth."""

    def __call__(self, image, label):
        if self.padding is not None:
            image = F.pad(image, self.padding, self.fill, self.padding_mode)
            label = F.pad(label, self.padding, self.fill, self.padding_mode)

        # PIL size is (width, height).
        target_h, target_w = self.size
        if self.pad_if_needed and image.size[0] < target_w:
            pad_w = target_w - image.size[0]
            image = F.pad(image, (pad_w, 0), self.fill, self.padding_mode)
            label = F.pad(label, (pad_w, 0), self.fill, self.padding_mode)

        if self.pad_if_needed and image.size[1] < target_h:
            pad_h = target_h - image.size[1]
            image = F.pad(image, (0, pad_h), self.fill, self.padding_mode)
            label = F.pad(label, (0, pad_h), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(image, self.size)
        return F.crop(image, i, j, h, w), F.crop(label, i, j, h, w)


class PairCompose(transforms.Compose):
    """Apply each transform to an image/ground-truth pair."""

    def __call__(self, image, label):
        for transform in self.transforms:
            image, label = transform(image, label)
        return image, label


class PairRandomHorizontalFlip(transforms.RandomHorizontalFlip):
    """Randomly flip both images horizontally using the same decision."""

    def __call__(self, image, label):
        if random.random() < self.p:
            return F.hflip(image), F.hflip(label)
        return image, label


# Backward-compatible spelling used in the provided sample code.
PairRandomHorizontalFilp = PairRandomHorizontalFlip


class PairToTensor(transforms.ToTensor):
    """Convert both images to tensors in [0, 1]."""

    def __call__(self, image, label):
        return F.to_tensor(image), F.to_tensor(label)
