import math

import torch
import torch.nn.functional as F

from data.data_load import valid_dataloader


def _valid(model, args, ep=None):
    device = next(model.parameters()).device
    validation_source = getattr(args, "validation_roots", None) or getattr(args, "dataset_roots", None) or args.data_dir
    loader = valid_dataloader(validation_source, batch_size=1, num_workers=0)
    was_training = model.training
    model.eval()
    psnr_values = []

    with torch.inference_mode():
        for input_img, label_img in loader:
            input_img = input_img.to(device, dtype=torch.float32)
            label_img = label_img.to(device, dtype=torch.float32)

            h, w = input_img.shape[-2:]
            pad_h, pad_w = (-h) % 4, (-w) % 4
            if pad_h or pad_w:
                mode = "reflect" if h > pad_h and w > pad_w else "replicate"
                input_img = F.pad(input_img, (0, pad_w, 0, pad_h), mode=mode)

            # This FocalNet returns one full-resolution restored image tensor.
            pred = model(input_img)[..., :h, :w].clamp(0.0, 1.0)
            mse = F.mse_loss(pred, label_img).item()
            psnr_values.append(float("inf") if mse == 0 else -10.0 * math.log10(mse))

    if was_training:
        model.train()

    if not psnr_values:
        raise RuntimeError("Validation set is empty. Check the dataset roots contain paired validation or test images.")

    return sum(psnr_values) / len(psnr_values)
