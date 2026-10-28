"""Train FocalNet on paired hazy/clear images."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models import FocalNet
from data.data_load import PairedHazeDataset
from valid import validate
from utils import save_checkpoint, load_checkpoint


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--hazy-dir", required=True)
    p.add_argument("--clear-dir", required=True)
    p.add_argument("--val-hazy-dir", default=None)
    p.add_argument("--val-clear-dir", default=None)
    p.add_argument("--output-dir", default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--num-res", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every", type=int, default=1)
    return p.parse_args()


def main():
    a = args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    train_ds = PairedHazeDataset(a.hazy_dir, a.clear_dir, a.patch_size, augment=True)
    train_loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True,
                              num_workers=a.workers, pin_memory=(device.type == "cuda"))
    val_loader = None
    if a.val_hazy_dir and a.val_clear_dir:
        val_ds = PairedHazeDataset(a.val_hazy_dir, a.val_clear_dir, None, augment=False)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    model = FocalNet(a.width, a.num_res).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    criterion = torch.nn.L1Loss()
    start_epoch, best_psnr = 0, float("-inf")
    if a.resume:
        start_epoch, best_psnr = load_checkpoint(a.resume, model, optimizer, device)
        print(f"Resumed from epoch {start_epoch}, best PSNR={best_psnr:.3f}")
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, a.epochs):
        model.train()
        total = 0.0
        for hazy, clear, _ in train_loader:
            hazy, clear = hazy.to(device), clear.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(hazy)
            loss = criterion(pred, clear)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg_loss = total / max(1, len(train_loader))
        score = validate(model, val_loader, device) if val_loader else float("nan")
        print(f"Epoch {epoch+1}/{a.epochs} | L1={avg_loss:.6f} | Val PSNR={score:.3f}")
        save_checkpoint(out / "last_checkpoint.pkl", model, optimizer, epoch+1,
                        best_psnr, {"loss": avg_loss})
        if val_loader and score > best_psnr:
            best_psnr = score
            save_checkpoint(out / "best_model.pkl", model, optimizer, epoch+1,
                            best_psnr, {"loss": avg_loss})
        elif not val_loader and (epoch + 1) % max(1, a.save_every) == 0:
            save_checkpoint(out / f"epoch_{epoch+1:04d}.pkl", model, optimizer,
                            epoch+1, best_psnr, {"loss": avg_loss})
    print("Training finished. Checkpoints:", out.resolve())


if __name__ == "__main__":
    main()
