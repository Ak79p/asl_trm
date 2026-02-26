import argparse
import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
import pandas as pd
from tqdm import tqdm

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig


# -------------------------
# Dataset
# -------------------------
class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, dataset_root, training=False):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = Path(dataset_root)
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = self.dataset_root / feature_path

        x = torch.load(feature_path).float()  # (T, D)
        y = int(row["class_id"])

        if self.training:
            # ---- Random temporal shift (motion invariance) ----
            shift = torch.randint(-4, 5, (1,)).item()
            x = torch.roll(x, shifts=shift, dims=0)

            # ---- Small Gaussian noise ----
            noise = 0.01 * torch.randn_like(x)
            x = x + noise

        return x, y


# -------------------------
# Accuracy
# -------------------------
@torch.no_grad()
def accuracy(logits, targets, topk=(1, 5, 10)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[k] = (correct_k / targets.size(0)).item()

    return res


# -------------------------
# Train
# -------------------------
def train_epoch(model, loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with torch.autocast(device_type=device, enabled=(device != "cpu")):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(loader)


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        all_logits.append(logits)
        all_targets.append(y)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    acc = accuracy(logits, targets)
    return total_loss / len(loader), acc


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=["asl100", "asl300", "asl1000", "asl2000"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", default="checkpoints")
    args = parser.parse_args()

    cfg = DatasetConfig(args.dataset)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\n🚀 Device: {device}")
    print(f"📦 Dataset: {args.dataset} ({cfg.num_classes} classes)\n")

    # ---- Data ----
    train_ds = ASLDataset(cfg.train_csv, cfg.root, training=True)
    val_ds   = ASLDataset(cfg.val_csv, cfg.root, training=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ---- Model ----
    model = TRMMicro(num_classes=cfg.num_classes)
    model.to(device)

    # ---- Class weighting ----
    class_counts = train_ds.df["class_id"].value_counts().sort_index()
    weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
    weights = weights / weights.sum() * len(weights)
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # ---- Warmup + Cosine Scheduler ----
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))))

    scheduler = LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler(enabled=(device != "cpu"))

    # ---- Logging ----
    best_top1 = 0.0
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_dir = Path("runs") / f"trm_micro_{args.dataset}" / timestamp
    writer = SummaryWriter(log_dir=str(log_dir))

    ckpt_dir = Path(args.out_dir) / f"trm_micro_{args.dataset}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / f"best_model_{timestamp}.pt"

    # ---- Training Loop ----
    for epoch in range(1, args.epochs + 1):

        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, scaler
        )

        val_loss, acc = eval_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Top1: {acc[1]*100:.2f}% | "
            f"Top5: {acc[5]*100:.2f}% | "
            f"Top10: {acc[10]*100:.2f}%"
        )

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Top1", acc[1], epoch)

        if acc[1] > best_top1:
            best_top1 = acc[1]
            torch.save({
                "model_state": model.state_dict(),
                "val_top1": best_top1,
                "dataset": args.dataset
            }, checkpoint_path)
            print("✅ Saved new best model")

    writer.close()

    print(f"\n🏁 Best Top-1: {best_top1*100:.2f}%\n")


if __name__ == "__main__":
    main()