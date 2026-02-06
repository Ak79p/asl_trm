import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig


# -------------------------
# Dataset
# -------------------------
class ASLDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, dataset_root):
        self.df = pd.read_csv(csv_path)
        self.dataset_root = Path(dataset_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = self.dataset_root / feature_path

        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature file: {feature_path}")

        x = torch.load(feature_path)   # (T, D)
        y = int(row["class_id"])

        return x.float(), y



# -------------------------
# Metrics
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
# Train / Eval
# -------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


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
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["asl100", "asl300", "asl1000", "asl2000"],
        help="Dataset to train on"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out_dir", default="checkpoints")
    args = parser.parse_args()

    # -------------------------
    # Dataset config
    # -------------------------
    cfg = DatasetConfig(args.dataset)
    print(f"\n📦 Using dataset: {cfg}")
    print(f"   Classes: {cfg.num_classes}\n")

    # -------------------------
    # Device
    # -------------------------
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🚀 Training on device: {device}\n")

    # -------------------------
    # Data
    # -------------------------
    train_ds = ASLDataset(cfg.train_csv, cfg.root)
    val_ds   = ASLDataset(cfg.val_csv, cfg.root)
    # test_ds  = ASLDataset(cfg.test_csv, cfg.root)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # -------------------------
    # Model
    # -------------------------
    model = TRMMicro(num_classes=cfg.num_classes)
    model.to(device)

    print("Classifier shape:", model.classifier.weight.shape)

    # -------------------------
    # Optimizer / Loss
    # -------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # -------------------------
    # Training loop
    # -------------------------
    best_top1 = 0.0
    out_dir = Path(args.out_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"\n===== Epoch {epoch}/{args.epochs} =====")

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        val_loss, acc = eval_epoch(
            model, val_loader, criterion, device
        )

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Top-1: {acc[1]*100:.2f}% | "
            f"Top-5: {acc[5]*100:.2f}% | "
            f"Top-10: {acc[10]*100:.2f}%"
        )

        if acc[1] > best_top1:
            best_top1 = acc[1]
            torch.save(
                model.state_dict(),
                out_dir / "best_model.pt"
            )
            print("✅ Saved new best model")

    print(f"\n🏁 Training complete. Best Top-1: {best_top1*100:.2f}%\n")


if __name__ == "__main__":
    main()
