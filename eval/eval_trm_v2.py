import argparse
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from models.trm_micro_v2 import TRMMicro
from data.datasets import DatasetConfig


# -------------------------
# Dataset (Aligned with Training)
# -------------------------
class ASLDataset(Dataset):
    def __init__(self, csv_path, dataset_root):
        self.dataset_root = Path(dataset_root)

        df = pd.read_csv(csv_path)
        original_len = len(df)

        df = df[df["feature_path"].notna()]
        df = df[df["feature_path"].apply(lambda x: isinstance(x, str))]

        valid_rows = []
        for _, row in df.iterrows():
            feature_path = Path(row["feature_path"])
            if not feature_path.is_absolute():
                feature_path = self.dataset_root / feature_path

            if feature_path.exists():
                valid_rows.append(row)

        self.df = pd.DataFrame(valid_rows).reset_index(drop=True)

        removed = original_len - len(self.df)
        if removed > 0:
            print(f"⚠️ Removed {removed} invalid samples from dataset")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        feature_path = Path(row["feature_path"])
        if not feature_path.is_absolute():
            feature_path = self.dataset_root / feature_path

        x = torch.load(feature_path).float()  # (T, J, D)
        y = int(row["class_id"])

        return x, y


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
# Utility: Find Latest Checkpoint
# -------------------------
def get_latest_checkpoint(dataset):
    ckpt_dir = Path("checkpoints") / f"trm_micro_{dataset}"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"No checkpoint directory found for {dataset}")

    ckpts = sorted(ckpt_dir.glob("best_model_*.pt"))
    if not ckpts:
        raise FileNotFoundError("No checkpoints found")

    return ckpts[-1]


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["asl100", "asl300", "asl1000", "asl2000", "asl-full", "wlasl100",
                 "wlasl300", "wlasl1000", "wlasl2000"]
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = DatasetConfig(args.dataset)

    print(f"\n🧪 Evaluating dataset: {args.dataset}")
    print(f"Classes: {cfg.num_classes}\n")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🖥 Device: {device}")

    # ---- Dataset ----
    test_ds = ASLDataset(cfg.test_csv, cfg.root)
    print(f"📊 Valid test samples: {len(test_ds)}")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # ---- Model ----
    model = TRMMicro(num_classes=cfg.num_classes).to(device)

    # ---- Load Checkpoint ----
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = get_latest_checkpoint(args.dataset)

    print(f"📦 Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    # STRICT loading (no silent mismatch)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    log_dir = checkpoint.get("log_dir", None)

    if log_dir:
        writer = SummaryWriter(log_dir=log_dir)
        print(f"📊 Logging test metrics to: {log_dir}")
    else:
        writer = None
        print("⚠️ No log_dir found in checkpoint — skipping TensorBoard logging")

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Evaluation
    # -------------------------
    total_loss = 0.0
    all_logits, all_targets = [], []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            all_logits.append(logits)
            all_targets.append(y)

    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)

    acc = accuracy(logits, targets)
    test_loss = total_loss / len(test_loader)

    # -------------------------
    # Report
    # -------------------------
    print("\n===== TEST RESULTS =====")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Top-1     : {acc[1]*100:.2f}%")
    print(f"Top-5     : {acc[5]*100:.2f}%")
    print(f"Top-10    : {acc[10]*100:.2f}%")
    print("========================\n")

    # ---- TensorBoard Logging ----
    if writer:
        writer.add_scalar("Loss/Test", test_loss, 0)
        writer.add_scalar("Accuracy/Test_Top1", acc[1], 0)
        writer.add_scalar("Accuracy/Test_Top5", acc[5], 0)
        writer.add_scalar("Accuracy/Test_Top10", acc[10], 0)
        writer.close()


if __name__ == "__main__":
    main()