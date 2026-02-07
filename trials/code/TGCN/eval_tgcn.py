import os
import argparse
import numpy as np
import torch
from sklearn.metrics import accuracy_score

from configs import Config
from sign_dataset import Sign_Dataset
from tgcn_model import GCN_muti_att


def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i in range(truths.shape[0]):
        if truths[i] in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


@torch.no_grad()
def evaluate(model, loader, device, num_copies=4, max_batches=0):
    model.eval()

    all_y = []
    all_y_pred = []
    all_logits = []

    total_batches = 0

    for batch_idx, data in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break

        print(f"batch {batch_idx}")
        X, y, video_ids = data

        X = X.to(device)
        y = y.to(device).view(-1)

        # same multi-crop idea as your test file
        stride = X.size(2) // num_copies
        outputs = []
        for i in range(num_copies):
            X_slice = X[:, :, i * stride:(i + 1) * stride]
            outputs.append(model(X_slice))

        outputs = torch.stack(outputs, dim=1)        # (B, num_copies, C)
        logits = torch.mean(outputs, dim=1)          # (B, C)

        y_pred = logits.argmax(dim=1)

        all_y.append(y.detach().cpu())
        all_y_pred.append(y_pred.detach().cpu())
        all_logits.append(logits.detach().cpu())

        total_batches += 1

    all_y = torch.cat(all_y).numpy()
    all_y_pred = torch.cat(all_y_pred).numpy()
    all_logits = torch.cat(all_logits).numpy()

    top1 = accuracy_score(all_y, all_y_pred)
    top3 = compute_top_n_accuracy(all_y, all_logits, 3)
    top5 = compute_top_n_accuracy(all_y, all_logits, 5)
    top10 = compute_top_n_accuracy(all_y, all_logits, 10)

    print("\n=== Final ===")
    print("Samples:", len(all_y))
    print("Top-1:", top1)
    print("Top-3:", top3)
    print("Top-5:", top5)
    print("Top-10:", top10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".", help="repo root that contains WLASL/ and code/")
    ap.add_argument("--trained-on", default="asl2000", choices=["asl100", "asl2000"])
    ap.add_argument("--subset", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--batch-size", type=int, default=0, help="0 uses ini file")
    ap.add_argument("--max-batches", type=int, default=0, help="0 = all")
    ap.add_argument("--checkpoint", default="ckpt.pth")
    args = ap.parse_args()

    repo_root = os.path.abspath(args.repo_root)

    # You likely have repo_root/WLASL/...
    wlasl_root = os.path.join(repo_root, "WLASL")

    split_file = os.path.join(wlasl_root, "data", "splits", f"{args.trained_on}.json")
    pose_data_root = os.path.join(wlasl_root, "data", "pose_per_individual_videos")

    # NOTE: your repo uses "archived" for asl2000 and sometimes "archive" for asl100 in older scripts.
    # Try archived first.
    config_file = os.path.join(repo_root, "code", "TGCN", "archived", args.trained_on, f"{args.trained_on}.ini")
    if not os.path.exists(config_file):
        config_file = os.path.join(repo_root, "code", "TGCN", "archive", args.trained_on, f"{args.trained_on}.ini")

    ckpt_path = os.path.join(repo_root, "code", "TGCN", "archived", args.trained_on, args.checkpoint)
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(repo_root, "code", "TGCN", "archive", args.trained_on, args.checkpoint)

    assert os.path.exists(split_file), f"Missing split_file: {split_file}"
    assert os.path.exists(pose_data_root), f"Missing pose_data_root: {pose_data_root}"
    assert os.path.exists(config_file), f"Missing config_file: {config_file}"
    assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"

    configs = Config(config_file)

    num_samples = configs.num_samples
    hidden_size = configs.hidden_size
    drop_p = configs.drop_p
    num_stages = configs.num_stages
    batch_size = args.batch_size or configs.batch_size

    device = pick_device(args.device)
    print("Using device:", device)

    dataset = Sign_Dataset(
        index_file_path=split_file,
        split=args.subset,
        pose_root=pose_data_root,
        img_transforms=None,
        video_transforms=None,
        num_samples=num_samples,
        sample_strategy="k_copies",
        test_index_file=split_file,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    num_classes = int(args.trained_on[3:])  # asl2000 -> 2000
    model = GCN_muti_att(
        input_feature=num_samples * 2,
        hidden_feature=hidden_size,
        num_class=num_classes,
        p_dropout=drop_p,
        num_stage=num_stages,
    ).to(device)

    print("Loading model:", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu")
    # remove DataParallel prefix if present
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)

    evaluate(model, loader, device=device, num_copies=4, max_batches=args.max_batches)


if __name__ == "__main__":
    main()