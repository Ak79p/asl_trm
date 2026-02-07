import argparse
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import videotransforms
from pytorch_i3d import InceptionI3d

# Prefer the repo's "all" dataset (it returns scalar labels like test_i3d.py expects)
from datasets.nslt_dataset_all import NSLT as Dataset


def skip_none_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    return torch.utils.data.default_collate(batch)

def pick_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(mode: str, num_classes: int, weights: str, device: torch.device) -> InceptionI3d:
    # base imagenet weights
    if mode == "flow":
        i3d = InceptionI3d(400, in_channels=2)
        i3d.load_state_dict(torch.load("models/weights/flow_imagenet.pt", map_location="cpu"))
    else:
        i3d = InceptionI3d(400, in_channels=3)
        i3d.load_state_dict(torch.load("models/weights/rgb_imagenet.pt", map_location="cpu"))

    i3d.replace_logits(num_classes)

    ckpt = torch.load(weights, map_location="cpu")
    # if someone saved {"state_dict": ...}
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    # remove DataParallel prefix if present
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()} if isinstance(ckpt, dict) else ckpt

    i3d.load_state_dict(ckpt, strict=False)
    i3d.to(device)
    i3d.eval()
    return i3d


@torch.no_grad()
def predict_logits(i3d: InceptionI3d, inputs: torch.Tensor, segment_len: int = 64) -> torch.Tensor:
    """
    inputs: (B,C,T,H,W)
    returns: clip logits (B,C)
    """
    t = int(inputs.size(2))
    if t <= segment_len:
        per_frame_logits = i3d(inputs)  # (B,C,T')
        return torch.mean(per_frame_logits, dim=2)  # (B,C)

    # segment + average (matches test_i3d.py ensemble())
    num_segments = math.floor(t / segment_len)
    segments = [inputs[:, :, k * segment_len:(k + 1) * segment_len, :, :] for k in range(num_segments)]
    seg_tensor = torch.cat(segments, dim=0)  # (S*B,C,segment_len,H,W)

    per_frame_logits = i3d(seg_tensor)       # (S*B,C,T')
    seg_logits = torch.mean(per_frame_logits, dim=2)  # (S*B,C)

    # reshape to (S,B,C) then average over S -> (B,C)
    b = inputs.size(0)
    seg_logits = seg_logits.view(num_segments, b, -1).mean(dim=0)
    return seg_logits


def main():
    ap = argparse.ArgumentParser("I3D inference-only eval: Top-1/5/10 on WLASL split")
    ap.add_argument("--mode", default="rgb", choices=["rgb", "flow"])
    ap.add_argument("--num-classes", type=int, default=2000)
    ap.add_argument("--root", default="videos", help="Folder containing <video_id>.mp4")
    ap.add_argument("--split", default="code/I3D/preprocess/nslt_2000.json")
    ap.add_argument("--subset", default="test", help="test/val/train (whatever exists in json)")
    ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="auto", help="auto|cpu|mps|cuda")
    ap.add_argument("--segment-len", type=int, default=64)
    ap.add_argument("--max-videos", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    device = pick_device(args.device)
    print("Using device:", device)

    # deterministic eval transforms
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])

    dataset = Dataset(args.split, args.subset, args.root, args.mode, test_transforms)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=skip_none_collate,
    )

    i3d = build_model(args.mode, args.num_classes, args.weights, device)

    correct1 = 0
    correct5 = 0
    correct10 = 0
    total = 0

    # per-class accounting (optional)
    top1_fp = np.zeros(args.num_classes, dtype=np.int64)
    top1_tp = np.zeros(args.num_classes, dtype=np.int64)
    top5_fp = np.zeros(args.num_classes, dtype=np.int64)
    top5_tp = np.zeros(args.num_classes, dtype=np.int64)
    top10_fp = np.zeros(args.num_classes, dtype=np.int64)
    top10_tp = np.zeros(args.num_classes, dtype=np.int64)

    for batch in loader:
        if batch is None:
            continue
        inputs, labels, video_id = batch

        if args.max_videos and total >= args.max_videos:
            break

        inputs = inputs.to(device, non_blocking=True)
        y = int(labels[0].item())  # nslt_dataset_all returns scalar label

        logits = predict_logits(i3d, inputs, segment_len=args.segment_len)  # (1,C)
        logits = logits[0].detach().cpu().numpy()

        top10 = np.argsort(logits)[-10:]
        top5 = top10[-5:]
        top1 = top10[-1]

        total += 1

        if top1 == y:
            correct1 += 1
            top1_tp[y] += 1
        else:
            top1_fp[y] += 1

        if y in top5:
            correct5 += 1
            top5_tp[y] += 1
        else:
            top5_fp[y] += 1

        if y in top10:
            correct10 += 1
            top10_tp[y] += 1
        else:
            top10_fp[y] += 1

        if total % 50 == 0:
            print(f"[{total}] top1={correct1/total:.4f} top5={correct5/total:.4f} top10={correct10/total:.4f}")

    if total == 0:
        raise RuntimeError("Evaluated 0 samples. Check --subset and --split contents.")

    print("\n=== Final ===")
    print("Samples:", total)
    print("Top-1:", correct1 / total)
    print("Top-5:", correct5 / total)
    print("Top-10:", correct10 / total)

    # avoid div-by-zero for empty classes in subset
    top1_per_class = np.mean(top1_tp / (top1_tp + top1_fp + 1e-12))
    top5_per_class = np.mean(top5_tp / (top5_tp + top5_fp + 1e-12))
    top10_per_class = np.mean(top10_tp / (top10_tp + top10_fp + 1e-12))
    print("Per-class avg Top-1/5/10:", top1_per_class, top5_per_class, top10_per_class)


if __name__ == "__main__":
    main()