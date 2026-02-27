import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor_continuous

# --------------------------------------------------
# Sliding Window (Top-10 per stride)
# --------------------------------------------------
def sliding_window_predict(
    model,
    full_tensor,
    idx_to_gloss,
    window_size=48,
    stride=4,
    device="cuda",
    topk=10
):
    model.eval()
    T_full = full_tensor.shape[0]

    print(f"\nTotal Frames: {T_full}")
    print(f"Window Size : {window_size}")
    print(f"Stride      : {stride}\n")

    with torch.no_grad():

        for start in range(0, T_full - window_size + 1, stride):

            end = start + window_size
            window = full_tensor[start:end]

            window_t = torch.from_numpy(window).float().unsqueeze(0).to(device)

            logits = model(window_t).squeeze(0)
            probs = F.softmax(logits, dim=0)

            values, indices = probs.topk(topk)

            print("--------------------------------------------------")
            print(f"Window {start}-{end}")

            for rank, (idx, score) in enumerate(zip(indices, values), 1):
                word = idx_to_gloss.get(idx.item(), "UNKNOWN")
                print(f"{rank:>2}. {word:<20} {score.item()*100:6.2f}%")

    print("\n================ END ================\n")

# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--dataset", required=True,
                        choices=["asl100", "asl300", "asl1000", "asl2000", "app"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()

    cfg = DatasetConfig(args.dataset)

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")
    print(f"Dataset: {args.dataset}")
    print(f"Checkpoint: {args.checkpoint}")

    # Load model
    model = TRMMicro(num_classes=cfg.num_classes).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    idx_to_gloss = {v: k for k, v in cfg.label_map.items()}

    # Extract keypoints
    print("\nExtracting keypoints...")
    keypoints = extract_video_keypoints(args.video)

    print("Building feature tensor...")
    full_tensor = build_feature_tensor_continuous(keypoints)

    print(f"Feature tensor shape: {full_tensor.shape}")

    # Sliding window
    sliding_window_predict(
        model=model,
        full_tensor=full_tensor,
        idx_to_gloss=idx_to_gloss,
        window_size=args.window,
        stride=args.stride,
        device=device,
        topk=10
    )


if __name__ == "__main__":
    main()