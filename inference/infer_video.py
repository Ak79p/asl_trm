import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor


# -------------------------
# Utils
# -------------------------
def load_label_maps(label_map):
    id_to_gloss = {v: k for k, v in label_map.items()}
    return id_to_gloss


def trim_classifier_if_needed(model, state, num_classes):
    ckpt_classes = state["classifier.weight"].shape[0]

    if ckpt_classes != num_classes:
        print(
            f"⚠️ Class mismatch: checkpoint={ckpt_classes}, dataset={num_classes}"
        )
        print("✂️ Trimming classifier weights")

        state["classifier.weight"] = state["classifier.weight"][:num_classes]
        state["classifier.bias"]   = state["classifier.bias"][:num_classes]

    model.load_state_dict(state, strict=False)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["asl100", "asl300", "asl1000", "asl2000"]
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override checkpoint path"
    )
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found: {video_path}"

    # -------------------------
    # Dataset config
    # -------------------------
    cfg = DatasetConfig(args.dataset)

    print(f"\n🎥 Inference on video: {video_path.name}")
    print(f"📚 Dataset: {cfg.name} ({cfg.num_classes} classes)\n")

    # -------------------------
    # Device
    # -------------------------
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🖥 Device: {device}")

    # -------------------------
    # Load model
    # -------------------------
    model = TRMMicro(num_classes=cfg.num_classes).to(device)

    ckpt_path = (
        args.checkpoint
        if args.checkpoint
        else Path("checkpoints") / cfg.name / "best_model.pt"
    )

    print(f"📦 Loading checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)
    trim_classifier_if_needed(model, state, cfg.num_classes)

    model.eval()

    # -------------------------
    # Label maps
    # -------------------------
    id_to_gloss = load_label_maps(cfg.label_map)

    # -------------------------
    # Feature extraction
    # -------------------------
    print("🔍 Extracting keypoints...")
    kps = extract_video_keypoints(video_path)

    print("🧱 Building feature tensor...")
    X = build_feature_tensor(kps)          # (T, J, D)
    X = torch.from_numpy(X).float()

    if X.ndim == 3:
        X = X.view(X.shape[0], -1)          # (T, J*D)

    X = X.unsqueeze(0).to(device)           # (1, T, D)

    # -------------------------
    # Forward
    # -------------------------
    with torch.no_grad():
        logits = model(X).squeeze(0)
        probs = F.softmax(logits, dim=0)

    # -------------------------
    # Top-K results
    # -------------------------
    topk = min(args.topk, cfg.num_classes)
    values, indices = probs.topk(topk)

    print("\n===== PREDICTIONS =====")
    for rank, (idx, score) in enumerate(zip(indices, values), 1):
        gloss = id_to_gloss[idx.item()]
        print(f"Top-{rank:<2} {gloss:<20} {score.item()*100:6.2f}%")

    print("=======================\n")


if __name__ == "__main__":
    main()
