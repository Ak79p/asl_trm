import argparse
from pathlib import Path
import torch
import torch.nn.functional as F

from models.trm_micro import TRMMicro
from data.datasets import DatasetConfig
from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor


# -------------------------
# Robust checkpoint loader
# -------------------------
def load_checkpoint_safely(path, device):
    raw_state = torch.load(path, map_location=device)

    print("\n🔎 RAW CHECKPOINT KEYS:", list(raw_state.keys()))

    # Correct key detection
    if "model_state" in raw_state:
        state = raw_state["model_state"]
    elif "model_state_dict" in raw_state:
        state = raw_state["model_state_dict"]
    elif "state_dict" in raw_state:
        state = raw_state["state_dict"]
    elif "model" in raw_state:
        state = raw_state["model"]
    else:
        state = raw_state

    print("\n🔎 STATE DICT SAMPLE KEYS:", list(state.keys())[:10])

    return state


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["asl100", "asl300", "asl1000", "asl2000"]
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found: {video_path}"

    cfg = DatasetConfig(args.dataset)

    print(f"\n🎥 Video: {video_path.name}")
    print(f"📚 Dataset: {cfg.name} ({cfg.num_classes} classes)\n")

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"🖥 Device: {device}")

    # -------------------------
    # Model
    # -------------------------
    model = TRMMicro(num_classes=cfg.num_classes).to(device)

    ckpt_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else Path("checkpoints") / cfg.name / "best_model.pt"
    )

    print(f"📦 Loading checkpoint: {ckpt_path}")

    state = load_checkpoint_safely(ckpt_path, device)

    # Check classifier sizes
    if "classifier.weight" in state:
        print("\n🔎 Checkpoint classifier shape:",
              state["classifier.weight"].shape)
    else:
        print("\n⚠️ classifier.weight not found in checkpoint!")

    print("🔎 Model classifier shape:",
          model.classifier.weight.shape)

    try:
        model.load_state_dict(state, strict=False)
        print("✅ Model loaded successfully.\n")
    except Exception as e:
        print("❌ Error loading state dict:", e)
        return

    model.eval()

    # -------------------------
    # Feature extraction
    # -------------------------
    print("🔍 Extracting keypoints...")
    kps = extract_video_keypoints(video_path)
    print("🔎 Raw keypoints shape:", kps.shape)

    print("🧱 Building feature tensor...")
    X = build_feature_tensor(kps)
    print("🔎 Feature tensor shape (before torch):", X.shape)

    X = torch.from_numpy(X).float()

    if X.ndim == 3:
        print("🔎 Flattening (T, J, D) → (T, J*D)")
        X = X.view(X.shape[0], -1)

    print("🔎 Final input shape before batch:", X.shape)

    X = X.unsqueeze(0).to(device)
    print("🔎 Input shape fed to model:", X.shape)

    # -------------------------
    # Forward
    # -------------------------
    with torch.no_grad():
        logits = model(X).squeeze(0)
        probs = F.softmax(logits, dim=0)

    print("\n🔎 Logits shape:", logits.shape)

    # -------------------------
    # Top-K
    # -------------------------
    topk = min(args.topk, cfg.num_classes)
    values, indices = probs.topk(topk)

    id_to_gloss = {v: k for k, v in cfg.label_map.items()}

    print("\n===== PREDICTIONS =====")
    for rank, (idx, score) in enumerate(zip(indices, values), 1):
        gloss = id_to_gloss.get(idx.item(), "UNKNOWN")
        print(f"Top-{rank:<2} {gloss:<20} {score.item()*100:6.2f}%")
    print("=======================\n")


if __name__ == "__main__":
    main()