import os
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor


CACHE_DIR = Path("data/wlasl2000/features_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PREFIX = "features_cache"  


def extract_features(csv_path, out_csv_path):
    df = pd.read_csv(csv_path)

    print(f"\n🔨 Processing {len(df)} videos from {csv_path}\n")

    feature_paths = []
    extracted_count = 0
    reused_count = 0
    skipped_count = 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = Path(row["video_path"])
        video_id = video_path.stem
        out_path = CACHE_DIR / f"{video_id}.pt"

        csv_feature_path = f"{FEATURE_PREFIX}/{video_id}.pt"

        # If already exists → reuse
        if out_path.exists():
            reused_count += 1
            feature_paths.append(csv_feature_path)
            continue

        try:
            kps = extract_video_keypoints(video_path)
            X = build_feature_tensor(kps)

            torch.save(torch.from_numpy(X), out_path)

            extracted_count += 1
            feature_paths.append(csv_feature_path)

        except Exception as e:
            print(f"⚠️ Skipping {video_path.name}: {e}")
            skipped_count += 1
            feature_paths.append("")  # keep row alignment

    df["feature_path"] = feature_paths
    df.to_csv(out_csv_path, index=False)

    print("\n===== Summary =====")
    print(f"Reused existing: {reused_count}")
    print(f"Newly extracted: {extracted_count}")
    print(f"Skipped: {skipped_count}")
    print(f"✅ Saved updated CSV → {out_csv_path}\n")


if __name__ == "__main__":
    extract_features(
        "data/wlasl2000/train_labeled.csv",
        "data/wlasl2000/train_features.csv"
    )

    extract_features(
        "data/wlasl2000/val_labeled.csv",
        "data/wlasl2000/val_features.csv"
    )

    extract_features(
        "data/wlasl2000/test_labeled.csv",
        "data/wlasl2000/test_features.csv"
    )