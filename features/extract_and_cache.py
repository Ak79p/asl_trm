import os
import pandas as pd
import torch
from tqdm import tqdm
from pathlib import Path

from features.video_to_keypoints import extract_video_keypoints
from features.build_tensor import build_feature_tensor


CACHE_DIR = Path("a/features_cache")
CACHE_DIR.mkdir(exist_ok=True)

def extract_features(csv_path, out_csv_path):
    df = pd.read_csv(csv_path)
    feature_paths = []

    print(f"\n🔨 Extracting features for {len(df)} videos\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):
        video_path = row["video_path"]
        video_id = Path(video_path).stem  # filename without .mp4
        out_path = CACHE_DIR / f"{video_id}.pt"

        # reuse if already exists
        if out_path.exists():
            feature_paths.append(str(out_path))
            continue

        kps = extract_video_keypoints(video_path)
        X = build_feature_tensor(kps)

        torch.save(torch.from_numpy(X), out_path)
        feature_paths.append(str(out_path))

    df["feature_path"] = feature_paths
    df.to_csv(out_csv_path, index=False)

    print(f"\n✅ Saved updated CSV → {out_csv_path}")

if __name__ == "__main__":
    extract_features(
        "a/ASL_Citizen/train_labeled.csv",
        "a/ASL_Citizen/train_features.csv"
    )
    extract_features(
        "a/ASL_Citizen/val_labeled.csv",
        "a/ASL_Citizen/val_features.csv"
    )
    extract_features(
        "a/ASL_Citizen/test_labeled.csv",
        "a/ASL_Citizen/test_features.csv"
    )
