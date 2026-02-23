import pandas as pd
import shutil
from pathlib import Path
import argparse

# -------------------------
# Argument Parser
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    help="Dataset folder name (e.g., asl100)")
parser.add_argument("--global_features", type=str, default="features_cache",
                    help="Global features_cache directory")

args = parser.parse_args()

dataset_dir = Path(args.dataset)
global_features_dir = Path(args.global_features)
dataset_features_dir = dataset_dir / "features_cache"

train_csv = dataset_dir / "train_features.csv"
val_csv = dataset_dir / "val_features.csv"
test_csv = dataset_dir / "test_features.csv"

# -------------------------
# Validate paths
# -------------------------
if not dataset_dir.exists():
    raise FileNotFoundError(f"{dataset_dir} not found")

for csv_file in [train_csv, val_csv, test_csv]:
    if not csv_file.exists():
        raise FileNotFoundError(f"{csv_file} not found")

if not global_features_dir.exists():
    raise FileNotFoundError(f"{global_features_dir} not found")

# -------------------------
# Create destination folder
# -------------------------
dataset_features_dir.mkdir(exist_ok=True)

# -------------------------
# Collect required features
# -------------------------
print("Reading feature CSV files...")

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

required_features = set(
    Path(fp).name for fp in full_df["feature_path"].dropna()
)

print(f"Total required feature files: {len(required_features)}")

# -------------------------
# Copy files
# -------------------------
copied = 0
already_exists = 0
missing = 0

for feature_name in required_features:
    src = global_features_dir / feature_name
    dst = dataset_features_dir / feature_name

    if not src.exists():
        missing += 1
        continue

    if dst.exists():
        already_exists += 1
        continue

    shutil.copy2(src, dst)
    copied += 1

print("\n✔ Feature cache build complete.")
print(f"Copied: {copied}")
print(f"Already existed: {already_exists}")
print(f"Missing in global cache: {missing}")