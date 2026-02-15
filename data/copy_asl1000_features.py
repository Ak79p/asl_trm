import json
import shutil
import pandas as pd
from pathlib import Path

# ===== PATH CONFIG =====
LABEL_MAP_1000 = "label_map.json"

SRC_TRAIN_CSV = "/Users/amit/Desktop/SLD/asl_trm/data/asl2000/train_features.csv"
SRC_VAL_CSV = "/Users/amit/Desktop/SLD/asl_trm/data/asl2000/val_features.csv"
SRC_TEST_CSV = "/Users/amit/Desktop/SLD/asl_trm/data/asl2000/test_features.csv"

SRC_CACHE_DIR = Path("/Users/amit/Desktop/SLD/asl_trm/data/asl2000/features_cache")
DST_ROOT = Path("features_cache")
DST_CACHE_DIR = DST_ROOT
DST_CACHE_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_PREFIX = "features_cache"
# =======================


def load_label_map(path):
    with open(path, "r") as f:
        return json.load(f)


def process_split(csv_path, split_name, label_map):
    df = pd.read_csv(csv_path)

    valid_glosses = set(label_map.keys())
    new_rows = []

    copied = 0
    already_exists = 0
    missing_source = 0

    print(f"\nProcessing {split_name} split...")

    for _, row in df.iterrows():
        gloss = str(row["class_gloss"]).strip().lower()

        if gloss not in valid_glosses:
            continue

        # 🔥 Extract video_id from video_path instead (safe)
        video_id = Path(row["video_path"]).stem

        src_file = SRC_CACHE_DIR / f"{video_id}.pt"
        dst_file = DST_CACHE_DIR / f"{video_id}.pt"

        # Copy tensor if needed
        if dst_file.exists():
            already_exists += 1
        else:
            if src_file.exists():
                shutil.copy(src_file, dst_file)
                copied += 1
            else:
                missing_source += 1
                continue

        new_class_id = label_map[gloss]

        new_rows.append([
            row["video_path"],
            gloss,
            new_class_id,
            f"{FEATURE_PREFIX}/{video_id}.pt"
        ])

    out_csv = f"{split_name}_features.csv"

    out_df = pd.DataFrame(
        new_rows,
        columns=["video_path", "class_gloss", "class_id", "feature_path"]
    )

    out_df.to_csv(out_csv, index=False)

    print(f"✔ Saved {out_csv} ({len(new_rows)} samples)")
    print(f"   Copied: {copied}")
    print(f"   Already existed: {already_exists}")
    print(f"   Missing tensors: {missing_source}")



def main():
    print("Loading label_map1000...")
    label_map = load_label_map(LABEL_MAP_1000)

    print(f"Total ASL1000 classes: {len(label_map)}")

    process_split(SRC_TRAIN_CSV, "train", label_map)
    process_split(SRC_VAL_CSV, "val", label_map)
    process_split(SRC_TEST_CSV, "test", label_map)

    print("\n===== ASL1000 subset creation complete =====")


if __name__ == "__main__":
    main()
