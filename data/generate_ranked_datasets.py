import pandas as pd
import json
from pathlib import Path

# -------------------------
# CONFIG
# -------------------------
ranked_gloss_file = Path("gloss_ranked_2302.txt")

train_csv = Path("train_labeled.csv")
val_csv = Path("val_labeled.csv")
test_csv = Path("test_labeled.csv")

features_root = Path("features_cache")

output_root = Path(".")

dataset_sizes = [100, 300, 1000, 2000]

# -------------------------
# Load ranked gloss list
# -------------------------
with open(ranked_gloss_file) as f:
    ranked_glosses = [line.strip() for line in f.readlines()]

print(f"Total ranked glosses: {len(ranked_glosses)}")

# -------------------------
# Load full CSV splits
# -------------------------
train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# -------------------------
# Function to generate dataset
# -------------------------
def generate_dataset(top_n):
    dataset_name = f"asl{top_n}"
    print(f"\nGenerating {dataset_name}...")

    selected_glosses = ranked_glosses[:top_n]

    # Create label_map
    label_map = {gloss: idx for idx, gloss in enumerate(selected_glosses)}

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(exist_ok=True)

    # Save label_map.json
    with open(dataset_dir / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print("✔ label_map.json saved")

    # Filter and remap function
    def filter_and_save(split_df, split_name):
        df = split_df[split_df["class_gloss"].isin(selected_glosses)].copy()

        # Remap class_id
        df["class_id"] = df["class_gloss"].map(label_map)

        # Keep only rows where feature exists
        def feature_exists(row):
            feature_path = features_root / Path(row["video_path"]).with_suffix(".pt").name
            return feature_path.exists()

        df = df[df.apply(feature_exists, axis=1)]

        # Build correct feature_path column
        df["feature_path"] = df["video_path"].apply(
            lambda x: f"features_cache/{Path(x).with_suffix('.pt').name}"
        )

        output_csv = dataset_dir / f"{split_name}_features.csv"
        df.to_csv(output_csv, index=False)

        print(f"✔ {split_name}_features.csv saved ({len(df)} samples)")

    filter_and_save(train_df, "train")
    filter_and_save(val_df, "val")
    filter_and_save(test_df, "test")

# -------------------------
# Generate all datasets
# -------------------------
for size in dataset_sizes:
    generate_dataset(size)

print("\nAll datasets generated successfully.")