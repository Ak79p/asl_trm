import os
import pandas as pd

CSV_FILES = [
    "asl-full/train_features.csv",
    "asl-full/val_features.csv",
    "asl-full/test_features.csv",
]

BASE_FEATURE_DIR = "."

def check_features(csv_path):
    print(f"\nChecking: {csv_path}")
    df = pd.read_csv(csv_path)

    missing_files = []
    total = len(df)

    for _, row in df.iterrows():
        feature_path = os.path.join(BASE_FEATURE_DIR, row["feature_path"])

        if not os.path.exists(feature_path):
            missing_files.append(feature_path)

    print(f"Total entries: {total}")
    print(f"Missing features: {len(missing_files)}")

    if missing_files:
        print("\nMissing file paths:")
        for path in missing_files:
            print(path)

    return len(missing_files)


def main():
    total_missing = 0

    for csv_file in CSV_FILES:
        total_missing += check_features(csv_file)

    print("\n==============================")
    if total_missing == 0:
        print("All feature files exist. Dataset integrity verified.")
    else:
        print(f"Total missing feature files: {total_missing}")
        print("Fix these before training.")


if __name__ == "__main__":
    main()