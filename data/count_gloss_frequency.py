import pandas as pd
from pathlib import Path

# -------------------------
# Paths (update if needed)
# -------------------------
train_csv = Path("train_labeled.csv")
val_csv = Path("val_labeled.csv")
test_csv = Path("test_labeled.csv")

output_ranked = Path("gloss_ranked_2302.txt")
output_counts = Path("gloss_frequency.csv")

# -------------------------
# Load CSVs
# -------------------------
print("Loading CSV files...")

train_df = pd.read_csv(train_csv)
val_df = pd.read_csv(val_csv)
test_df = pd.read_csv(test_csv)

# -------------------------
# Combine splits
# -------------------------
full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

print(f"Total samples across splits: {len(full_df)}")

# -------------------------
# Count videos per gloss
# -------------------------
gloss_counts = (
    full_df
    .groupby("class_gloss")
    .size()
    .reset_index(name="video_count")
    .sort_values(by="video_count", ascending=False)
)

print(f"Total unique glosses found: {len(gloss_counts)}")

# -------------------------
# Save CSV with counts
# -------------------------
gloss_counts.to_csv(output_counts, index=False)
print(f"✔ Saved frequency CSV → {output_counts}")

# -------------------------
# Save ranked TXT (only gloss names)
# -------------------------
gloss_counts["class_gloss"].to_csv(
    output_ranked,
    index=False,
    header=False
)

print(f"✔ Saved ranked gloss list → {output_ranked}")
print("Done.")