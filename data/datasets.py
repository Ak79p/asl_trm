from pathlib import Path
import json


DATA_ROOT = Path(__file__).parent


class DatasetConfig:
    def __init__(self, name):
        self.name = name
        self.root = DATA_ROOT / name

        if not self.root.exists():
            raise ValueError(f"Dataset not found: {self.root}")

        self.label_map_path = self.root / "label_map.json"
        self.train_csv = self.root / "train_features.csv"
        self.val_csv = self.root / "val_features.csv"
        self.test_csv = self.root / "test_features.csv"
        self.feature_dir = self.root / "features_cache"

        with open(self.label_map_path) as f:
            self.label_map = json.load(f)

        self.num_classes = len(self.label_map)

    def __repr__(self):
        return (
            f"DatasetConfig(name={self.name}, "
            f"num_classes={self.num_classes})"
        )
