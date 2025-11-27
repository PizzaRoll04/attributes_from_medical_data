import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

class MultiModalMedicalDataset(Dataset):
    def __init__(self, csv_path, img_root, img_transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.img_transform = img_transform

        # Choose tabular features (excluding sensitive ones)
        self.tabular_cols = ["age", "bmi"]  # extend as needed
        # Sensitive attributes kept only for evaluation / auditing
        self.sensitive_cols = ["race", "gender"]

        # Preprocess: numeric encode/normalize as needed outside or here
        # (e.g., standardize age/bmi, map race/gender to integers for analysis)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.img_root / row["img_path"]
        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)

        # Tabular features (float tensor)
        tab = torch.tensor(row[self.tabular_cols].values.astype("float32"))

        # Main target (non-sensitive)
        y = torch.tensor(row["label_y"], dtype=torch.float32)

        # Sensitive attributes kept in batch for analysis only
        sensitive = {col: row[col] for col in self.sensitive_cols}

        return {
            "image": img,
            "tabular": tab,
            "label_y": y,
            "sensitive": sensitive,  # NOT used in forward loss for main task
        }
