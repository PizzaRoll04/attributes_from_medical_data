import pandas as pd
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils.nih_cxr import nih_cxr14_path_map, all_labels, encode_multilabel

class NIHCXRDataset(Dataset):
    def __init__(self, csv_path, img_root, img_transform=None, dataset_name="nih_cxr"):
        self.all_labels = all_labels(csv_path)
        self.label_to_idx = {label : i for i, label in enumerate(self.all_labels)}
        self.idx_to_label = {idx : label for idx, label in self.label_to_idx.items()}
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.img_transform = img_transform
        self.dataset_name = dataset_name

        self.sensitive_category_to_column = {"age" : "Patient Age", "gender" : "Patient Gender"}
        self.other_category_to_column = {"follow_up_num" : "Follow-up #", } 

        self.path_map = nih_cxr14_path_map(self.img_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.path_map[row["Image Index"]]
        img = Image.open(img_path).convert("RGB")
        if self.img_transform is not None:
            img = self.img_transform(img)

        y = encode_multilabel(row["Finding Labels"], self.label_to_idx)
        sensitive = {cat: row[col] for cat, col in self.sensitive_category_to_column}
        other = {cat: row[col] for cat, col in self.other_category_to_column}

        return {
            "image": img,
            "label_y": y,
            "sensitive": sensitive,  
            "other": other,  
            "dataset_id": self.dataset_name
        }
