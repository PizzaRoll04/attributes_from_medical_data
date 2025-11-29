import pandas as pd
from pathlib import Path
from PIL import Image

import skimage.io
import numpy as np
import torchxrayvision as xrv
import torch
from torch.utils.data import Dataset

from classifier.utils.nih_cxr import nih_cxr14_path_map, all_labels, encode_multilabel


class NIHCXRDataset(Dataset):
    gender_map = {"M": 0, "F": 1}

    def __init__(
        self,
        csv_path,
        img_root,
        img_transform=None,
        split="train",
        dataset_name="nih_cxr",
        views=["PA"],  # this default is what torchxrayvision does
        train_frac=0.7,
        val_frac=0.1,
        seed=0,
    ):
        assert split in {"train", "val", "test"}

        self.all_labels = all_labels(csv_path)
        self.label_to_idx = {label: i for i, label in enumerate(self.all_labels)}
        self.idx_to_label = {idx: label for idx, label in self.label_to_idx.items()}
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.img_transform = img_transform
        self.dataset_name = dataset_name

        # filter views
        if views is not None:
            if not isinstance(views, (list, tuple, set)):
                views = [views]
            self.df = self.df[self.df["View Position"].isin(views)]

        # do the data split
        rng = np.random.RandomState(seed)

        patient_ids = self.df["Patient ID"].values
        unique_pids = np.unique(patient_ids)
        rng.shuffle(unique_pids)

        n = len(unique_pids)
        n_train = int(train_frac * n)
        n_val = int(val_frac * n)
        n_test = n - n_train - n_val

        train_pids = set(unique_pids[:n_train])
        val_pids = set(unique_pids[n_train : n_train + n_val])
        test_pids = set(unique_pids[n_train + n_val :])

        if split == "train":
            keep_pids = train_pids
        elif split == "val":
            keep_pids = val_pids
        else:  # "test"
            keep_pids = test_pids

        self.df = self.df[self.df["Patient ID"].isin(keep_pids)].reset_index(drop=True)

        self.sensitive_category_to_column = {
            "age": "Patient Age",
            "gender": "Patient Gender",
        }
        self.other_category_to_column = {
            "follow_up_num": "Follow-up #",
            "patient_id": "Patient ID",
            "view_position": "View Position",
            "width": "OriginalImage[Width",
            "height": "Height]",
        }

        self.path_map = nih_cxr14_path_map(self.img_root)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.path_map[row["Image Index"]]
        # img = Image.open(img_path).convert("RGB")

        # img formatting/processing from torchxrayvision pretrained model
        img = skimage.io.imread(img_path)
        img = xrv.datasets.normalize(img, 255)
        if len(img.shape) == 3:
            img = img.mean(2)

        img = img[None, ...]

        if self.img_transform is not None:
            img = self.img_transform(img)
        img = torch.from_numpy(img).float()

        y = encode_multilabel(row["Finding Labels"], self.label_to_idx)
        sensitive = {
            "age": torch.tensor(
                row[self.sensitive_category_to_column["age"]], dtype=torch.float32
            ),
            "gender": torch.tensor(
                self.gender_map[row[self.sensitive_category_to_column["gender"]]],
                dtype=torch.float32,
            ),
        }
        other = {cat: row[col] for cat, col in self.other_category_to_column.items()}

        return {
            "image": img,
            "label_y": y,
            "sensitive": sensitive,
            "other": other,
            "dataset_id": self.dataset_name,
        }
