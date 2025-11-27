import os
import torch
import pandas as pd
from pathlib import Path
from collections import defaultdict

def nih_cxr14_path_map(root_dir, exts={".png", ".jpg", ".jpeg"}):
    """
    Recursively walks through NIH ChestXray14 directory structure
    and returns a map: filename -> full_path.

    root_dir: path to the directory containing all NIH images
    exts: accepted file extensions
    """
    root_dir = Path(root_dir)

    path_map = {}
    duplicates = defaultdict(list)

    for path in root_dir.rglob("*"):
        if path.suffix.lower() in exts:
            fname = path.name

            if fname in path_map:
                # store duplicates to warn later
                duplicates[fname].append(str(path))
            else:
                path_map[fname] = str(path)

    # Warn about duplicates
    if duplicates:
        print("WARNING: Duplicate filenames detected!")
        for fname, paths in duplicates.items():
            print(f"  {fname}:")
            print(f"    original: {path_map[fname]}")
            for p in paths:
                print(f"    duplicate: {p}")

    print(f"Indexed {len(path_map)} unique image files")
    return path_map

def all_labels(path_csv:Path) -> list[str]:
    df = pd.read_csv(path_csv)
    all_labels = set()
    for raw in df["Finding Labels"].dropna():
        for label in str(raw).split("|"):
            label = label.strip()
            if label != "":
                all_labels.add(label)
    return sorted(all_labels)

def encode_multilabel(label_str : str, label_to_idx : dict):
    labs = [s.strip() for s in label_str.split("|")]
    y = torch.zeros(len(label_to_idx), dtype=torch.float32)
    for lab in labs:
        if lab in label_to_idx:
            y[label_to_idx[lab]] = 1.0
    return y
