"""Loading the WikiArt subset and splitting it without artist leakage.

A painter's hand (palette, brushwork, framing) is easy to memorise. If the same artist appears in
train and test, accuracy measures recognition of the artist, not of the genre. The museum will show
paintings by artists the model never saw, so the split is grouped by artist.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Relative to the working directory (run the scripts from the repository root), unless MUSEUM_GUIDE_DATA is set.
DATA_DIR = Path(os.environ.get("MUSEUM_GUIDE_DATA", "data/wikiart_genre5"))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_metadata(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "metadata.csv")
    # "Unknown Artist" is a catch-all, not one painter: give each such image its own group.
    df["group"] = np.where(df["artist"] == "Unknown Artist", "unknown_" + df["id"], df["artist"])
    return df


def split(df: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    """70 / 15 / 15 split with no artist shared between parts."""
    outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    train_idx, rest_idx = next(outer.split(df, groups=df["group"]))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_idx, test_idx = next(inner.split(rest, groups=rest["group"]))
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": rest.iloc[val_idx].reset_index(drop=True),
        "test": rest.iloc[test_idx].reset_index(drop=True),
    }


def class_names(df: pd.DataFrame) -> list[str]:
    return df.drop_duplicates("label_idx").sort_values("label_idx")["label"].tolist()


class WikiArt(Dataset):
    def __init__(self, df: pd.DataFrame, transform, data_dir: Path = DATA_DIR):
        self.paths = [data_dir / p for p in df["path"]]
        self.labels = torch.tensor(df["label_idx"].values)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as img:  # close the file handle: a loader opens thousands of images
            image = img.convert("RGB")
        return self.transform(image), self.labels[i]


def make_transforms(size: int, augment: bool) -> transforms.Compose:
    steps = []
    if augment:
        steps += [transforms.RandomResizedCrop(size, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip()]
    else:
        steps += [transforms.Resize(size), transforms.CenterCrop(size)]
    steps += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    return transforms.Compose(steps)


def loaders(parts: dict[str, pd.DataFrame], size: int, augment: bool, batch_size: int = 64) -> dict[str, DataLoader]:
    return {
        name: DataLoader(
            WikiArt(df, make_transforms(size, augment and name == "train")),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, df in parts.items()
    }


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
