"""Shared fixtures: a synthetic metadata table and tiny image folders, so no real data is needed."""
import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def metadata() -> pd.DataFrame:
    """300 images, 5 genres, 30 named artists (10 images each) and 90 'Unknown Artist' images."""
    rng = np.random.default_rng(0)
    n_named, n_unknown = 210, 90
    artists = [f"artist_{i:02d}" for i in range(30)]
    df = pd.DataFrame(
        {
            "id": [f"img{i:04d}" for i in range(n_named + n_unknown)],
            "artist": [artists[i // 7] for i in range(n_named)] + ["Unknown Artist"] * n_unknown,
            "label_idx": rng.integers(0, 5, n_named + n_unknown),
        }
    )
    names = ["cityscape", "landscape", "portrait", "religious_painting", "still_life"]
    df["label"] = df["label_idx"].map(dict(enumerate(names)))
    df["path"] = [f"{lab}/{i}.jpg" for lab, i in zip(df["label"], df["id"])]
    return df


@pytest.fixture
def image_folder(tmp_path):
    """A real on-disk mini dataset (12 small JPEGs + metadata.csv) laid out like the WikiArt subset."""
    rows = []
    for i in range(12):
        label, label_idx = ("portrait", 0) if i % 2 == 0 else ("landscape", 1)
        rel = f"{label}/img{i:02d}.jpg"
        (tmp_path / label).mkdir(exist_ok=True)
        Image.fromarray(np.full((40, 30, 3), i * 20, dtype=np.uint8)).save(tmp_path / rel)
        rows.append(
            {
                "path": rel,
                "label": label,
                "label_idx": label_idx,
                "id": f"img{i:02d}",
                "artist": "Unknown Artist" if i < 6 else f"painter_{i}",
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)
    return tmp_path
