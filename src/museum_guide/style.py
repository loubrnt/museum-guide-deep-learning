"""Helpers for the style task: metadata-only "floor" features and confusion matrices pooled over seeds.

The class set is not hard-coded anywhere here: the same code serves the 5-genre and the 6-style subsets.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

FLOOR_PARTS = ("genre", "shard", "format")


def metadata_floor_features(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Features that contain no pixel: 'genre', 'shard' (where the image sat in the source dataset), 'format'
    (log aspect ratio and log area of the original painting), or a '+' combination such as 'genre+format'.

    One-hot columns are built on the whole table, so train and test rows share the same columns.
    """
    parts = []
    for name in kind.split("+"):
        if name not in FLOOR_PARTS:
            raise ValueError(f"unknown floor feature {name!r}; choose from {FLOOR_PARTS}")
        if name == "format":
            w, h = df["orig_width"].to_numpy(float), df["orig_height"].to_numpy(float)
            parts.append(np.column_stack([np.log(w / h), np.log(w * h)]))
        else:
            parts.append(pd.get_dummies(df[name].astype(str)).to_numpy(float))
    return np.hstack(parts)


def pooled_confusion(pairs: Sequence[tuple[np.ndarray, np.ndarray]], n_classes: int) -> np.ndarray:
    """Counts matrix (rows = true class, columns = predicted) summed over several (y, prediction) pairs."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for y, pred in pairs:
        np.add.at(cm, (np.asarray(y), np.asarray(pred)), 1)
    return cm


def row_normalise(cm: np.ndarray) -> np.ndarray:
    """Each row divided by its sum (rows that are empty stay 0)."""
    cm = np.asarray(cm, dtype=float)
    return cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)


def top_confusions(cm: np.ndarray, names: Sequence[str], k: int = 5) -> list[dict]:
    """The k most frequent mistakes: true class, predicted class, count, and share of the true class's images."""
    cm = np.asarray(cm)
    totals = cm.sum(axis=1)
    off = [(i, j, int(cm[i, j])) for i in range(len(cm)) for j in range(len(cm)) if i != j and cm[i, j] > 0]
    off.sort(key=lambda t: -t[2])
    return [
        {"true": names[i], "predicted": names[j], "count": c, "share_of_true_class": float(c / totals[i])}
        for i, j, c in off[:k]
    ]
