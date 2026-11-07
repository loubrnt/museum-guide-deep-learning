"""Learning curves: how much labelled data does a model need?

Two pieces of bookkeeping that the experiment script relies on, kept here so they can be tested:

- drawing a smaller training set from the artist-disjoint train split (`subsample_train`) without breaking
  the guarantee that paintings of one artist stay together, and `few_shot_indices` for k images per class;
- reading the learning curve: how many images a model needs to reach a target accuracy (`images_needed`).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def subsample_train(train: pd.DataFrame, n: int, seed: int, group_col: str = "group") -> np.ndarray:
    """Positions (rows of `train`) of a class-balanced subset of about `n` images, drawn by artist group.

    Every class gets a quota of round(n / number of classes) images (at least one). Inside a class, whole
    artist groups are added in a random order until the quota is reached; the last group is trimmed at random
    to hit the quota exactly. A class with fewer images than its quota gives all of them. So:

    - the size is K * round(n / K) for K classes (n itself when n is a multiple of K), or less if a class is small;
    - at most one artist per class is split, so the subset is as close to "whole artists" as the quota allows;
    - every class is represented (n >= K), which a plain random draw does not guarantee for small n.

    Deterministic given `seed`. Returns sorted positions.
    """
    labels = train["label_idx"].to_numpy()
    groups = train[group_col].to_numpy()
    classes = np.unique(labels)
    quota = max(1, int(round(n / len(classes))))
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for c in classes:
        rows = np.flatnonzero(labels == c)
        by_group: dict = {}
        for r in rows:
            by_group.setdefault(groups[r], []).append(r)
        order = list(by_group)
        rng.shuffle(order)
        picked: list[int] = []
        for g in order:
            if len(picked) >= quota:
                break
            members = np.array(by_group[g])
            room = quota - len(picked)
            if len(members) > room:
                members = rng.choice(members, size=room, replace=False)
            picked.extend(int(m) for m in members)
        chosen.extend(picked)
    return np.sort(np.array(chosen, dtype=int))


def few_shot_indices(labels: Sequence[int], k: int, seed: int) -> np.ndarray:
    """Positions of `k` randomly drawn examples per class (all of them for a class with fewer than k). Sorted."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(labels):
        rows = np.flatnonzero(labels == c)
        chosen.extend(rng.choice(rows, size=min(k, len(rows)), replace=False).tolist())
    return np.sort(np.array(chosen, dtype=int))


def images_needed(sizes: Sequence[float], accuracy: Sequence[float], target: float) -> dict:
    """Training-set size at which the mean curve first reaches `target`, by linear interpolation.

    Returns {"reached": bool, "images": float | None, "at_most": bool}. "at_most" is True when even the
    smallest size already reaches the target (the true need is then below the smallest size measured).
    """
    sizes = np.asarray(sizes, dtype=float)
    acc = np.asarray(accuracy, dtype=float)
    if len(sizes) != len(acc) or len(sizes) == 0:
        raise ValueError("sizes and accuracy must be non-empty and have the same length")
    order = np.argsort(sizes)
    sizes, acc = sizes[order], acc[order]
    if acc[0] >= target:
        return {"reached": True, "images": float(sizes[0]), "at_most": True}
    for i in range(1, len(sizes)):
        if acc[i] >= target:
            lo, hi = acc[i - 1], acc[i]
            frac = (target - lo) / (hi - lo)
            return {"reached": True, "images": float(sizes[i - 1] + frac * (sizes[i] - sizes[i - 1])), "at_most": False}
    return {"reached": False, "images": None, "at_most": False}
