"""How much can we trust the numbers? Confidence intervals, and robustness to image shifts.

The test set is small (a few hundred images), so an accuracy is only known to a few points. The
intervals are computed by resampling *artists* (clusters), not single images: the split is grouped by
artist, so paintings by one painter are not independent observations.
"""
from __future__ import annotations

import numpy as np
import torch


def _group_sums(values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sum of `values` and per-group count, groups in sorted order."""
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return np.bincount(inverse, weights=values.astype(float)), counts.astype(float)


def bootstrap_accuracy(
    correct: np.ndarray, groups: np.ndarray | None = None, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Accuracy with a percentile bootstrap interval: (accuracy, low, high).

    `correct` holds 1 for a correct prediction, else 0. With `groups` (one id per image, e.g. the
    artist) whole groups are resampled; without, single images are.
    """
    correct = np.asarray(correct)
    groups = np.arange(len(correct)) if groups is None else np.asarray(groups)
    hits, counts = _group_sums(correct, groups)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(hits), size=(n_boot, len(hits)))
    accs = hits[draw].sum(1) / counts[draw].sum(1)
    lo, hi = np.quantile(accs, [alpha / 2, 1 - alpha / 2])
    return float(correct.mean()), float(lo), float(hi)


def bootstrap_difference(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    groups: np.ndarray | None = None,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Accuracy of model A minus model B on the *same* images, with a paired bootstrap interval.

    If the interval excludes 0, the gap is unlikely to be a sampling accident of this test set.
    """
    a, b = np.asarray(correct_a), np.asarray(correct_b)
    groups = np.arange(len(a)) if groups is None else np.asarray(groups)
    hits_a, counts = _group_sums(a, groups)
    hits_b, _ = _group_sums(b, groups)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(counts), size=(n_boot, len(counts)))
    diffs = (hits_a[draw].sum(1) - hits_b[draw].sum(1)) / counts[draw].sum(1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def mean_std(values) -> tuple[float, float]:
    """Mean and sample standard deviation (ddof=1; 0.0 for a single value)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def shift_pixels(size: int, fraction: float) -> int:
    """A shift given as a fraction of the image width, in pixels (so 64 px and 128 px models are comparable)."""
    return max(1, round(fraction * size))


@torch.no_grad()
def predict_with_shift(model, loader, device, pixels: int):
    """Labels, predictions, and predictions on the same images circularly shifted by `pixels` (down and right).

    Same perturbation as the shift test of `scripts/train_mlp.py`. Returns three numpy arrays.
    """
    model.eval().to(device)
    ys, preds, shifted = [], [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).argmax(1).cpu())
        shifted.append(model(torch.roll(x, shifts=(pixels, pixels), dims=(2, 3))).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy(), torch.cat(shifted).numpy()
