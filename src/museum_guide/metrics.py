"""Per-class metrics and error analysis: beyond a single accuracy number."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def per_class_report(y: np.ndarray, pred: np.ndarray, classes: Sequence[str]) -> dict:
    """Precision, recall, F1 and support per class, plus accuracy and macro-F1.

    A class that is never predicted gets precision 0 (not an error); every class in `classes` is listed
    even if it is absent from `y`.
    """
    y, pred = np.asarray(y), np.asarray(pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y, pred, labels=range(len(classes)), zero_division=0
    )
    return {
        "accuracy": float((y == pred).mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            c: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for c, p, r, f, s in zip(classes, prec, rec, f1, support)
        },
    }


def accuracy_by_group(y: np.ndarray, pred: np.ndarray, groups: np.ndarray, min_count: int = 1) -> dict:
    """Accuracy and number of images for each value of `groups`, skipping groups with fewer than `min_count` images."""
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    out = {}
    for g in sorted(set(groups.tolist())):
        mask = groups == g
        if mask.sum() >= min_count:
            out[g] = {"n": int(mask.sum()), "acc": float((y[mask] == pred[mask]).mean())}
    return out


def most_confident_errors(probs: np.ndarray, y: np.ndarray, k: int = 12) -> np.ndarray:
    """Indices of the `k` wrong predictions made with the highest confidence, most confident first."""
    probs, y = np.asarray(probs), np.asarray(y)
    pred, conf = probs.argmax(1), probs.max(1)
    wrong = np.flatnonzero(pred != y)
    return wrong[np.argsort(-conf[wrong], kind="stable")][:k]


def least_confident_correct(probs: np.ndarray, y: np.ndarray, k: int = 12) -> np.ndarray:
    """Indices of the `k` correct predictions made with the lowest confidence, least confident first."""
    probs, y = np.asarray(probs), np.asarray(y)
    pred, conf = probs.argmax(1), probs.max(1)
    right = np.flatnonzero(pred == y)
    return right[np.argsort(conf[right], kind="stable")][:k]
