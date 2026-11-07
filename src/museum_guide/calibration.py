"""Calibrated confidence and abstention: can the guide say "I am not sure"?

A fine-tuned network is usually over-confident: it says 90% when it is right 75% of the time. For a
kiosk, the confidence must mean something, so that low-confidence answers can be replaced by
"I am not sure". Temperature scaling divides the logits by one number T fitted on the validation set;
it does not change which class is predicted, only how sure the network claims to be.
"""
from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

DEFAULT_CALIBRATION = Path("reports/calibration.json")


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Row-wise softmax of `logits / temperature` (numerically stable)."""
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of the true class."""
    p = softmax(logits, temperature)
    return float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-12, None)).mean())


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE: confidence vs accuracy, averaged over `n_bins` equal-width confidence bins (weighted by bin size)."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    confidence, correct = probs.max(axis=1), probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> list[dict]:
    """Per confidence bin: its range, number of predictions, mean confidence and accuracy (for a reliability diagram)."""
    probs, labels = np.asarray(probs), np.asarray(labels)
    confidence, correct = probs.max(axis=1), probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        rows.append(
            {
                "low": float(lo),
                "high": float(hi),
                "n": int(in_bin.sum()),
                "confidence": float(confidence[in_bin].mean()) if in_bin.any() else None,
                "accuracy": float(correct[in_bin].mean()) if in_bin.any() else None,
            }
        )
    return rows


def fit_temperature(logits: np.ndarray, labels: np.ndarray, low: float = 0.05, high: float = 20.0) -> float:
    """The temperature T > 0 minimising the NLL of `logits / T` (fit it on a validation set, never on the test set).

    The NLL is convex in 1/T, hence unimodal in log T: a golden-section search finds the minimum, deterministically.
    """
    logits, labels = np.asarray(logits), np.asarray(labels)
    a, b = np.log(low), np.log(high)
    phi = (np.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(logits, labels, np.exp(c)), nll(logits, labels, np.exp(d))
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(logits, labels, np.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(logits, labels, np.exp(d))
    return float(np.exp((a + b) / 2))


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Calibrated probabilities."""
    return softmax(logits, temperature)


def selective_accuracy(confidence: np.ndarray, correct: np.ndarray, coverages: Sequence[float]) -> list[dict]:
    """Accuracy when only the `coverage` fraction of most confident predictions is answered.

    Ties at the threshold are kept, so the realised coverage can be slightly above the asked one.
    """
    confidence, correct = np.asarray(confidence), np.asarray(correct).astype(float)
    rows = []
    for cov in coverages:
        threshold = np.quantile(confidence, 1.0 - cov, method="lower") if cov < 1 else -np.inf
        answered = confidence >= threshold
        rows.append(
            {
                "coverage": float(cov),
                "realised_coverage": float(answered.mean()),
                "accuracy": float(correct[answered].mean()),
                "threshold": None if cov >= 1 else float(threshold),
            }
        )
    return rows


def confidence_threshold_for_accuracy(confidence: np.ndarray, correct: np.ndarray, target: float) -> float | None:
    """Smallest confidence threshold whose answered predictions reach accuracy `target` (None if none does).

    Choose it on a validation set; on the test set it gives an optimistic figure.
    """
    confidence, correct = np.asarray(confidence), np.asarray(correct).astype(float)
    for t in np.unique(confidence):
        answered = confidence >= t
        if correct[answered].mean() >= target:
            return float(t)
    return None


def load_temperature(path: Path = DEFAULT_CALIBRATION) -> float:
    """The temperature saved by `scripts/calibrate_resnet.py`, or 1.0 (no rescaling) if there is no such file."""
    path = Path(path)
    if not path.exists():
        return 1.0
    return float(json.loads(path.read_text())["temperature"])
