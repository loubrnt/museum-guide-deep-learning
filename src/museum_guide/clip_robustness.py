"""Measuring the CLIP head (CLIP image encoder + linear probe) under camera-like conditions.

The image vectors of every perturbed copy of the collection are computed once and cached; the probe is
then trained and tested on vectors, so a whole study costs a few embedding passes. This module holds the
reusable parts: shifting the picture the encoder sees, and the statistics a kiosk cares about (how often it
answers, how accurate the answers are, how confident it is when it is wrong).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from museum_guide.robustness import mean_std


def shift_pixel_values(pixel_values, fraction: float):
    """Circularly shift a batch of preprocessed images (N, 3, H, W) down and right by `fraction` of the width.

    Same perturbation as `museum_guide.robustness.predict_with_shift`; it is applied to the tensor the encoder
    actually sees (for CLIP, the 224 x 224 input), so 6.25% is 14 pixels and 12.5% is 28 pixels.
    """
    import torch

    pixels = max(1, round(fraction * pixel_values.shape[-1]))
    return torch.roll(pixel_values, shifts=(pixels, pixels), dims=(2, 3))


def embed_shifted(model, processor, images: Sequence, fraction: float, batch_size: int = 64) -> np.ndarray:
    """Normalised CLIP vectors of PIL images after preprocessing and a circular shift of `fraction` of the width."""
    import torch

    from museum_guide.clip import _as_tensor, normalise

    images = [img.convert("RGB") for img in images]
    vecs = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            pixel_values = processor(images=images[i : i + batch_size], return_tensors="pt")["pixel_values"]
            out = _as_tensor(model.get_image_features(pixel_values=shift_pixel_values(pixel_values, fraction)))
            vecs.append(out.cpu().numpy())
    return normalise(np.concatenate(vecs))


def answer_stats(confidence: np.ndarray, correct: np.ndarray, threshold: float | None) -> dict:
    """What a kiosk with an 'I am not sure' threshold does on a set of photos.

    `confidence` is the top probability, `correct` whether the top class is right. An image is answered when
    its confidence is at least `threshold` (None: no threshold, everything is answered).
    """
    confidence, correct = np.asarray(confidence, dtype=float), np.asarray(correct, dtype=bool)
    answered = np.ones(len(correct), dtype=bool) if threshold is None else confidence >= threshold
    wrong = ~correct
    return {
        "mean_confidence": float(confidence.mean()),
        "mean_confidence_when_wrong": float(confidence[wrong].mean()) if wrong.any() else None,
        "answered_share": float(answered.mean()),
        "accuracy_when_answered": float(correct[answered].mean()) if answered.any() else None,
    }


def change_rate(pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    """Share of images whose predicted class differs between two conditions."""
    return float((np.asarray(pred_a) != np.asarray(pred_b)).mean())


def paired_summary(a: Sequence[float], b: Sequence[float]) -> dict:
    """Per-seed paired comparison of two models on the same splits: mean difference a - b, its spread, wins."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "mean_diff": float(diff.mean()),
        "std_diff": mean_std(diff)[1],
        "seeds_a_better": int((diff > 0).sum()),
        "seeds_b_better": int((diff < 0).sum()),
        "n_seeds": int(diff.size),
    }


def mean_over_seeds(per_seed: Sequence[dict[str, float]], keys: Sequence[str]) -> dict[str, dict[str, float]]:
    """{key: {mean, std}} over the per-seed dictionaries."""
    return {k: dict(zip(("mean", "std"), mean_std([row[k] for row in per_seed]))) for k in keys}
