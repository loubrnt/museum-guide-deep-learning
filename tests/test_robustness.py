import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from museum_guide.robustness import (
    bootstrap_accuracy,
    bootstrap_difference,
    mean_std,
    predict_with_shift,
    shift_pixels,
)


def test_bootstrap_accuracy_contains_the_point_estimate_and_is_ordered():
    correct = np.array([1] * 70 + [0] * 30)
    acc, lo, hi = bootstrap_accuracy(correct, seed=1)
    assert acc == pytest.approx(0.7)
    assert lo < acc < hi
    assert 0.55 < lo and hi < 0.85  # about +-9 points for n=100


def test_bootstrap_interval_shrinks_with_more_data():
    rng = np.random.default_rng(0)
    small = (rng.random(100) < 0.7).astype(int)
    large = (rng.random(2000) < 0.7).astype(int)
    _, lo_s, hi_s = bootstrap_accuracy(small)
    _, lo_l, hi_l = bootstrap_accuracy(large)
    assert (hi_l - lo_l) < (hi_s - lo_s)


def test_bootstrap_perfect_and_constant_predictions_have_no_spread():
    assert bootstrap_accuracy(np.ones(50)) == (1.0, 1.0, 1.0)
    assert bootstrap_accuracy(np.zeros(50)) == (0.0, 0.0, 0.0)


def test_cluster_bootstrap_is_wider_than_image_bootstrap_when_groups_are_homogeneous():
    # 20 artists with 10 paintings each; an artist is either fully right or fully wrong.
    groups = np.repeat(np.arange(20), 10)
    correct = np.repeat((np.arange(20) < 14).astype(int), 10)
    _, lo_i, hi_i = bootstrap_accuracy(correct)
    _, lo_c, hi_c = bootstrap_accuracy(correct, groups)
    assert (hi_c - lo_c) > 2 * (hi_i - lo_i)


def test_bootstrap_difference_detects_a_clear_gap_and_ignores_a_tie():
    rng = np.random.default_rng(0)
    a = (rng.random(300) < 0.8).astype(int)
    b = (rng.random(300) < 0.5).astype(int)
    diff, lo, hi = bootstrap_difference(a, b)
    assert diff == pytest.approx(a.mean() - b.mean())
    assert lo > 0  # A is clearly better
    _, lo_t, hi_t = bootstrap_difference(a, a)
    assert (lo_t, hi_t) == (0.0, 0.0)


def test_bootstrap_difference_is_paired():
    # Identical errors on the same images: the difference is exactly 0 however noisy each accuracy is.
    a = np.array([1, 0] * 50)
    diff, lo, hi = bootstrap_difference(a, a.copy(), groups=np.arange(100) // 5)
    assert (diff, lo, hi) == (0.0, 0.0, 0.0)


def test_mean_std():
    assert mean_std([1.0, 2.0, 3.0]) == (2.0, 1.0)
    assert mean_std([0.5]) == (0.5, 0.0)


def test_shift_pixels_scales_with_image_size():
    assert shift_pixels(64, 0.125) == 8
    assert shift_pixels(128, 0.125) == 16
    assert shift_pixels(4, 0.01) == 1  # never 0


class _PositionDetector(nn.Module):
    """Predicts class 1 iff the top-left pixel is bright: a model that depends on absolute position."""

    def forward(self, x):
        return torch.stack([torch.zeros(len(x)), x[:, 0, 0, 0]], dim=1)


class _GlobalMean(nn.Module):
    """Predicts from the image mean: invariant to a circular shift."""

    def forward(self, x):
        m = x.mean(dim=(1, 2, 3))
        return torch.stack([torch.zeros_like(m), m], dim=1)


def _loader():
    x = torch.zeros(8, 3, 8, 8)
    x[:, :, 0, 0] = 1.0  # bright top-left corner on every image
    return DataLoader(TensorDataset(x, torch.ones(8, dtype=torch.long)), batch_size=4)


def test_predict_with_shift_changes_a_position_dependent_model():
    y, pred, shifted = predict_with_shift(_PositionDetector(), _loader(), torch.device("cpu"), pixels=2)
    assert y.shape == pred.shape == shifted.shape == (8,)
    assert (pred == 1).all()  # corner pixel bright: class 1
    assert (shifted == 0).all()  # the bright pixel moved away: prediction flips
    assert (pred != shifted).mean() == 1.0


def test_predict_with_shift_leaves_a_shift_invariant_model_unchanged():
    _, pred, shifted = predict_with_shift(_GlobalMean(), _loader(), torch.device("cpu"), pixels=3)
    assert np.array_equal(pred, shifted)
