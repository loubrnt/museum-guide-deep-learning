import json

import numpy as np
import pytest

from museum_guide.calibration import (
    apply_temperature,
    confidence_threshold_for_accuracy,
    expected_calibration_error,
    fit_temperature,
    load_temperature,
    nll,
    reliability_bins,
    selective_accuracy,
    softmax,
)


def _sample(n, scale, true_temperature, seed=0):
    """Logits z and labels drawn from softmax(z / true_temperature): the model is calibrated at T = true_temperature."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, scale, (n, 5))
    p = softmax(z, true_temperature)
    labels = np.array([rng.choice(5, p=row) for row in p])
    return z, labels


def test_softmax_rows_sum_to_one_and_temperature_flattens():
    z = np.array([[2.0, 0.0, -1.0], [1000.0, 0.0, 0.0]])  # the second row would overflow a naive softmax
    p = softmax(z)
    assert np.allclose(p.sum(1), 1) and np.isfinite(p).all()
    assert softmax(z, 10.0)[0].max() < p[0].max()
    assert np.allclose(softmax(z, 1e6), 1 / 3, atol=1e-2)


def test_temperature_never_changes_the_predicted_class():
    z = np.random.default_rng(1).normal(0, 3, (200, 5))
    assert np.array_equal(apply_temperature(z, 0.3).argmax(1), apply_temperature(z, 7.0).argmax(1))


def test_fit_temperature_recovers_a_known_temperature():
    z, y = _sample(20000, scale=3.0, true_temperature=2.0)  # over-confident logits: the right T is 2
    assert fit_temperature(z, y) == pytest.approx(2.0, rel=0.1)


def test_fit_temperature_is_close_to_one_for_calibrated_logits_and_is_deterministic():
    z, y = _sample(20000, scale=2.0, true_temperature=1.0, seed=3)
    t = fit_temperature(z, y)
    assert t == pytest.approx(1.0, rel=0.1)
    assert fit_temperature(z, y) == t


def test_fit_temperature_lowers_the_nll():
    z, y = _sample(3000, scale=3.0, true_temperature=2.5, seed=4)
    assert nll(z, y, fit_temperature(z, y)) < nll(z, y, 1.0)


def test_ece_on_a_hand_computed_example():
    probs = np.array([[0.9, 0.1], [0.9, 0.1]])
    assert expected_calibration_error(probs, np.array([0, 1])) == pytest.approx(0.4)  # confidence 0.9, accuracy 0.5


def test_ece_is_near_zero_when_calibrated_and_large_when_overconfident():
    z, y = _sample(20000, scale=3.0, true_temperature=2.0, seed=5)
    assert expected_calibration_error(softmax(z, 2.0), y) < 0.03
    assert expected_calibration_error(softmax(z, 1.0), y) > 0.08


def test_reliability_bins_count_every_prediction():
    z, y = _sample(500, scale=2.0, true_temperature=1.0, seed=6)
    bins = reliability_bins(softmax(z), y)
    assert sum(b["n"] for b in bins) == 500
    assert all((b["accuracy"] is None) == (b["n"] == 0) for b in bins)


def test_selective_accuracy_rises_as_coverage_falls():
    rng = np.random.default_rng(0)
    confidence = rng.random(2000)
    correct = rng.random(2000) < confidence  # more confident, more often right
    rows = selective_accuracy(confidence, correct, [1.0, 0.8, 0.5, 0.2])
    accs = [r["accuracy"] for r in rows]
    assert accs == sorted(accs)
    assert rows[0]["accuracy"] == pytest.approx(correct.mean())
    assert rows[0]["realised_coverage"] == 1.0
    assert rows[2]["realised_coverage"] == pytest.approx(0.5, abs=0.01)


def test_selective_accuracy_is_flat_when_confidence_is_uninformative():
    rng = np.random.default_rng(1)
    confidence, correct = rng.random(5000), rng.random(5000) < 0.7
    rows = selective_accuracy(confidence, correct, [1.0, 0.5, 0.25])
    assert all(abs(r["accuracy"] - 0.7) < 0.05 for r in rows)


def test_confidence_threshold_for_accuracy():
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    correct = np.array([1, 1, 1, 0, 0])
    assert confidence_threshold_for_accuracy(confidence, correct, 1.0) == 0.7
    assert confidence_threshold_for_accuracy(confidence, correct, 0.6) == 0.5
    assert confidence_threshold_for_accuracy(confidence, np.zeros(5), 0.9) is None


def test_load_temperature(tmp_path):
    assert load_temperature(tmp_path / "missing.json") == 1.0
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"temperature": 2.5, "fitted_on": "validation set"}))
    assert load_temperature(path) == 2.5
