import numpy as np
import pytest

from museum_guide.metrics import accuracy_by_group, least_confident_correct, most_confident_errors, per_class_report

CLASSES = ["a", "b", "c"]


def test_per_class_report_perfect_predictions():
    y = np.array([0, 1, 2, 0, 1, 2])
    rep = per_class_report(y, y, CLASSES)
    assert rep["accuracy"] == 1.0 and rep["macro_f1"] == 1.0
    assert all(v["f1"] == 1.0 and v["support"] == 2 for v in rep["per_class"].values())


def test_per_class_report_known_values():
    y = np.array([0, 0, 0, 0, 1, 1, 2, 2])
    pred = np.array([0, 0, 0, 1, 1, 0, 2, 2])
    rep = per_class_report(y, pred, CLASSES)["per_class"]
    assert rep["a"]["precision"] == pytest.approx(3 / 4)  # 3 of the 4 predicted "a" are right
    assert rep["a"]["recall"] == pytest.approx(3 / 4)  # 3 of the 4 true "a" are found
    assert rep["b"]["precision"] == pytest.approx(1 / 2)
    assert rep["b"]["recall"] == pytest.approx(1 / 2)
    assert rep["c"]["f1"] == 1.0
    assert rep["a"]["support"] == 4


def test_per_class_report_class_never_predicted_or_absent():
    y = np.array([0, 0, 1, 1])
    pred = np.array([0, 0, 0, 0])  # "b" never predicted, "c" absent from y
    rep = per_class_report(y, pred, CLASSES)
    assert rep["per_class"]["b"]["precision"] == 0.0 and rep["per_class"]["b"]["recall"] == 0.0
    assert rep["per_class"]["c"]["support"] == 0
    assert rep["macro_f1"] == pytest.approx(rep["per_class"]["a"]["f1"] / 3)  # 2/3 f1 for "a", 0 elsewhere


def test_accuracy_by_group_and_min_count():
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 0])
    groups = np.array(["x", "x", "y", "y", "y"])
    out = accuracy_by_group(y, pred, groups)
    assert out["x"] == {"n": 2, "acc": 0.5}
    assert out["y"]["acc"] == pytest.approx(2 / 3)
    assert "x" not in accuracy_by_group(y, pred, groups, min_count=3)


def _probs():
    # sample: (true, probabilities)
    return (
        np.array([0, 1, 2, 0, 1]),
        np.array(
            [
                [0.9, 0.05, 0.05],  # right, confident
                [0.6, 0.3, 0.1],  # wrong (pred 0), conf 0.6
                [0.4, 0.1, 0.5],  # right, conf 0.5
                [0.1, 0.05, 0.85],  # wrong (pred 2), conf 0.85
                [0.2, 0.45, 0.35],  # right, conf 0.45
            ]
        ),
    )


def test_most_confident_errors_are_wrong_and_sorted():
    y, probs = _probs()
    idx = most_confident_errors(probs, y, k=5)
    assert idx.tolist() == [3, 1]  # only the two wrong ones, most confident first
    assert most_confident_errors(probs, y, k=1).tolist() == [3]


def test_least_confident_correct_are_right_and_sorted():
    y, probs = _probs()
    assert least_confident_correct(probs, y, k=5).tolist() == [4, 2, 0]
    assert least_confident_correct(probs, y, k=2).tolist() == [4, 2]


def test_no_errors_gives_empty_selection():
    y = np.array([0, 1])
    probs = np.array([[0.9, 0.1], [0.2, 0.8]])
    assert most_confident_errors(probs, y).size == 0
