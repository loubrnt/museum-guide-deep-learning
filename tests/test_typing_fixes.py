"""Behaviour at the edges that the typing clean-up made explicit (mypy flagged each of these sites)."""
import numpy as np
import pytest

from museum_guide.clip import fit_linear_probe
from museum_guide.demo_logic import DemoResult, display
from museum_guide.notices import normalise_label, story_entry
from museum_guide.ood import NonPaintingRule
from museum_guide.search import average_precision, ndcg_at_k, precision_at_k


def test_linear_probe_without_any_C_is_a_clear_error():
    # before: "TypeError: 'NoneType' object is not subscriptable", from an empty candidate list
    X, y = np.eye(4), np.array([0, 1, 0, 1])
    with pytest.raises(ValueError, match="at least one regularisation value"):
        fit_linear_probe(X, y, X, y, Cs=())


def test_linear_probe_with_a_single_C_returns_it():
    X, y = np.eye(4), np.array([0, 1, 0, 1])
    clf, C = fit_linear_probe(X, y, X, y, Cs=(5.0,))
    assert C == 5.0 and hasattr(clf, "predict")


@pytest.mark.parametrize("status", ["painting", "unsure"])
def test_display_of_a_guess_without_any_genre_is_a_clear_error(tmp_path, status):
    # before: "TypeError: cannot unpack non-iterable NoneType object"
    with pytest.raises(ValueError, match="at least one genre"):
        display(DemoResult(status, genre=[]), tmp_path, "en")


def test_story_entry_uses_the_canonical_label_and_the_title():
    entry = story_entry("Post_Impressionism", "en")
    assert entry["label"] == normalise_label("Post_Impressionism")
    assert set(entry) == {"label", "lang", "title", "text"} and entry["lang"] == "en" and entry["title"] and entry["text"]
    assert story_entry("not an art movement", "en") is None


def test_the_non_painting_rule_takes_a_scalar_or_an_array():
    rule = NonPaintingRule(("a painting",), ("a wall",), threshold=0.5)
    assert bool(rule.accepts(0.7)) is True and bool(rule.accepts(0.2)) is False
    assert rule.accepts(np.array([0.1, 0.5, 0.9])).tolist() == [False, True, True]


def test_the_search_metrics_accept_numpy_arrays():
    relevant = np.array([1, 0, 1, 0, 0])
    assert precision_at_k(relevant, 2) == precision_at_k([1, 0, 1, 0, 0], 2) == 0.5
    assert average_precision(relevant) == average_precision([1, 0, 1, 0, 0])
    assert ndcg_at_k(relevant, 5) == ndcg_at_k([1, 0, 1, 0, 0], 5)
