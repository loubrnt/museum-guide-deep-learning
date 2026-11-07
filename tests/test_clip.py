import numpy as np
import pytest

from museum_guide.clip import (
    PROMPT_TEMPLATES,
    cache_key,
    cached_embeddings,
    class_prompts,
    fit_linear_probe,
    normalise,
    zero_shot_predict,
)


def test_class_prompts_replace_underscores():
    assert class_prompts(["still_life", "portrait"], PROMPT_TEMPLATES["painting"]) == [
        "a painting of a still life",
        "a painting of a portrait",
    ]
    assert class_prompts(["portrait"], PROMPT_TEMPLATES["simple"]) == ["a portrait"]


def test_normalise_gives_unit_rows_and_survives_zero_vectors():
    x = normalise(np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]]))
    assert np.allclose(np.linalg.norm(x[[0, 2]], axis=1), 1.0)
    assert np.isfinite(x).all() and (x[1] == 0).all()
    assert x.dtype == np.float32


def test_zero_shot_picks_the_closest_text_vector():
    text = normalise(np.eye(3))
    images = normalise(np.array([[0.9, 0.1, 0.0], [0.0, 0.2, 0.8], [0.1, 0.9, 0.1]]))
    assert zero_shot_predict(images, text).tolist() == [0, 2, 1]


def _blobs(n, noise, seed):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    centres = np.eye(3) * 3
    return normalise(centres[y] + rng.normal(scale=noise, size=(n, 3))), y


def test_linear_probe_learns_separable_embeddings():
    Xtr, ytr = _blobs(90, 0.3, 0)
    Xva, yva = _blobs(30, 0.3, 1)
    Xte, yte = _blobs(30, 0.3, 2)
    clf, C = fit_linear_probe(Xtr, ytr, Xva, yva)
    assert C in (0.1, 1, 10, 100)
    assert clf.score(Xte, yte) > 0.9


def test_linear_probe_chooses_the_C_with_the_best_validation_score():
    from sklearn.linear_model import LogisticRegression

    Xtr, ytr = _blobs(60, 1.5, 0)  # noisy: different C give different validation scores
    Xva, yva = _blobs(60, 1.5, 1)
    Cs = (0.001, 0.1, 100)
    scores = {C: LogisticRegression(C=C, max_iter=3000).fit(Xtr, ytr).score(Xva, yva) for C in Cs}
    assert len(set(scores.values())) > 1  # the test is only meaningful if the choice matters
    expected = min(C for C in Cs if scores[C] == max(scores.values()))
    clf, C = fit_linear_probe(Xtr, ytr, Xva, yva, Cs=Cs)
    assert C == expected
    assert clf.C == expected


def test_linear_probe_ties_go_to_the_smallest_C():
    Xtr, ytr = _blobs(60, 0.05, 0)  # trivially separable: every C reaches 100% on validation
    Xva, yva = _blobs(30, 0.05, 1)
    _, C = fit_linear_probe(Xtr, ytr, Xva, yva, Cs=(100, 1, 10))
    assert C == 1


def test_cache_key_depends_on_model_and_ordered_paths():
    a = cache_key("m", ["x.jpg", "y.jpg"])
    assert a == cache_key("m", ["x.jpg", "y.jpg"])
    assert a != cache_key("m2", ["x.jpg", "y.jpg"])
    assert a != cache_key("m", ["y.jpg", "x.jpg"])
    assert a != cache_key("m", ["x.jpg", "y.jpg", "z.jpg"])


def test_cache_round_trip_and_invalidation(tmp_path):
    cache = tmp_path / "sub" / "emb.npz"
    calls = []

    def compute():
        calls.append(1)
        return np.arange(6, dtype=np.float32).reshape(3, 2)

    paths = ["a", "b", "c"]
    first = cached_embeddings(cache, "m", paths, compute)
    second = cached_embeddings(cache, "m", paths, compute)
    assert len(calls) == 1  # second call came from the file
    assert np.array_equal(first, second)
    cached_embeddings(cache, "m", paths[:2], compute)  # different images: recompute
    assert len(calls) == 2
    cached_embeddings(cache, "other-model", paths[:2], compute)
    assert len(calls) == 3


def test_cache_ignores_a_stale_file_of_another_dataset(tmp_path):
    cache = tmp_path / "emb.npz"
    cached_embeddings(cache, "m", ["a"], lambda: np.ones((1, 2), dtype=np.float32))
    out = cached_embeddings(cache, "m", ["b"], lambda: np.zeros((1, 2), dtype=np.float32))
    assert (out == 0).all()
    assert pytest.approx(0) == float(out.sum())
