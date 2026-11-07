import math

import numpy as np
import pandas as pd
import pytest

from museum_guide.clip import normalise
from museum_guide.search import (
    average_precision,
    artist_name,
    build_query_set,
    load_index,
    make_index,
    ndcg_at_k,
    precision_at_k,
    random_baseline,
    rank_all,
    relevance_in_rank_order,
    save_index,
    scores,
    search,
)


def _df(n=6):
    return pd.DataFrame(
        {
            "id": [f"i{i}" for i in range(n)],
            "path": [f"portrait/i{i}.jpg" for i in range(n)],
            "label": ["portrait"] * n,
            "style": ["Baroque"] * n,
            "artist": [f"artist{i}" for i in range(n)],
            "extra": range(n),
        }
    )


def test_precision_at_k():
    assert precision_at_k([1, 0, 1, 0, 0], k=2) == 0.5
    assert precision_at_k([1, 1, 1], k=3) == 1.0
    assert precision_at_k([0, 0, 1], k=2) == 0.0


def test_average_precision_known_values():
    assert average_precision([1, 1, 0, 0]) == 1.0
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2
    assert average_precision([1, 0, 1, 0]) == pytest.approx((1 + 2 / 3) / 2)
    # a single relevant item at rank 4
    assert average_precision([0, 0, 0, 1]) == pytest.approx(0.25)
    assert average_precision([0, 0, 0]) == 0.0


def test_ndcg_known_values():
    assert ndcg_at_k([1, 1, 0, 0], k=2) == pytest.approx(1.0)
    assert ndcg_at_k([0, 1, 0], k=3) == pytest.approx((1 / math.log2(3)) / 1.0)  # one relevant item, ideal = 1
    assert ndcg_at_k([1, 0, 1], k=3) == pytest.approx((1 + 1 / 2) / (1 + 1 / math.log2(3)))
    assert ndcg_at_k([0, 0, 0], k=3) == 0.0


def test_ndcg_is_capped_by_k_when_more_items_are_relevant_than_k():
    assert ndcg_at_k([1, 1, 1, 1], k=2) == pytest.approx(1.0)


def test_random_baseline_matches_the_base_rate():
    base = random_baseline(n_relevant=150, n_items=1500, k=10, n_draws=400)
    assert base["precision@10"] == pytest.approx(0.10, abs=0.03)
    assert base["average_precision"] == pytest.approx(0.10, abs=0.03)
    assert random_baseline(0, 50)["average_precision"] == 0.0


def test_search_orders_by_cosine_similarity_and_respects_k():
    vecs = normalise(np.array([[1, 0, 0], [0.6, 0.8, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32))
    index = make_index(vecs, _df(4))
    query = normalise(np.array([[0.0, 1.0, 0.0]]))
    top = search(index, query, k=3)
    assert top["id"].tolist() == ["i2", "i1", "i0"]
    assert top["rank"].tolist() == [1, 2, 3]
    assert top["score"].is_monotonic_decreasing
    assert len(search(index, query, k=99)) == 4
    assert rank_all(index, query).tolist() == [2, 1, 0, 3]


def test_scores_accept_a_vector_or_a_one_row_matrix():
    index = make_index(np.eye(3, dtype=np.float32), _df(3))
    assert np.allclose(scores(index, np.array([0.0, 1.0, 0.0])), scores(index, np.array([[0.0, 1.0, 0.0]])))


def test_make_index_normalises_keeps_only_metadata_columns_and_checks_lengths():
    index = make_index(np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32), _df(2))
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1.0)
    assert list(index.meta.columns) == ["id", "path", "label", "style", "artist"]
    with pytest.raises(ValueError):
        make_index(np.ones((3, 2), dtype=np.float32), _df(2))


def test_index_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    index = make_index(rng.normal(size=(6, 8)).astype(np.float32), _df(6), model="some/model")
    save_index(index, tmp_path / "sub" / "index.npz")
    loaded = load_index(tmp_path / "sub" / "index.npz")
    assert np.array_equal(loaded.vectors, index.vectors)
    assert loaded.meta.equals(index.meta)
    assert loaded.model == "some/model"
    assert len(loaded) == 6


def _collection():
    rows = []
    for i in range(40):
        rows.append(("Cubism" if i < 12 else "Baroque", "still_life" if i < 20 else "portrait", "georges-braque" if i < 11 else "Unknown Artist"))
    df = pd.DataFrame(rows, columns=["style", "label", "artist"])
    df["id"] = [f"i{i}" for i in range(len(df))]
    df["path"] = df["id"] + ".jpg"
    return df


def test_query_set_relevance_comes_from_the_metadata():
    df = _collection()
    queries = {q["id"]: q for q in build_query_set(df, min_relevant=10)}
    cubist = queries["style_genre:Cubist:still_life"]
    assert cubist["text"] == "a Cubist still life"
    assert cubist["relevant"] == list(range(12))
    assert queries["artist:georges-braque"]["text"] == "paintings by Georges Braque"
    assert queries["artist:georges-braque"]["relevant"] == list(range(11))
    assert queries["genre:portrait"]["relevant"] == list(range(20, 40))
    assert queries["style_genre:Baroque:portrait"]["relevant"] == list(range(20, 40))


def test_query_set_drops_combinations_with_too_few_relevant_paintings():
    queries = {q["id"] for q in build_query_set(_collection(), min_relevant=10)}
    assert "style_genre:Baroque:still_life" not in queries
    assert "style_genre:Cubist:portrait" not in queries
    assert not any(q.startswith("artist:Unknown") for q in queries)


def test_query_set_merges_the_cubism_labels_and_uses_the_right_article():
    df = pd.DataFrame(
        {
            "style": ["Cubism"] * 6 + ["Synthetic_Cubism"] * 6 + ["Impressionism"] * 10 + ["Ukiyo_e"] * 10,
            "label": ["still_life"] * 12 + ["landscape"] * 10 + ["portrait"] * 10,
            "artist": ["Unknown Artist"] * 32,
        }
    )
    q = {x["id"]: x for x in build_query_set(df, min_relevant=10)}
    assert q["style_genre:Cubist:still_life"]["relevant"] == list(range(12))  # both Cubism labels together
    assert q["style_genre:Impressionist:landscape"]["text"] == "an Impressionist landscape"
    assert q["style_genre:Ukiyo-e:portrait"]["text"] == "a Ukiyo-e portrait"


def test_artist_name():
    assert artist_name("claude-monet") == "Claude Monet"
    assert artist_name("bartolome-esteban-murillo") == "Bartolome Esteban Murillo"


def test_relevance_in_rank_order():
    assert relevance_in_rank_order(np.array([2, 0, 1, 3]), [0, 3], 4).tolist() == [0, 1, 0, 1]
