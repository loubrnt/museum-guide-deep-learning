import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from museum_guide.catalogue import Catalogue
from museum_guide.search import hybrid_search, make_index, rank_with_filter, search

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def collection():
    """Six paintings. Vectors are 3-d: axis 0 = 'angel', axis 1 = 'boat', axis 2 = anything else."""
    df = pd.DataFrame(
        {
            "id": [f"p{i}" for i in range(6)],
            "path": [f"x/p{i}.jpg" for i in range(6)],
            "label": ["religious_painting", "religious_painting", "religious_painting", "landscape", "landscape", "portrait"],
            "style": ["Baroque", "Baroque", "Impressionism", "Baroque", "Impressionism", "Baroque"],
            "artist": ["Unknown Artist", "Unknown Artist", "claude-monet", "Unknown Artist", "claude-monet", "Unknown Artist"],
        }
    )
    vectors = np.array(
        [
            [0.2, 0.0, 1.0],  # Baroque religious, weakly 'angel'
            [0.6, 0.0, 0.8],  # Baroque religious, more 'angel'
            [1.0, 0.0, 0.0],  # Impressionist religious (not Baroque), the most 'angel'
            [0.0, 1.0, 0.0],  # Baroque landscape, the most 'boat'
            [0.0, 0.7, 0.7],  # Impressionist landscape by Monet
            [0.9, 0.0, 0.4],  # Baroque portrait, very 'angel'
        ]
    )
    return make_index(vectors, df), Catalogue.from_metadata(df)


class Embedder:
    """Text -> vector, recording what it was asked to embed."""

    def __init__(self):
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        v = np.zeros(3)
        if "angel" in text:
            v[0] = 1.0
        if "boat" in text:
            v[1] = 1.0
        if not v.any():
            v[2] = 1.0
        return v / np.linalg.norm(v)


def test_without_catalogue_fields_it_is_plain_clip_search(collection):
    index, cat = collection
    embed = Embedder()
    results, parsed = hybrid_search(index, "boat", embed, cat, k=3)
    plain = search(index, Embedder()("boat"), 3)
    assert not parsed.has_filters
    assert results["path"].tolist() == plain["path"].tolist()
    assert not results["catalogue_match"].any()


def test_a_hard_filter_returns_only_the_paintings_that_fit_ranked_by_clip(collection):
    index, cat = collection
    results, parsed = hybrid_search(index, "a Baroque religious painting with angels", Embedder(), cat, k=5)
    assert parsed.styles == ("Baroque",) and parsed.genres == ("religious_painting",)
    assert results["path"].tolist() == ["x/p1.jpg", "x/p0.jpg"]  # p2 is more 'angel' but not Baroque
    assert results["catalogue_match"].all()
    assert results["rank"].tolist() == [1, 2]


def test_clip_receives_the_remaining_words_not_the_whole_question(collection):
    index, cat = collection
    embed = Embedder()
    hybrid_search(index, "a Baroque religious painting with angels", embed, cat, k=3)
    assert embed.calls == ["angels"]


def test_clip_receives_the_whole_question_when_nothing_else_is_left(collection):
    index, cat = collection
    embed = Embedder()
    hybrid_search(index, "a Baroque landscape", embed, cat, k=3)
    assert embed.calls == ["a Baroque landscape"]


def test_fewer_than_k_are_returned_when_the_catalogue_has_fewer(collection):
    index, cat = collection
    results, _ = hybrid_search(index, "paintings by Claude Monet", Embedder(), cat, k=5)
    assert sorted(results["path"]) == ["x/p2.jpg", "x/p4.jpg"]


def test_when_nothing_fits_the_fields_it_falls_back_to_plain_search(collection):
    index, cat = collection
    # Impressionist portraits do not exist in this collection
    results, parsed = hybrid_search(index, "an Impressionist portrait of an angel", Embedder(), cat, k=3)
    assert parsed.has_filters
    assert not results["catalogue_match"].any()
    plain = search(index, Embedder()("an Impressionist portrait of an angel"), 3)
    assert results["path"].tolist() == plain["path"].tolist()


def test_a_soft_boost_keeps_good_clip_matches_that_do_not_fit(collection):
    index, cat = collection
    weak, _ = hybrid_search(index, "a Baroque religious painting with angels", Embedder(), cat, k=3, boost=0.05)
    strong, _ = hybrid_search(index, "a Baroque religious painting with angels", Embedder(), cat, k=3, boost=5.0)
    # a small boost does not outweigh CLIP: the two most 'angel' paintings come first although neither fits both fields
    assert weak["path"].tolist()[:2] == ["x/p2.jpg", "x/p5.jpg"]
    assert not weak["catalogue_match"].all()
    assert strong["path"].tolist()[:2] == ["x/p1.jpg", "x/p0.jpg"]
    assert strong["catalogue_match"].iloc[:2].all()
    assert len(weak) == len(strong) == 3


def test_rank_with_filter_modes():
    df = pd.DataFrame({"id": list("abcd"), "path": list("abcd"), "label": ["x"] * 4, "style": ["y"] * 4, "artist": ["z"] * 4})
    index = make_index(np.eye(4), df)
    text = np.array([4.0, 3.0, 2.0, 1.0])  # cosine order: a, b, c, d (before normalisation by the index)
    match = np.array([False, False, True, True])
    assert rank_with_filter(index, text, match).tolist() == [2, 3, 0, 1]  # fits first, CLIP order inside each group
    assert rank_with_filter(index, text, match, boost=0.0).tolist() == [0, 1, 2, 3]  # no boost: plain CLIP
    assert rank_with_filter(index, text, match, boost=100.0).tolist() == [2, 3, 0, 1]
    assert rank_with_filter(index, text, np.ones(4, dtype=bool)).tolist() == [0, 1, 2, 3]


def test_results_keep_the_columns_of_plain_search(collection):
    index, cat = collection
    plain = search(index, Embedder()("angels"), 2)
    hybrid, _ = hybrid_search(index, "a Baroque religious painting with angels", Embedder(), cat, k=2)
    assert list(hybrid.columns) == [*plain.columns, "catalogue_match"]


def _run(*args):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "search.py"), *args],
        capture_output=True, text=True, env={"PYTHONPATH": str(ROOT / "src"), "PATH": ""},
    )


def test_cli_lists_the_new_options_and_keeps_the_old_ones():
    out = _run("--help")
    assert out.returncode == 0, out.stderr
    for flag in ("--hybrid", "--boost", "--encoder", "--story", "--grid", "--rebuild"):
        assert flag in out.stdout


def test_cli_rejects_boost_without_hybrid():
    out = _run("a stormy sea", "--boost", "0.1")
    assert out.returncode == 2
    assert "--boost only applies together with --hybrid" in out.stderr


# ---------------------------------------------------------------------------------------- the committed judgments
def test_manual_judgments_match_the_manual_questions():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import search_hybrid_eval as ev
    finally:
        sys.path.pop(0)
    saved = json.loads((ROOT / "reports" / "search_hybrid_manual.json").read_text())
    ids = [qid for qid, _, _ in ev.MANUAL_QUERIES]
    assert list(saved["queries"]) == ids
    for qid, entry in saved["queries"].items():
        for method in ("plain", "hybrid"):
            assert len(entry[method]["paths"]) == len(entry[method]["visual"]) <= 5, qid
            assert set(entry[method]["visual"]) <= {0, 1}, qid
    assert "human should confirm" in saved["reviewer"]
    assert "images" in saved["protocol"] and "never images" in saved["protocol"]


def test_expected_fields_and_parser_report_on_a_tiny_collection(collection):
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import search_hybrid_eval as ev
    finally:
        sys.path.pop(0)
    _, cat = collection
    queries = [
        {"id": "a", "type": "artist", "text": "paintings by Claude Monet", "relevance": {"artist": "claude-monet"}},
        {"id": "b", "type": "style_genre", "text": "a Baroque landscape", "relevance": {"styles": ["Baroque"], "genre": "landscape"}},
        {"id": "c", "type": "genre", "text": "boats on a lake", "relevance": {"genre": "landscape"}},  # not named: a miss
    ]
    assert ev.expected_fields(queries[1]) == {"artist": set(), "style": {"Baroque"}, "genre": {"landscape"}}
    report = ev.parser_report(cat, queries)
    assert report["exact_match_rate"] == pytest.approx(2 / 3)
    assert report["per_field"]["artist"] == {"precision": 1.0, "recall": 1.0, "true_positive": 1, "false_positive": 0, "false_negative": 0}
    assert report["per_field"]["genre"]["recall"] == pytest.approx(0.5)
    assert report["per_field"]["genre"]["precision"] == 1.0
    assert [f["id"] for f in report["failures"]] == ["c"]
