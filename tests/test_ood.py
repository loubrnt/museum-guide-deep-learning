"""Scores that tell a painting from a wall, a face or an object: metrics, sentence margins, collection similarity, the rule."""
import numpy as np
import pytest

from museum_guide.ood import (
    PROMPT_PAIRS,
    NonPaintingRule,
    accepted_share,
    auroc,
    collection_similarity,
    fit_linear_score,
    fpr_at_tpr,
    load_rule,
    max_probability,
    save_rule,
    side_embedding,
    threshold_for_tpr,
    zero_shot_margin,
    zero_shot_probability,
)

# ---- metrics ------------------------------------------------------------------------------------


def test_auroc_known_values():
    assert auroc([0.9, 0.8, 0.4], [0.7, 0.3]) == pytest.approx(5 / 6)
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([0, 1], [2, 3]) == 0.0


def test_auroc_counts_ties_as_one_half():
    assert auroc([1, 1, 1], [1, 1]) == 0.5
    assert auroc([2, 1], [1, 0]) == pytest.approx(0.875)  # 3 wins and 1 tie out of 4 pairs


def test_auroc_of_random_scores_is_near_one_half():
    rng = np.random.default_rng(0)
    assert auroc(rng.random(2000), rng.random(2000)) == pytest.approx(0.5, abs=0.03)


def test_threshold_for_tpr_accepts_at_least_that_share_of_paintings():
    pos = np.arange(1, 101)
    thr = threshold_for_tpr(pos, 0.95)
    assert thr == 5
    assert accepted_share(pos, thr) >= 0.95
    assert accepted_share(pos, thr + 1) < 0.96  # not far above the target: it is the highest such threshold


def test_fpr_at_tpr():
    pos = np.linspace(1, 2, 200)
    assert fpr_at_tpr(pos, np.zeros(50), 0.95)[0] == 0.0  # everything else far below
    assert fpr_at_tpr(pos, np.full(50, 5.0), 0.95)[0] == 1.0  # everything else above
    fpr, thr = fpr_at_tpr(pos, np.linspace(0, 1.5, 100), 0.95)
    assert 0 < fpr < 1 and thr == threshold_for_tpr(pos, 0.95)


def test_max_probability():
    assert max_probability(np.array([[0.2, 0.8], [0.5, 0.5]])).tolist() == [0.8, 0.5]


# ---- sentences ----------------------------------------------------------------------------------


def test_side_embedding_is_a_unit_vector_between_the_sentences():
    side = side_embedding(np.array([[3.0, 0, 0], [0, 4.0, 0]]))  # lengths do not matter
    assert np.linalg.norm(side) == pytest.approx(1.0)
    assert side[0] == pytest.approx(side[1]) and side[2] == 0


def test_zero_shot_margin_sign_and_scale_invariance():
    painting_text, other_text = np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]])
    photos = np.array([[2.0, 0.1, 0], [0.1, 2.0, 0]])
    margin = zero_shot_margin(photos, painting_text, other_text)
    assert margin[0] > 0 > margin[1]
    assert np.allclose(margin, zero_shot_margin(5 * photos, painting_text, other_text))


def test_zero_shot_probability():
    assert zero_shot_probability(np.array([0.0]))[0] == pytest.approx(0.5)
    p = zero_shot_probability(np.array([-0.1, 0.0, 0.1]))
    assert p[0] < p[1] < p[2]


def test_prompt_pairs_are_nonempty_sentences():
    assert len(PROMPT_PAIRS) >= 3
    for painting, other in PROMPT_PAIRS.values():
        assert painting and other and all(isinstance(t, str) and t.strip() for t in painting + other)


# ---- collection ---------------------------------------------------------------------------------


@pytest.fixture
def collection():
    return np.array([[1.0, 0, 0], [0.9, 0.1, 0], [0, 1.0, 0], [0, 0, 1.0]])


def test_collection_similarity_top_k_is_a_mean_of_the_closest(collection):
    photo = np.array([[1.0, 0.0, 0.0]])
    top1 = collection_similarity(photo, collection, top_k=1)
    top2 = collection_similarity(photo, collection, top_k=2)
    assert top1[0] == pytest.approx(1.0)
    assert top2[0] == pytest.approx((1.0 + 0.9 / np.hypot(0.9, 0.1)) / 2)
    assert collection_similarity(photo, collection, top_k=99).shape == (1,)  # k larger than the collection


def test_collection_similarity_can_ignore_the_photo_itself(collection):
    photo = collection[:1]
    assert collection_similarity(photo, collection, 1, ignore_above=0.9999)[0] < 0.999  # next closest, not itself


def test_collection_similarity_can_exclude_a_known_original(collection):
    degraded = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = collection_similarity(degraded, collection, 1, exclude_index=np.array([0, -1]))
    assert out[0] < 1.0  # its original (row 0) is left out
    assert out[1] == pytest.approx(1.0)  # no exclusion for the second one


def test_a_painting_is_closer_to_the_collection_than_a_wall(collection):
    painting, wall = np.array([[0.8, 0.3, 0.0]]), np.array([[0.1, 0.1, -1.0]])
    assert collection_similarity(painting, collection, 2)[0] > collection_similarity(wall, collection, 2)[0]


# ---- the rule -----------------------------------------------------------------------------------


def make_rule(**kw):
    base = dict(painting_prompts=("a painting",), other_prompts=("a photo of a wall",), threshold=0.0)
    return NonPaintingRule(**{**base, **kw})


def test_rule_without_collection_uses_only_the_margin():
    rule = make_rule(w_margin=2.0, bias=0.5)
    painting_text, other_text = np.array([[1.0, 0]]), np.array([[0, 1.0]])
    photos = np.array([[1.0, 0.0], [0.0, 1.0]])
    scores = rule.score(photos, painting_text, other_text)
    assert not rule.needs_collection
    assert scores.tolist() == pytest.approx([2.5, -1.5])
    assert rule.accepts(scores).tolist() == [True, False]


def test_rule_threshold_is_inclusive():
    assert make_rule(threshold=1.0).accepts(np.array([1.0, 0.999])).tolist() == [True, False]


def test_rule_with_collection_needs_it_and_adds_the_similarity(collection):
    rule = make_rule(w_margin=0.0, w_similarity=1.0, top_k=1)
    photos = np.array([[0.0, 1.0, 0.1]])
    texts = (np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))
    assert rule.needs_collection
    with pytest.raises(ValueError, match="collection"):
        rule.score(photos, *texts)
    assert rule.score(photos, *texts, reference=collection)[0] == pytest.approx(collection_similarity(photos, collection, 1, 0.9999)[0])


def test_rule_round_trip(tmp_path):
    rule = make_rule(w_margin=76.5, w_similarity=39.5, bias=-30.0, threshold=0.07, top_k=5, target_tpr=0.95, info={"method": "x"})
    save_rule(rule, tmp_path / "rule.json")
    back = load_rule(tmp_path / "rule.json")
    assert back == rule
    assert isinstance(back.painting_prompts, tuple)


def test_missing_rule_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="non_painting.py"):
        load_rule(tmp_path / "none.json")


def test_fit_linear_score_separates_two_clouds():
    rng = np.random.default_rng(0)
    pos = rng.normal([2.0, 1.0], 0.3, (100, 2))
    neg = rng.normal([-2.0, -1.0], 0.3, (100, 2))
    w, b = fit_linear_score(pos, neg)
    assert (pos @ w + b > 0).mean() > 0.95 and (neg @ w + b < 0).mean() > 0.95
    assert (w > 0).all()
