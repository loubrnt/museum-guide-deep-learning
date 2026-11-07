import numpy as np
import pandas as pd
import pytest

from museum_guide.curves import few_shot_indices, images_needed, subsample_train


@pytest.fixture
def train():
    """300 images, 3 classes; class 0 has 5 artists of 20 images, the others 100 single-image 'unknown' groups."""
    rows = []
    for i in range(100):
        rows.append({"label_idx": 0, "group": f"artist_{i // 20}"})
    for c in (1, 2):
        for i in range(100):
            rows.append({"label_idx": c, "group": f"unknown_{c}_{i}"})
    return pd.DataFrame(rows)


def test_subsample_has_the_requested_size_and_every_class(train):
    idx = subsample_train(train, 30, seed=0)
    assert len(idx) == 30
    counts = train.iloc[idx]["label_idx"].value_counts()
    assert set(counts.index) == {0, 1, 2}
    assert (counts == 10).all()


def test_subsample_is_deterministic_and_depends_on_the_seed(train):
    a, b, c = subsample_train(train, 30, 0), subsample_train(train, 30, 0), subsample_train(train, 30, 1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_subsample_keeps_artists_together_except_for_the_last_one(train):
    idx = subsample_train(train, 30, seed=3)  # class 0 quota 10: one artist of 20 images, trimmed to 10
    groups = train.iloc[idx].query("label_idx == 0")["group"]
    assert groups.nunique() == 1
    idx = subsample_train(train, 90, seed=3)  # quota 30: one whole artist (20) + 10 from a second one
    sizes = train.iloc[idx].query("label_idx == 0").groupby("group").size().sort_values()
    assert list(sizes) == [10, 20]


def test_subsample_small_n_still_represents_every_class(train):
    for n in (1, 2, 3):  # n=1 or 2 rounds the quota of each class down to 0: the minimum of one image kicks in
        for seed in range(20):
            idx = subsample_train(train, n, seed)
            assert set(train.iloc[idx]["label_idx"]) == {0, 1, 2}


def test_subsample_takes_everything_when_n_is_large(train):
    assert len(subsample_train(train, 10_000, seed=0)) == len(train)


def test_subsample_class_with_fewer_images_than_its_quota():
    df = pd.DataFrame({"label_idx": [0] * 50 + [1] * 3, "group": [f"g{i}" for i in range(53)]})
    idx = subsample_train(df, 40, seed=0)  # quota 20 per class, class 1 only has 3
    counts = df.iloc[idx]["label_idx"].value_counts()
    assert counts[0] == 20 and counts[1] == 3


def test_few_shot_draws_k_per_class_without_replacement():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 2)
    idx = few_shot_indices(labels, 3, seed=1)
    assert len(idx) == len(set(idx)) == 3 + 3 + 2  # class 2 only has 2
    assert np.array_equal(np.bincount(labels[idx]), [3, 3, 2])
    assert not np.array_equal(idx, few_shot_indices(labels, 3, seed=2))
    assert np.array_equal(idx, few_shot_indices(labels, 3, seed=1))


def test_images_needed_interpolates_linearly():
    r = images_needed([100, 200, 400], [0.6, 0.8, 0.9], 0.7)
    assert r == {"reached": True, "images": pytest.approx(150.0), "at_most": False}
    r = images_needed([100, 200, 400], [0.6, 0.8, 0.9], 0.85)
    assert r["images"] == pytest.approx(300.0)


def test_images_needed_edge_cases():
    assert images_needed([100, 200], [0.9, 0.95], 0.8) == {"reached": True, "images": 100.0, "at_most": True}
    assert images_needed([100, 200], [0.5, 0.6], 0.8) == {"reached": False, "images": None, "at_most": False}
    # unsorted input is sorted by size
    assert images_needed([400, 100, 200], [0.9, 0.6, 0.8], 0.7)["images"] == pytest.approx(150.0)
    # the target is hit exactly at a measured point
    assert images_needed([100, 200], [0.5, 0.8], 0.8)["images"] == pytest.approx(200.0)
    with pytest.raises(ValueError):
        images_needed([], [], 0.5)
