import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageOps

from museum_guide.audit import (
    cross_split_pairs,
    dhash,
    hamming_matrix,
    metadata_features,
    mirrored_duplicate_pairs,
    named_artist_leaks,
    near_duplicate_pairs,
    neighbour_genre_agreement,
    same_shard_genre_agreement,
    shard_majority_accuracy,
    split_composition,
)


def _picture(seed: int, size: int = 64) -> Image.Image:
    """A smooth random picture (8x8 colours blown up), a stand-in for a painting."""
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 255, (8, 8, 3), dtype=np.uint8))
    return small.resize((size, size), Image.BILINEAR)


def _noisy(img: Image.Image, std: float = 3.0, seed: int = 0) -> Image.Image:
    arr = np.asarray(img, dtype=float) + np.random.default_rng(seed).normal(0, std, (img.height, img.width, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def test_dhash_is_identical_for_identical_images_and_has_hash_size_squared_bits():
    img = _picture(0)
    assert np.array_equal(dhash(img), dhash(img.copy()))
    assert dhash(img).shape == (64,)
    assert dhash(img, hash_size=16).shape == (256,)


def test_dhash_accepts_a_path(tmp_path):
    img = _picture(1)
    path = tmp_path / "a.png"
    img.save(path)
    assert np.array_equal(dhash(path), dhash(img))


def test_dhash_survives_noise_and_rescaling_but_not_a_different_picture():
    img = _picture(0)
    near_noise = int((dhash(img) != dhash(_noisy(img))).sum())
    near_scale = int((dhash(img) != dhash(img.resize((32, 32)))).sum())
    far = [int((dhash(img) != dhash(_picture(s))).sum()) for s in range(1, 8)]
    assert near_noise <= 4 and near_scale <= 4
    assert np.mean(far) > 15  # unrelated pictures differ in a large share of the 64 bits


def test_hamming_matrix_matches_brute_force():
    rng = np.random.default_rng(0)
    h = rng.random((12, 64)) > 0.5
    dist = hamming_matrix(h)
    brute = np.array([[(a != b).sum() for b in h] for a in h])
    assert np.array_equal(dist, brute)
    assert (np.diag(dist) == 0).all()
    other = rng.random((5, 64)) > 0.5
    assert hamming_matrix(h, other).shape == (12, 5)


def test_near_duplicate_pairs_finds_a_planted_copy_and_nothing_else():
    pics = [_picture(s) for s in range(6)]
    pics.append(_noisy(pics[2]))  # image 6 is a noisy copy of image 2
    hashes = np.stack([dhash(p) for p in pics])
    pairs = near_duplicate_pairs(hashes, max_distance=6)
    assert [(i, j) for i, j, _ in pairs] == [(2, 6)]
    assert pairs[0][2] <= 6
    assert near_duplicate_pairs(hashes[:6], max_distance=6) == []


def test_mirrored_duplicates_are_found_only_by_the_mirror_search():
    pics = [_picture(s) for s in range(4)]
    pics.append(ImageOps.mirror(pics[1]))  # image 4 is image 1 flipped left-right
    hashes = np.stack([dhash(p) for p in pics])
    mirrored = np.stack([dhash(ImageOps.mirror(p)) for p in pics])
    assert (1, 4) not in [(i, j) for i, j, _ in near_duplicate_pairs(hashes, 3)]
    assert [(i, j) for i, j, _ in mirrored_duplicate_pairs(hashes, mirrored, 3)] == [(1, 4)]


def test_cross_split_pairs_keeps_only_pairs_across_parts():
    ids = ["a", "b", "c", "d"]
    parts = {"train": ["a", "b"], "test": ["c", "d"]}
    pairs = [(0, 1, 2), (0, 2, 5), (2, 3, 1)]  # within train, across, within test
    assert cross_split_pairs(pairs, ids, parts) == [(0, 2, 5, "train", "test")]
    assert cross_split_pairs([], ids, parts) == []


@pytest.fixture
def parts():
    return {
        "train": pd.DataFrame({"label": ["a", "a", "b", "b", "b"], "artist": ["x", "y", "Unknown Artist", "z", "z"]}),
        "test": pd.DataFrame({"label": ["a", "b"], "artist": ["Unknown Artist", "w"]}),
    }


def test_split_composition(parts):
    comp = split_composition(parts)
    assert comp["train"]["n"] == 5 and comp["test"]["n"] == 2
    assert comp["train"]["unknown_artist_share"] == pytest.approx(0.2)
    assert comp["test"]["unknown_artist_share"] == pytest.approx(0.5)
    assert comp["train"]["class_share"] == {"a": pytest.approx(0.4), "b": pytest.approx(0.6)}
    assert comp["train"]["class_spread"] == pytest.approx(0.2)
    assert comp["test"]["class_spread"] == pytest.approx(0.0)


def test_named_artist_leaks_ignores_unknown_artist(parts):
    assert named_artist_leaks(parts) == set()  # "Unknown Artist" appears in both parts and is not a leak
    parts["test"].loc[1, "artist"] = "z"
    assert named_artist_leaks(parts) == {"z"}


def test_metadata_features_shapes_and_content():
    df = pd.DataFrame(
        {"style": ["Baroque", "Cubism", "Baroque"], "orig_width": [200, 100, 50], "orig_height": [100, 100, 100]}
    )
    assert metadata_features(df, "style").shape == (3, 2)
    shape = metadata_features(df, "format")
    assert shape.shape == (3, 2)
    assert shape[0, 0] == pytest.approx(np.log(2)) and shape[1, 0] == pytest.approx(0)
    assert metadata_features(df, "style+format").shape == (3, 4)
    with pytest.raises(KeyError):
        metadata_features(df, "nope")


def _rows():
    return pd.DataFrame(
        {
            "shard": [1, 1, 1, 1, 2, 2],
            "row": [10, 11, 12, 500, 3, 4],
            "label": ["a", "a", "b", "b", "a", "b"],
            "label_idx": [0, 0, 1, 1, 0, 1],
            "artist": ["Unknown Artist"] * 5 + ["named"],
        }
    )


def test_neighbour_genre_agreement_counts_consecutive_unknown_pairs_per_shard():
    # shard 1 unknown rows 10,11,12,500: pairs (10,11) same genre, (11,12) different, (12,500) same
    assert neighbour_genre_agreement(_rows()) == (3, pytest.approx(2 / 3))
    assert neighbour_genre_agreement(_rows(), max_gap=1) == (2, pytest.approx(0.5))  # drops the 12 -> 500 pair
    pairs, share = neighbour_genre_agreement(_rows().iloc[:1])
    assert pairs == 0 and np.isnan(share)


def test_same_shard_genre_agreement_is_the_pair_baseline():
    # shard 1: labels a,a,b,b -> same-genre pairs 2 of 6; shard 2 has one unknown image: no pair
    assert same_shard_genre_agreement(_rows()) == pytest.approx(2 / 6)


def test_shard_majority_accuracy_falls_back_for_unseen_shards():
    df = _rows()
    train = df[df["row"].isin([10, 11, 500])]  # shard 1 majority: a
    test = pd.DataFrame({"shard": [1, 1, 9], "label_idx": [0, 1, 0]})  # shard 9 unseen -> global mode: a
    assert shard_majority_accuracy(train, test) == pytest.approx(2 / 3)
