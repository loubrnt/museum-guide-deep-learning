import numpy as np
import pandas as pd
import pytest

from museum_guide.style import metadata_floor_features, pooled_confusion, row_normalise, top_confusions


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "genre": ["landscape", "portrait", "landscape", "Unknown Genre"],
            "shard": [3, 3, 7, 9],
            "orig_width": [200, 100, 400, 50],
            "orig_height": [100, 100, 100, 100],
        }
    )


def test_floor_features_shapes_and_content(table):
    assert metadata_floor_features(table, "genre").shape == (4, 3)  # 3 distinct genres
    assert metadata_floor_features(table, "shard").shape == (4, 3)  # 3 distinct shards
    fmt = metadata_floor_features(table, "format")
    assert fmt.shape == (4, 2)
    assert fmt[0, 0] == pytest.approx(np.log(2.0))  # aspect ratio 200/100
    assert fmt[1, 0] == pytest.approx(0.0)  # square
    assert fmt[1, 1] == pytest.approx(np.log(100 * 100))
    assert metadata_floor_features(table, "genre+format").shape == (4, 5)


def test_floor_features_are_one_hot(table):
    onehot = metadata_floor_features(table, "genre")
    assert (onehot.sum(axis=1) == 1).all()
    assert onehot[0].tolist() == onehot[2].tolist()  # same genre, same row


def test_floor_features_reject_unknown_names(table):
    with pytest.raises(ValueError, match="artist"):
        metadata_floor_features(table, "genre+artist")


def test_pooled_confusion_sums_over_seeds():
    a = (np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]))
    b = (np.array([0, 1, 1, 2]), np.array([0, 1, 0, 2]))
    cm = pooled_confusion([a, b], 3)
    assert cm.tolist() == [[2, 1, 0], [1, 2, 0], [0, 0, 2]]
    assert cm.sum() == 8


def test_row_normalise_handles_empty_rows():
    out = row_normalise(np.array([[1, 3], [0, 0]]))
    assert out.tolist() == [[0.25, 0.75], [0.0, 0.0]]


def test_top_confusions_ranks_off_diagonal_cells():
    cm = np.array([[10, 4, 0], [1, 8, 3], [0, 0, 5]])
    top = top_confusions(cm, ["a", "b", "c"], k=2)
    assert [(t["true"], t["predicted"], t["count"]) for t in top] == [("a", "b", 4), ("b", "c", 3)]
    assert top[0]["share_of_true_class"] == pytest.approx(4 / 14)
    assert len(top_confusions(cm, ["a", "b", "c"], k=10)) == 3  # only 3 non-zero mistakes
