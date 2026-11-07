"""Sanity checks on the data: near-duplicate images, split composition, and metadata-only "floor" features.

None of this proves the split is leak-free. It rules out the cheap failure modes (the same picture in
train and test, a named painter on both sides) and shows how much of the task a shortcut can solve
(the painting's style, its original shape) before any pixel is looked at.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

UNKNOWN_ARTIST = "Unknown Artist"


def dhash(image: Image.Image | Path | str, hash_size: int = 8) -> np.ndarray:
    """Difference hash: `hash_size`**2 booleans, true where a pixel is brighter than its left neighbour.

    The image is turned to grey and shrunk to (hash_size + 1) x hash_size, so the hash survives
    rescaling, re-compression and mild noise, but not crops, flips or colour-only changes.
    """
    if not isinstance(image, Image.Image):
        with Image.open(image) as img:
            return dhash(img.convert("L"), hash_size)
    grey = np.asarray(image.convert("L").resize((hash_size + 1, hash_size)), dtype=np.int16)
    return (grey[:, 1:] > grey[:, :-1]).ravel()


def hamming_matrix(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Hamming distances between the rows of two boolean matrices (`b` defaults to `a`)."""
    a = np.asarray(a, dtype=np.float32)
    b = a if b is None else np.asarray(b, dtype=np.float32)
    return np.rint(a @ (1 - b).T + (1 - a) @ b.T).astype(np.int32)


def near_duplicate_pairs(hashes: np.ndarray, max_distance: int) -> list[tuple[int, int, int]]:
    """All pairs (i, j, distance) with i < j whose hashes differ in at most `max_distance` bits."""
    dist = hamming_matrix(hashes)
    i, j = np.where(np.triu(dist <= max_distance, k=1))
    return [(int(a), int(b), int(dist[a, b])) for a, b in zip(i, j)]


def mirrored_duplicate_pairs(hashes: np.ndarray, mirrored: np.ndarray, max_distance: int) -> list[tuple[int, int, int]]:
    """Pairs (i, j, distance), i < j, where one image matches the horizontal mirror of the other.

    `mirrored[k]` is the hash of image k after a left-right flip. A pair counts if either direction matches.
    """
    dist = hamming_matrix(hashes, mirrored)
    dist = np.minimum(dist, dist.T)
    i, j = np.where(np.triu(dist <= max_distance, k=1))
    return [(int(a), int(b), int(dist[a, b])) for a, b in zip(i, j)]


def cross_split_pairs(
    pairs: Iterable[tuple[int, int, int]], ids: Sequence[str], parts: Mapping[str, Iterable[str]]
) -> list[tuple[int, int, int, str, str]]:
    """Keep the pairs whose two images fall in different parts: (i, j, distance, part_i, part_j).

    `ids[k]` is the id of image k, `parts` maps a part name to the ids it contains.
    """
    where = {i: name for name, members in parts.items() for i in members}
    return [(i, j, d, where[ids[i]], where[ids[j]]) for i, j, d in pairs if where[ids[i]] != where[ids[j]]]


def split_composition(parts: Mapping[str, pd.DataFrame]) -> dict[str, dict]:
    """Size, unknown-artist share and class shares of each part (needs `label` and `artist` columns).

    `class_spread` is the gap, in share of the part, between its most and least frequent class.
    """
    out = {}
    for name, df in parts.items():
        shares = df["label"].value_counts(normalize=True)
        out[name] = {
            "n": int(len(df)),
            "unknown_artist_share": float((df["artist"] == UNKNOWN_ARTIST).mean()),
            "class_share": {k: float(v) for k, v in shares.sort_index().items()},
            "class_spread": float(shares.max() - shares.min()),
        }
    return out


def named_artist_leaks(parts: Mapping[str, pd.DataFrame]) -> set[str]:
    """Named painters present in more than one part (should be empty: that is the point of the split)."""
    seen: dict[str, set[str]] = {}
    for name, df in parts.items():
        for artist in set(df["artist"]) - {UNKNOWN_ARTIST}:
            seen.setdefault(artist, set()).add(name)
    return {artist for artist, where in seen.items() if len(where) > 1}


def metadata_features(df: pd.DataFrame, kind: str) -> np.ndarray:
    """Features that use no pixel: `style` (one-hot), `format` (log aspect ratio, log area) or `style+format`."""
    style = pd.get_dummies(df["style"]).to_numpy(dtype=float)
    shape = np.column_stack(
        [np.log(df["orig_width"] / df["orig_height"]), np.log(df["orig_width"] * df["orig_height"])]
    )
    return {"style": style, "format": shape, "style+format": np.hstack([style, shape])}[kind]


def neighbour_genre_agreement(df: pd.DataFrame, max_gap: int | None = None) -> tuple[int, float]:
    """Consecutive Unknown-Artist images of the same shard: (number of pairs, share with the same genre).

    The dataset is ordered in blocks of artists, so neighbouring rows may share a hand. With `max_gap`, only
    pairs at most that many rows apart count. Needs `shard`, `row`, `label` and `artist` columns.
    """
    u = df[df["artist"] == UNKNOWN_ARTIST].sort_values(["shard", "row"])
    shard, row, label = u["shard"].to_numpy(), u["row"].to_numpy(), u["label"].to_numpy()
    keep = shard[1:] == shard[:-1]
    if max_gap is not None:
        keep &= (row[1:] - row[:-1]) <= max_gap
    pairs = int(keep.sum())
    return pairs, float((label[1:] == label[:-1])[keep].mean()) if pairs else float("nan")


def same_shard_genre_agreement(df: pd.DataFrame) -> float:
    """Share of Unknown-Artist pairs from the same shard (any distance) that have the same genre.

    This is what neighbouring rows would score if only the shard mattered, not how close the rows are.
    """
    same = total = 0.0
    for _, group in df[df["artist"] == UNKNOWN_ARTIST].groupby("shard"):
        counts = group["label"].value_counts().to_numpy(dtype=float)
        same += (counts * (counts - 1) / 2).sum()
        total += len(group) * (len(group) - 1) / 2
    return float(same / total)


def shard_majority_accuracy(train: pd.DataFrame, test: pd.DataFrame) -> float:
    """Accuracy of predicting a test image's genre as the most common genre of its shard in the training part.

    A shard is where the image sat in the source dataset, not something a model sees: it measures how much the
    way the subset was built ties genre to source, a bias that pixel models could pick up through style.
    """
    fallback = train["label_idx"].mode().iloc[0]
    by_shard = train.groupby("shard")["label_idx"].agg(lambda s: s.mode().iloc[0])
    return float((test["shard"].map(by_shard).fillna(fallback).to_numpy() == test["label_idx"].to_numpy()).mean())
