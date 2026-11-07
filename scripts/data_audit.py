"""Sanity checks on the WikiArt subset: duplicates, split composition, and how far metadata alone gets.

1. Near-duplicate audit: perceptual hashes (8x8 and 16x16 difference hashes, and a mirrored 8x8 search)
   over all 1500 images, counting close pairs and those that fall on two sides of a split.
2. Split composition for seeds 0-9: size, share of "Unknown Artist" images and class imbalance per part,
   and a hard check that no named painter appears in two parts.
3. Neighbouring rows: the dataset is ordered in blocks of artists, so consecutive "Unknown Artist" rows might share a
   hand. We check whether neighbours look alike beyond what their shard (source block) already explains.
4. Floors: a random forest that sees no pixel (the painting's style, its original shape) predicts the genre.
   If a pixel model is not clearly above this, it may have learnt the style, not the genre.

    python scripts/data_audit.py

Writes `reports/data_audit.json`. Seed 0 is the split of the README.
"""
import json

import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier

from museum_guide.audit import (
    cross_split_pairs,
    dhash,
    metadata_features,
    mirrored_duplicate_pairs,
    named_artist_leaks,
    near_duplicate_pairs,
    neighbour_genre_agreement,
    same_shard_genre_agreement,
    shard_majority_accuracy,
    split_composition,
)
from museum_guide.data import DATA_DIR, load_metadata, split
from museum_guide.robustness import mean_std
from museum_guide.train import REPORTS

SEEDS = range(10)
FRACTIONS = (0.0, 0.05, 0.10, 0.15)  # max Hamming distance as a share of the hash bits
LIMITS = [
    "A perceptual hash only catches near-identical copies (same picture, rescaled, recompressed or lightly noisy; "
    "mirrored copies are checked separately). A crop, a different photograph of the same canvas or a colour-only "
    "change can slip through.",
    "Unknown Artist is 48% of the images and each one is its own split group, so the artist-disjoint guarantee "
    "covers named painters only. This audit cannot tell whether two unknown paintings share a hand.",
    "The shard is where an image sat in the source dataset (5 shards here, chosen to build the classes), not "
    "something a model sees. The shard floor measures a construction bias; it is not a feature any model has.",
    "Metadata floors use the original WikiArt labels; they show how much a shortcut explains, not what a model learnt.",
]


def hash_audit(df):
    """Count near-duplicate pairs (and pairs split across parts for each seed) at several thresholds."""
    hashes = {"dhash8": [], "dhash16": [], "dhash8_mirrored": []}
    for p in df["path"]:
        with Image.open(DATA_DIR / p) as img:
            grey = img.convert("L")
            hashes["dhash8"].append(dhash(grey, 8))
            hashes["dhash16"].append(dhash(grey, 16))
            hashes["dhash8_mirrored"].append(dhash(ImageOps.mirror(grey), 8))
    H = {k: np.stack(v) for k, v in hashes.items()}
    ids = df["id"].tolist()
    seed_parts = {s: {k: v["id"].tolist() for k, v in split(df, s).items()} for s in SEEDS}

    out, closest = {}, []
    for name, kind in (("dhash8", "plain"), ("dhash16", "plain"), ("dhash8_mirrored", "mirror")):
        bits = H[name].shape[1]
        out[name] = {}
        for frac in FRACTIONS:
            thr = int(frac * bits)
            if kind == "plain":
                pairs = near_duplicate_pairs(H[name], thr)
            else:
                pairs = mirrored_duplicate_pairs(H["dhash8"], H["dhash8_mirrored"], thr)
            crossing = [len(cross_split_pairs(pairs, ids, seed_parts[s])) for s in SEEDS]
            out[name][f"max_distance_{thr}"] = {
                "bits": bits,
                "pairs": len(pairs),
                "pairs_across_parts_per_seed": crossing,
                "pairs_across_parts_seed0": crossing[0],
            }
    # the closest pairs of the 64-bit hash, to look at by eye
    for i, j, d in sorted(near_duplicate_pairs(H["dhash8"], int(max(FRACTIONS) * 64)), key=lambda t: t[2])[:5]:
        a, b = df.iloc[i], df.iloc[j]
        closest.append({"distance": d, "a": a["path"], "b": b["path"], "same_genre": bool(a["label"] == b["label"])})
    return out, closest


def composition_audit(df):
    per_seed, leaks = {}, {}
    for s in SEEDS:
        parts = split(df, s)
        leaks[s] = sorted(named_artist_leaks(parts))
        assert not leaks[s], f"seed {s}: named artists in several parts: {leaks[s]}"
        per_seed[s] = split_composition(parts)
    summary = {}
    for part in ("train", "val", "test"):
        unk = [per_seed[s][part]["unknown_artist_share"] for s in SEEDS]
        spread = [per_seed[s][part]["class_spread"] for s in SEEDS]
        n = [per_seed[s][part]["n"] for s in SEEDS]
        summary[part] = {
            "n": {"min": min(n), "max": max(n)},
            "unknown_artist_share": {"min": min(unk), "max": max(unk)},
            "class_spread": {"min": min(spread), "max": max(spread)},
        }
    return {"seed0": per_seed[0], "across_seeds": summary, "named_artists_in_several_parts": leaks}


def neighbour_audit(df):
    """Do consecutive Unknown-Artist rows share a genre more often than any two rows of the same shard?"""
    out = {"same_shard_any_distance": same_shard_genre_agreement(df), "chance_if_independent": 0.2, "by_row_gap": {}}
    for gap in (1, 5, 20, None):
        pairs, share = neighbour_genre_agreement(df, gap)
        out["by_row_gap"]["any" if gap is None else f"<={gap}"] = {"pairs": pairs, "same_genre_share": share}
    return out


def floors(df):
    """Genre accuracy of metadata-only forests, and of always predicting the most common training class."""
    kinds = ("style", "format", "style+format")
    X = {k: metadata_features(df, k) for k in kinds}
    row = {i: n for n, i in enumerate(df["id"])}
    y = df["label_idx"].to_numpy()
    acc = {k: [] for k in (*kinds, "majority_class", "shard_majority")}
    for s in SEEDS:
        parts = split(df, s)
        tr = [row[i] for i in parts["train"]["id"]]
        te = [row[i] for i in parts["test"]["id"]]
        for k in kinds:
            forest = RandomForestClassifier(200, min_samples_leaf=3, random_state=0).fit(X[k][tr], y[tr])
            acc[k].append(float(forest.score(X[k][te], y[te])))
        acc["shard_majority"].append(shard_majority_accuracy(parts["train"], parts["test"]))
        acc["majority_class"].append(float((y[te] == np.bincount(y[tr]).argmax()).mean()))
    return {k: {"per_seed": v, **dict(zip(("mean", "std"), mean_std(v)))} for k, v in acc.items()}


def main():
    df = load_metadata()
    print(f"{len(df)} images, {df['artist'].nunique() - 1} named artists, "
          f"{(df['artist'] == 'Unknown Artist').mean():.1%} unknown\n")
    hashes, closest = hash_audit(df)
    comp = composition_audit(df)
    neigh = neighbour_audit(df)
    fl = floors(df)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "data_audit.json").write_text(json.dumps(
        {"n_images": len(df), "seeds": list(SEEDS), "near_duplicates": hashes, "closest_pairs_dhash8": closest,
         "split_composition": comp, "neighbouring_rows": neigh, "metadata_floors": fl, "limits": LIMITS}, indent=2))

    print("Near-duplicate pairs (all 1500 images; pairs across parts for seed 0 / over 10 seeds):")
    for name, rows in hashes.items():
        for key, r in rows.items():
            print(f"  {name:16s} {key:16s} pairs {r['pairs']:3d}   across parts: seed0 {r['pairs_across_parts_seed0']}, "
                  f"max over seeds {max(r['pairs_across_parts_per_seed'])}")
    print("\nClosest pairs (64-bit hash):")
    for c in closest:
        print(f"  d={c['distance']}  {c['a']}  ~  {c['b']}  same genre: {c['same_genre']}")
    print("\nNamed artists present in several parts: none, for all 10 seeds (checked).")
    print("Split composition across seeds (min - max):")
    for part, s in comp["across_seeds"].items():
        print(f"  {part:5s} n {s['n']['min']}-{s['n']['max']}   unknown artists {s['unknown_artist_share']['min']:.0%}-"
              f"{s['unknown_artist_share']['max']:.0%}   class spread {s['class_spread']['min']:.0%}-{s['class_spread']['max']:.0%}")
    print("\nConsecutive Unknown-Artist rows of one shard: share with the same genre (chance 20%):")
    for gap, r in neigh["by_row_gap"].items():
        print(f"  row gap {gap:>4s}: {r['same_genre_share']:.1%}  ({r['pairs']} pairs)")
    print(f"  any two unknown images of one shard: {neigh['same_shard_any_distance']:.1%}  (what the shard alone explains)")
    print("\nGenre accuracy without any pixel (test, mean +- std over 10 seeds; chance 20%):")
    for k, v in fl.items():
        print(f"  {k:15s} {v['mean']:.1%} +- {v['std']:.1%}")
    print("\nLimits:")
    for line in LIMITS:
        print(f"  - {line}")


if __name__ == "__main__":
    main()
