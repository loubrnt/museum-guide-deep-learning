"""Recognise the STYLE of a painting, not only its genre: the same comparison on a 6-style subset of WikiArt.

The brief asks for "its style or genre"; the rest of the project is the 5-genre task. This script reuses the
project code on `data/wikiart_style6` (6 frequent styles, 300 images each, built with `--strategy spread`, see
the README) with the same artist-disjoint split, for 10 seeds, and compares:

- logistic regression and random forest on colour/edge features (as `baseline_classical.py`);
- the ResNet-18 pretrained on ImageNet, `layer4` + head fine-tuned (as `train_resnet.py`);
- CLIP zero-shot, with two sentence templates (neither tuned on the test set), and CLIP + linear probe;
- "floors" that see no pixel: a random forest on the painting's genre label, on the shard it came from, on its
  original shape, to know how much a shortcut already explains.

    python scripts/style_task.py                                   # data/wikiart_style6, writes reports/style_task.json
    MUSEUM_GUIDE_DATA=data/wikiart_style6_fewest python scripts/style_task.py --tag _fewest   # confounded comparison

The number of classes comes from the data. Seed 0 uses `split(df, 0)`, like the other scripts.
"""
import argparse
import json
import os
import time

os.environ.setdefault("MUSEUM_GUIDE_DATA", "data/wikiart_style6")  # before museum_guide.data is imported

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from baseline_classical import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from museum_guide.clip import (
    MODEL,
    cached_embeddings,
    class_prompts,
    embed_images,
    embed_texts,
    fit_linear_probe,
    load_clip,
    zero_shot_predict,
)
from museum_guide.data import DATA_DIR, class_names, device, load_metadata, loaders, split
from museum_guide.metrics import per_class_report
from museum_guide.models import pretrained_resnet18
from museum_guide.robustness import bootstrap_accuracy, bootstrap_difference, mean_std
from museum_guide.style import metadata_floor_features, pooled_confusion, row_normalise, top_confusions
from museum_guide.train import REPORTS, fit, predict

TEMPLATES = {"style_of": "a painting in the style of {name}", "plain": "a {name} painting"}
FLOORS = ("genre", "shard", "format", "genre+format")
IMAGE_MODELS = ("logistic_regression", "random_forest", "resnet18_layer4", "clip_zero_shot_style_of", "clip_zero_shot_plain", "clip_probe")
COMPARISONS = [  # is the first model better than the second?
    ("clip_probe", "resnet18_layer4"),
    ("resnet18_layer4", "random_forest"),
    ("resnet18_layer4", "logistic_regression"),
    ("clip_zero_shot_style_of", "resnet18_layer4"),
]


def forest():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=3, random_state=0, n_jobs=-1)


def run_seed(seed, df, feats, clip_emb, text_emb, floor_x, dev, n_classes):
    """Predictions of every model on the test part of split `seed`, in the row order of `parts['test']`."""
    parts = split(df, seed)
    pos = pd.Series(np.arange(len(df)), index=df["id"].to_numpy())
    idx = {k: pos[v["id"]].to_numpy() for k, v in parts.items()}  # rows of df, in each part's own order
    y = {k: v["label_idx"].to_numpy() for k, v in parts.items()}
    preds = {}

    X = {k: np.stack([feats[p] for p in v["path"]]) for k, v in parts.items()}
    for name, model in {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    }.items():
        preds[name] = model.fit(X["train"], y["train"]).predict(X["test"])

    data = loaders(parts, 128, augment=True, batch_size=32)  # same settings as scripts/train_resnet.py
    torch.manual_seed(seed)
    resnet = pretrained_resnet18(n_classes, train_layer4=True)
    fit(resnet, data, dev, epochs=20, lr=3e-4, weight_decay=1e-4, patience=8, verbose=False)
    y_check, preds["resnet18_layer4"] = predict(resnet, data["test"], dev)
    assert (y_check == y["test"]).all()

    for tname in TEMPLATES:
        preds[f"clip_zero_shot_{tname}"] = zero_shot_predict(clip_emb[idx["test"]], text_emb[tname])
    probe, _ = fit_linear_probe(clip_emb[idx["train"]], y["train"], clip_emb[idx["val"]], y["val"])
    preds["clip_probe"] = probe.predict(clip_emb[idx["test"]])

    for kind, fx in floor_x.items():
        preds[f"floor_{kind}"] = forest().fit(fx[idx["train"]], y["train"]).predict(fx[idx["test"]])
    return parts, y["test"], preds


def plot_accuracy(per_seed, summary, path):
    names = list(summary)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, m in enumerate(names):
        accs = [s["test_acc"][m] for s in per_seed]
        floor = m.startswith("floor_")
        ax.scatter([i] * len(accs), accs, alpha=0.6, color="tab:gray" if floor else "tab:blue", zorder=3)
        ax.hlines(summary[m]["mean"], i - 0.3, i + 0.3, color="black", zorder=4)
    ax.axhline(1 / 6, color="tab:red", ls="--", label="chance (6 classes)")
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Style recognition, {len(per_seed)} seeds (blue: models that see the image, grey: metadata floors)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusions(cms, classes, path):
    fig, axes = plt.subplots(1, len(cms), figsize=(6.2 * len(cms), 5.2))
    for ax, (name, cm) in zip(np.atleast_1d(axes), cms.items()):
        ax.imshow(row_normalise(cm), cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes, fontsize=8)
        for i in range(len(classes)):
            for j in range(len(classes)):
                v = row_normalise(cm)[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8, color="white" if v > 0.5 else "black")
        ax.set_title(f"{name} (pooled over seeds, rows = true style)")
        ax.set_xlabel("predicted")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def composition(df, classes):
    named = df[df["artist"] != "Unknown Artist"]
    by_shard = pd.crosstab(df["label"], df["shard"])
    return {
        "n_images": len(df),
        "classes": classes,
        "unknown_artist_share": float((df["artist"] == "Unknown Artist").mean()),
        "named_artists": int(named["artist"].nunique()),
        "named_artists_per_class": {c: int(named[named["label"] == c]["artist"].nunique()) for c in classes},
        "n_shards": int(df["shard"].nunique()),
        "largest_shard_share_per_class": {c: float(by_shard.loc[c].max() / by_shard.loc[c].sum()) for c in classes},
        "genre_label_counts_per_class": {c: df[df["label"] == c]["genre"].value_counts().to_dict() for c in classes},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--tag", default="", help="suffix of the output files, e.g. _fewest")
    args = ap.parse_args()

    dev, df = device(), load_metadata()
    classes = class_names(df)
    n_classes = len(classes)
    paths = [DATA_DIR / q for q in df["path"]]
    t0 = time.time()
    feats = {p: features(DATA_DIR / p) for p in df["path"]}
    model, processor = load_clip()
    clip_emb = cached_embeddings(DATA_DIR / "clip_embeddings.npz", MODEL, paths, lambda: embed_images(model, processor, paths))
    text_emb = {t: embed_texts(model, processor, class_prompts(classes, tpl)) for t, tpl in TEMPLATES.items()}
    floor_x = {k: metadata_floor_features(df, k) for k in FLOORS}
    print(f"{len(df)} images, {n_classes} classes, features and CLIP vectors ready ({time.time() - t0:.0f}s)", flush=True)

    per_seed, pairs, unknown_flags, seed0 = [], {}, [], None
    for seed in range(args.seeds):
        t = time.time()
        parts, y_test, preds = run_seed(seed, df, feats, clip_emb, text_emb, floor_x, dev, n_classes)
        reports = {m: per_class_report(y_test, p, classes) for m, p in preds.items()}
        per_seed.append({"seed": seed, "n_test": len(y_test),
                         "test_acc": {m: r["accuracy"] for m, r in reports.items()},
                         "macro_f1": {m: r["macro_f1"] for m, r in reports.items()},
                         "majority_class_share": float(np.bincount(y_test).max() / len(y_test))})
        for m, p in preds.items():
            pairs.setdefault(m, []).append((y_test, p))
        unknown_flags.append((parts["test"]["artist"] == "Unknown Artist").to_numpy())
        if seed == 0:
            seed0 = {"correct": {m: (p == y_test).astype(int) for m, p in preds.items()},
                     "groups": parts["test"]["group"].to_numpy(), "report": reports}
        print(f"seed {seed} ({time.time() - t:.0f}s): " + "  ".join(f"{m} {a:.3f}" for m, a in per_seed[-1]["test_acc"].items()), flush=True)

    models = list(per_seed[0]["test_acc"])
    summary = {m: dict(zip(("mean", "std"), mean_std([s["test_acc"][m] for s in per_seed]))) for m in models}
    summary_f1 = {m: dict(zip(("mean", "std"), mean_std([s["macro_f1"][m] for s in per_seed]))) for m in models}
    wins = {f"{a}_beats_{b}": int(sum(s["test_acc"][a] > s["test_acc"][b] for s in per_seed)) for a, b in COMPARISONS}
    ci = {m: dict(zip(("acc", "low", "high"), bootstrap_accuracy(seed0["correct"][m], seed0["groups"]))) for m in models}
    diffs = {f"{a}_minus_{b}": dict(zip(("diff", "low", "high"), bootstrap_difference(seed0["correct"][a], seed0["correct"][b], seed0["groups"])))
             for a, b in COMPARISONS}

    unknown_all = np.concatenate(unknown_flags)
    by_artist = {}
    for m in IMAGE_MODELS:
        ok = np.concatenate([(y == p) for y, p in pairs[m]])
        by_artist[m] = {"unknown_artist": float(ok[unknown_all].mean()), "named_artist": float(ok[~unknown_all].mean()),
                        "n_unknown": int(unknown_all.sum()), "n_named": int((~unknown_all).sum())}

    cms = {m: pooled_confusion(pairs[m], n_classes) for m in ("resnet18_layer4", "clip_probe")}
    results = {
        "data_dir": str(DATA_DIR),
        "composition": composition(df, classes),
        "n_seeds": args.seeds,
        "chance": 1 / n_classes,
        "majority_class_share_mean": float(np.mean([s["majority_class_share"] for s in per_seed])),
        "per_seed": per_seed,
        "test_accuracy": summary,
        "macro_f1": summary_f1,
        "seeds_where_first_model_is_better": wins,
        "seed0_confidence_intervals_95": ci,
        "seed0_paired_differences_95": diffs,
        "seed0_per_class": {m: seed0["report"][m]["per_class"] for m in ("resnet18_layer4", "clip_probe")},
        "accuracy_by_artist_known": by_artist,
        "pooled_confusion_counts": {m: cm.tolist() for m, cm in cms.items()},
        "top_confusions": {m: top_confusions(cm, classes, 6) for m, cm in cms.items()},
        "clip_zero_shot_templates": TEMPLATES,
        "ci_method": "percentile bootstrap, 2000 resamples of artists (each Unknown Artist image is its own group)",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"style_task{args.tag}.json").write_text(json.dumps(results, indent=2))
    plot_accuracy(per_seed, summary, REPORTS / "figures" / f"style_task{args.tag}.png")
    plot_confusions(cms, classes, REPORTS / "figures" / f"style_task_confusion{args.tag}.png")

    print(f"\nTest accuracy over {args.seeds} seeds (chance {1 / n_classes:.3f}, majority class {results['majority_class_share_mean']:.3f}):")
    for m in models:
        print(f"  {m:26s} {summary[m]['mean']:.3f} +- {summary[m]['std']:.3f}   macro-F1 {summary_f1[m]['mean']:.3f}")
    print("\nSeed 0 paired differences (95% interval):")
    for k, v in diffs.items():
        print(f"  {k:55s} {v['diff']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]")
    print("\nTop confusions:", json.dumps(results["top_confusions"], indent=1))


if __name__ == "__main__":
    main()
