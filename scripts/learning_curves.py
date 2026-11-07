"""How many labelled paintings does the guide need? Learning curves and few-shot behaviour.

A small team with a limited budget has to label paintings by hand, and a museum will want to add new works. For the
5-genre task and the 6-style task, on the same artist-disjoint splits as the rest of the project (seeds 0 to 4), this
script trains on a growing part of the TRAIN split and tests on the unchanged test split:

- CLIP image vectors + a linear probe (regularisation C = 1, fixed: it uses no validation label);
- the fine-tuned ResNet-18, `layer4` + head (same settings as `train_resnet.py`; it still uses the 249 validation
  images for early stopping at every size, which flatters it at small sizes);
- a random forest on colour/edge features;
- CLIP zero-shot (no training image at all), as a flat reference.

A training subset of about n images is class-balanced and drawn artist group by artist group (see
`museum_guide.curves.subsample_train`). A few-shot view for CLIP alone draws k = 1, 2, 5, 10, 20 images per class,
10 draws per seed. The number of images needed to reach 70, 80 and 85% is read on the mean curve.

    python scripts/learning_curves.py                  # both tasks, about 25 minutes (the ResNet dominates)
    python scripts/learning_curves.py --skip-resnet    # a few minutes

Each task runs in its own process because the data folder is read when `museum_guide.data` is imported.
Writes `reports/learning_curves.json` and `reports/figures/learning_curves.png`.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TASK_DATA = {"genre": "data/wikiart_genre5", "style": "data/wikiart_style6"}
if "--worker" in sys.argv:  # before museum_guide.data is imported
    os.environ["MUSEUM_GUIDE_DATA"] = TASK_DATA[sys.argv[sys.argv.index("--worker") + 1]]

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from baseline_classical import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from museum_guide.clip import (
    MODEL,
    cached_embeddings,
    class_prompts,
    embed_images,
    embed_texts,
    load_clip,
    zero_shot_predict,
)
from museum_guide.curves import few_shot_indices, images_needed, subsample_train
from museum_guide.data import DATA_DIR, class_names, device, load_metadata, loaders, split
from museum_guide.models import pretrained_resnet18
from museum_guide.robustness import mean_std
from museum_guide.train import REPORTS, fit, predict

SIZES = (25, 50, 100, 200, 400)
FEW_SHOT_KS = (1, 2, 5, 10, 20)
TARGETS = (0.70, 0.80, 0.85)
# One sentence template per task for the zero-shot reference line; neither is tuned on a test set.
PROMPTS = {"genre": "a painting of a {name}", "style": "a painting in the style of {name}"}
MODELS = ("clip_probe", "resnet18_layer4", "random_forest")


def run_task(task, seeds, sizes, draws, ks, skip_resnet):
    """All curves for one task, in this process (whose data folder is the task's)."""
    dev, df = device(), load_metadata()
    classes = class_names(df)
    n_classes = len(classes)
    paths = [DATA_DIR / q for q in df["path"]]
    t0 = time.time()
    x_feat = np.stack([features(DATA_DIR / p) for p in df["path"]])  # rows aligned with df
    model, processor = load_clip()
    clip_emb = cached_embeddings(DATA_DIR / "clip_embeddings.npz", MODEL, paths, lambda: embed_images(model, processor, paths))
    text_emb = embed_texts(model, processor, class_prompts(classes, PROMPTS[task]))
    pos = pd.Series(np.arange(len(df)), index=df["id"].to_numpy())
    print(f"[{task}] {len(df)} images, {n_classes} classes, features and CLIP vectors ready ({time.time() - t0:.0f}s)", flush=True)

    nominal = list(sizes) + ["full"]
    acc = {m: [[] for _ in nominal] for m in MODELS}  # [model][size][seed]
    actual = [[] for _ in nominal]
    zero_shot, few = [], {k: {"probe": [], "zero_shot": []} for k in ks}
    for seed in range(seeds):
        t = time.time()
        parts = split(df, seed)
        idx = {k: pos[v["id"]].to_numpy() for k, v in parts.items()}
        y = {k: v["label_idx"].to_numpy() for k, v in parts.items()}
        zs = float((zero_shot_predict(clip_emb[idx["test"]], text_emb) == y["test"]).mean())
        zero_shot.append(zs)

        for si, n in enumerate(nominal):
            sub = np.arange(len(parts["train"])) if n == "full" else subsample_train(parts["train"], n, seed * 100_003 + n)
            rows, ytr = idx["train"][sub], y["train"][sub]
            actual[si].append(len(sub))
            probe = LogisticRegression(C=1.0, max_iter=3000).fit(clip_emb[rows], ytr)
            acc["clip_probe"][si].append(float(probe.score(clip_emb[idx["test"]], y["test"])))
            forest = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1)
            acc["random_forest"][si].append(float(forest.fit(x_feat[rows], ytr).score(x_feat[idx["test"]], y["test"])))
            if not skip_resnet:
                subset = {"train": parts["train"].iloc[sub].reset_index(drop=True), "val": parts["val"], "test": parts["test"]}
                data = loaders(subset, 128, augment=True, batch_size=32)  # the settings of scripts/train_resnet.py
                torch.manual_seed(seed)
                resnet = pretrained_resnet18(n_classes, train_layer4=True)
                fit(resnet, data, dev, epochs=20, lr=3e-4, weight_decay=1e-4, patience=8, verbose=False)
                y_check, pred = predict(resnet, data["test"], dev)
                assert (y_check == y["test"]).all()
                acc["resnet18_layer4"][si].append(float((pred == y_check).mean()))

        for k in ks:  # few-shot: k labelled images per class, CLIP probe against zero-shot
            for draw in range(draws):
                sub = few_shot_indices(y["train"], k, seed * 1_000_003 + k * 1009 + draw)
                probe = LogisticRegression(C=1.0, max_iter=3000).fit(clip_emb[idx["train"][sub]], y["train"][sub])
                few[k]["probe"].append(float(probe.score(clip_emb[idx["test"]], y["test"])))
                few[k]["zero_shot"].append(zs)
        last = {m: round(v[-1][-1], 3) for m, v in acc.items() if v[-1]}
        print(f"[{task}] seed {seed} ({time.time() - t:.0f}s), full train: {last}, zero-shot {zs:.3f}", flush=True)

    mean_size = [float(np.mean(a)) for a in actual]
    curves = {}
    for m, per_size in acc.items():
        if not per_size[0]:  # the ResNet was skipped
            continue
        stats = [mean_std(v) for v in per_size]
        entry = {"mean": [s[0] for s in stats], "std": [s[1] for s in stats], "per_seed": per_size}
        entry["images_needed"] = {str(round(t * 100)): images_needed(mean_size, entry["mean"], t) for t in TARGETS}
        for r in entry["images_needed"].values():
            r["per_class"] = None if r["images"] is None else r["images"] / n_classes
        entry["share_of_full_accuracy"] = [v / entry["mean"][-1] for v in entry["mean"]]
        curves[m] = entry
    return {
        "data_dir": str(DATA_DIR),
        "classes": classes,
        "n_classes": n_classes,
        "chance": 1 / n_classes,
        "n_seeds": seeds,
        "nominal_sizes": nominal,
        "actual_size_mean": mean_size,
        "actual_size_per_seed": actual,
        "curves": curves,
        "zero_shot": {"template": PROMPTS[task], **dict(zip(("mean", "std"), mean_std(zero_shot))), "per_seed": zero_shot},
        "few_shot": {
            "ks": list(ks),
            "draws_per_seed": draws,
            "probe": {str(k): dict(zip(("mean", "std"), mean_std(v["probe"]))) | {"n_runs": len(v["probe"])} for k, v in few.items()},
            "zero_shot": {str(k): dict(zip(("mean", "std"), mean_std(v["zero_shot"]))) for k, v in few.items()},
        },
    }


def plot(results, path):
    tasks = list(results["tasks"])
    fig, axes = plt.subplots(2, len(tasks), figsize=(6.4 * len(tasks), 9.0), squeeze=False)
    colours = {"clip_probe": "tab:blue", "resnet18_layer4": "tab:red", "random_forest": "tab:green"}
    labels = {"clip_probe": "CLIP + linear probe", "resnet18_layer4": "ResNet-18 fine-tuned", "random_forest": "random forest"}
    for col, task in enumerate(tasks):
        r = results["tasks"][task]
        ax = axes[0, col]
        sizes = r["actual_size_mean"]
        for m, c in r["curves"].items():
            mu, sd = np.array(c["mean"]), np.array(c["std"])
            ax.plot(sizes, mu, "-o", color=colours[m], label=labels[m])
            ax.fill_between(sizes, mu - sd, mu + sd, color=colours[m], alpha=0.15)
        ax.axhline(r["zero_shot"]["mean"], color="tab:blue", ls=":", label="CLIP zero-shot (no training image)")
        ax.axhline(r["chance"], color="gray", ls="--", label="chance")
        top = max(max(np.array(c["mean"]) + np.array(c["std"])) for c in r["curves"].values())
        top = max(top, r["zero_shot"]["mean"]) + 0.04
        for t in TARGETS:
            if t < top:
                ax.axhline(t, color="black", alpha=0.12, lw=0.8)
        ax.set_ylim(r["chance"] - 0.05, top)
        ax.set_xscale("log")
        ax.set_xlabel("labelled training images (log scale)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{task}: learning curves ({r['n_seeds']} seeds, mean +- std)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)

        ax = axes[1, col]
        ks = r["few_shot"]["ks"]
        mu = np.array([r["few_shot"]["probe"][str(k)]["mean"] for k in ks])
        sd = np.array([r["few_shot"]["probe"][str(k)]["std"] for k in ks])
        ax.errorbar(ks, mu, yerr=sd, fmt="-o", color="tab:blue", capsize=3, label="CLIP + linear probe")
        ax.axhline(r["zero_shot"]["mean"], color="tab:blue", ls=":", label="CLIP zero-shot")
        ax.axhline(r["chance"], color="gray", ls="--", label="chance")
        ax.set_xscale("log")
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.set_xlabel("labelled images per class (k)")
        ax.set_ylabel("test accuracy")
        ax.set_title(f"{task}: few-shot, {r['few_shot']['draws_per_seed']} draws x {r['n_seeds']} seeds")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def print_summary(results):
    for task, r in results["tasks"].items():
        print(f"\n== {task} (chance {r['chance']:.3f}, zero-shot {r['zero_shot']['mean']:.3f}) ==")
        print("  size(mean)  " + "  ".join(f"{m:>16s}" for m in r["curves"]))
        for i, s in enumerate(r["actual_size_mean"]):
            print(f"  {s:9.0f}   " + "  ".join(f"{c['mean'][i]:.3f} +- {c['std'][i]:.3f}" for c in r["curves"].values()))
        for m, c in r["curves"].items():
            need = {t: ("not reached" if v["images"] is None else f"{v['images']:.0f}{'-' if v['at_most'] else ''}") for t, v in c["images_needed"].items()}
            print(f"  images needed, {m}: {need}")
        print("  few-shot probe: " + ", ".join(f"k={k}: {r['few_shot']['probe'][str(k)]['mean']:.3f}" for k in r["few_shot"]["ks"]))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=("genre", "style", "both"), default="both")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--sizes", type=int, nargs="+", default=list(SIZES), help="nominal training-set sizes (plus the full train split)")
    p.add_argument("--draws", type=int, default=10, help="few-shot draws per seed and per k")
    p.add_argument("--few-shot-ks", type=int, nargs="+", default=list(FEW_SHOT_KS))
    p.add_argument("--skip-resnet", action="store_true", help="skip the fine-tuned ResNet (the slow part)")
    p.add_argument("--worker", choices=tuple(TASK_DATA), help="internal: run one task in this process")
    p.add_argument("--partial", help="internal: where a worker writes its result")
    args = p.parse_args()

    if args.worker:
        result = run_task(args.worker, args.seeds, args.sizes, args.draws, args.few_shot_ks, args.skip_resnet)
        Path(args.partial).write_text(json.dumps(result))
        return

    tasks = list(TASK_DATA) if args.task == "both" else [args.task]
    results = {
        "method": {
            "split": "museum_guide.data.split(df, seed), artist-disjoint, seeds 0 to n_seeds-1; val and test parts never shrink",
            "subsample": "class-balanced, whole artist groups added in random order, the last one trimmed; see museum_guide.curves.subsample_train",
            "clip_probe": "logistic regression, C = 1 fixed: no validation label is used (the project's own probe chooses C on the validation set)",
            "resnet": "settings of scripts/train_resnet.py; the 249-image validation set is used for early stopping at every size",
            "thresholds_percent": [round(t * 100) for t in TARGETS],
        },
        "tasks": {},
    }
    with tempfile.TemporaryDirectory() as tmp:
        for task in tasks:
            partial = Path(tmp) / f"{task}.json"
            cmd = [sys.executable, "-W", "ignore", __file__, "--worker", task, "--partial", str(partial), "--seeds", str(args.seeds),
                   "--sizes", *map(str, args.sizes), "--draws", str(args.draws), "--few-shot-ks", *map(str, args.few_shot_ks)]
            if args.skip_resnet:
                cmd.append("--skip-resnet")
            subprocess.run(cmd, check=True)
            results["tasks"][task] = json.loads(partial.read_text())
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "learning_curves.json").write_text(json.dumps(results, indent=2))
    plot(results, REPORTS / "figures" / "learning_curves.png")
    print_summary(results)


if __name__ == "__main__":
    main()
