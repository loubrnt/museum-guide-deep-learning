"""Does it matter that the "frozen" part of the pretrained ResNet-18 still updates its BatchNorm statistics?

`pretrained_resnet18` freezes the backbone weights, but `fit()` puts the model in train mode, so the frozen
BatchNorm layers keep updating their running statistics and normalise each batch with its own statistics.
`bn_eval=True` keeps them in eval mode (the ImageNet statistics stay as they are).

For each seed (a seed changes the artist-disjoint split and the training randomness), this script trains both
variants of `train_resnet.py` (head only; layer4 + head) with `bn_eval` False and True, and compares the test
accuracies seed by seed.

    python scripts/bn_study.py --seeds 10

Writes `reports/bn_study.json` and `reports/figures/bn_study.png`. The default (`bn_eval=False`) at seed 0
reproduces the numbers of `scripts/train_resnet.py`.
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from museum_guide.data import device, load_metadata, loaders, split
from museum_guide.models import pretrained_resnet18
from museum_guide.robustness import mean_std
from museum_guide.train import REPORTS, fit, predict

# (name, train_layer4, learning rate): the settings of `scripts/train_resnet.py`
VARIANTS = (("head_only", False, 1e-3), ("layer4", True, 3e-4))
SIZE = 128


def run_seed(seed, df, dev):
    parts = split(df, seed)
    data = loaders(parts, SIZE, augment=True, batch_size=32)
    n_classes = df["label_idx"].nunique()
    out = {"seed": seed, "n_test": len(parts["test"])}
    for name, layer4, lr in VARIANTS:
        for bn_eval in (False, True):
            torch.manual_seed(seed)
            model = pretrained_resnet18(n_classes, train_layer4=layer4, bn_eval=bn_eval)
            fit(model, data, dev, epochs=20, lr=lr, weight_decay=1e-4, patience=8, verbose=False)
            y, pred = predict(model, data["test"], dev)
            out[f"{name}_{'bn_eval' if bn_eval else 'default'}"] = float((y == pred).mean())
    return out


def summarise(per_seed):
    summary = {}
    for name, _, _ in VARIANTS:
        default = np.array([s[f"{name}_default"] for s in per_seed])
        bn_eval = np.array([s[f"{name}_bn_eval"] for s in per_seed])
        diff = bn_eval - default
        summary[name] = {
            "default": dict(zip(("mean", "std"), mean_std(default))),
            "bn_eval": dict(zip(("mean", "std"), mean_std(bn_eval))),
            "paired_difference": dict(zip(("mean", "std"), mean_std(diff))),
            "paired_difference_standard_error": float(diff.std(ddof=1) / np.sqrt(len(diff))) if len(diff) > 1 else 0.0,
            "seeds_improved": int((diff > 0).sum()),
            "seeds_tied": int((diff == 0).sum()),
            "seeds_worse": int((diff < 0).sum()),
        }
    return summary


def plot(per_seed, summary, path):
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(9, 4), sharey=True)
    for ax, (name, _, _) in zip(axes, VARIANTS):
        for s in per_seed:
            a, b = s[f"{name}_default"], s[f"{name}_bn_eval"]
            ax.plot([0, 1], [a, b], color="tab:green" if b > a else "tab:red" if b < a else "gray", alpha=0.6, marker="o")
        for x, key in enumerate(("default", "bn_eval")):
            ax.hlines(summary[name][key]["mean"], x - 0.15, x + 0.15, color="black", lw=2, zorder=4)
        d = summary[name]["paired_difference"]["mean"]
        ax.set_xticks([0, 1], ["default", "frozen BN in eval"])
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{name.replace('_', ' ')}: {d * 100:+.1f} points on average ({summary[name]['seeds_improved']}/{len(per_seed)} seeds better)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("test accuracy (one line per seed)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=10, help="number of seeds (0 .. N-1); seed 0 is the README split")
    args = p.parse_args()

    dev, df = device(), load_metadata()
    per_seed = []
    for seed in range(args.seeds):
        t = time.time()
        per_seed.append(run_seed(seed, df, dev))
        print(f"seed {seed} ({time.time() - t:.0f}s): " + "  ".join(f"{k} {v:.3f}" for k, v in per_seed[-1].items() if k not in ("seed", "n_test")), flush=True)

    summary = summarise(per_seed)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "bn_study.json").write_text(json.dumps({"n_seeds": args.seeds, "per_seed": per_seed, "summary": summary}, indent=2))
    plot(per_seed, summary, REPORTS / "figures" / "bn_study.png")

    print("\nTest accuracy over seeds (mean +- std):")
    for name, s in summary.items():
        d = s["paired_difference"]
        print(f"  {name:10s} default {s['default']['mean']:.3f} +- {s['default']['std']:.3f}   bn_eval {s['bn_eval']['mean']:.3f} +- {s['bn_eval']['std']:.3f}   "
              f"paired difference {d['mean']:+.3f} +- {s['paired_difference_standard_error']:.3f} (s.e.); better on {s['seeds_improved']}/{args.seeds} seeds")


if __name__ == "__main__":
    main()
