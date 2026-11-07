"""How solid are the headline results? Several seeds, confidence intervals, and a shift test on the ResNet.

The README numbers come from one split and one training run, on a 279-image test set. This script:

1. repeats the whole comparison for N seeds (a seed changes the artist-disjoint split *and* the training
   randomness), for the four models of the README: logistic regression, random forest, the feed-forward
   network and the fine-tuned ResNet-18, each with the configuration used in its own script;
2. gives artist-level bootstrap confidence intervals and paired differences for seed 0 (the split used in
   the README), so we know which gaps are real;
3. runs the shift test of `train_mlp.py` (circular shift of the test images) on the ResNet as well as on
   the MLP, with the shift expressed as a fraction of the image width so 64 px and 128 px are comparable.

    python scripts/robustness.py --seeds 5

Writes `reports/robustness.json` and `reports/figures/robustness.png`. Seed 0 reproduces the README numbers.
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from baseline_classical import features
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from train_mlp import HIDDEN
from train_mlp import SIZE as MLP_SIZE
from train_resnet import SIZE as RESNET_SIZE

from museum_guide.data import DATA_DIR, device, load_metadata, loaders, split
from museum_guide.models import FeedForward, pretrained_resnet18
from museum_guide.robustness import bootstrap_accuracy, bootstrap_difference, mean_std, predict_with_shift, shift_pixels
from museum_guide.train import REPORTS, fit

SHIFT_FRACTIONS = (0.0625, 0.125)  # 0.125 of the width = the 8 px of `train_mlp.py` on a 64 px image
MODELS = ("logistic_regression", "random_forest", "mlp", "resnet18_layer4")
COMPARISONS = [  # (better?, reference): is the first model really more accurate than the second?
    ("resnet18_layer4", "random_forest"),
    ("resnet18_layer4", "logistic_regression"),
    ("random_forest", "logistic_regression"),
    ("random_forest", "mlp"),
]


def classical_models(seed):
    """Same models and settings as `baseline_classical.py`."""
    return {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    }


def shift_report(model, test_loader, size, dev):
    """Accuracy and prediction-change rate under each circular shift."""
    out = {}
    for frac in SHIFT_FRACTIONS:
        px = shift_pixels(size, frac)
        y, pred, shifted = predict_with_shift(model, test_loader, dev, px)
        out[str(frac)] = {
            "pixels": px,
            "acc": float((y == pred).mean()),
            "acc_shifted": float((y == shifted).mean()),
            "prediction_change_rate": float((pred != shifted).mean()),
        }
    return out


def run_seed(seed, df, feats, dev):
    parts = split(df, seed)
    n_classes = df["label_idx"].nunique()
    y_test = parts["test"]["label_idx"].values
    correct, result = {}, {}

    # 1-2. Classical models on colour/edge features.
    X = {k: np.stack([feats[p] for p in v["path"]]) for k, v in parts.items()}
    for name, model in classical_models(seed).items():
        model.fit(X["train"], parts["train"]["label_idx"].values)
        correct[name] = (model.predict(X["test"]) == y_test).astype(int)

    # 3. Feed-forward network: dropout 0.5 + weight decay, lr 1e-4, 60 epochs (the final config of `train_mlp.py`).
    plain = loaders(parts, MLP_SIZE, augment=False)
    torch.manual_seed(seed)
    mlp = FeedForward(3 * MLP_SIZE * MLP_SIZE, HIDDEN, n_classes, dropout=0.5)
    fit(mlp, plain, dev, epochs=60, lr=1e-4, weight_decay=1e-4, verbose=False)
    y, pred, _ = predict_with_shift(mlp, plain["test"], dev, 1)
    correct["mlp"] = (y == pred).astype(int)
    result["mlp_shift"] = shift_report(mlp, plain["test"], MLP_SIZE, dev)

    # 4. ResNet-18 pretrained, layer4 + head fine-tuned (as in `train_resnet.py`).
    data = loaders(parts, RESNET_SIZE, augment=True, batch_size=32)
    torch.manual_seed(seed)
    resnet = pretrained_resnet18(n_classes, train_layer4=True)
    fit(resnet, data, dev, epochs=20, lr=3e-4, weight_decay=1e-4, patience=8, verbose=False)
    result["resnet18_layer4_shift"] = shift_report(resnet, data["test"], RESNET_SIZE, dev)
    y, pred, _ = predict_with_shift(resnet, data["test"], dev, 1)
    correct["resnet18_layer4"] = (y == pred).astype(int)

    result["test_acc"] = {k: float(v.mean()) for k, v in correct.items()}
    result["n_test"] = len(y_test)
    return result, correct, parts["test"]["group"].values


def summarise(per_seed):
    summary = {m: dict(zip(("mean", "std"), mean_std([s["test_acc"][m] for s in per_seed]))) for m in MODELS}
    shift = {}
    for net in ("mlp", "resnet18_layer4"):
        for frac in map(str, SHIFT_FRACTIONS):
            rows = [s[f"{net}_shift"][frac] for s in per_seed]
            shift.setdefault(net, {})[frac] = {
                "pixels": rows[0]["pixels"],
                **{k: dict(zip(("mean", "std"), mean_std([r[k] for r in rows]))) for k in ("acc", "acc_shifted", "prediction_change_rate")},
            }
    wins = {f"{a}_beats_{b}": int(sum(s["test_acc"][a] > s["test_acc"][b] for s in per_seed)) for a, b in COMPARISONS}
    return {"test_acc": summary, "shift": shift, "seeds_where_first_model_is_better": wins}


def plot(per_seed, summary, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, m in enumerate(MODELS):
        accs = [s["test_acc"][m] for s in per_seed]
        ax1.scatter([i] * len(accs), accs, color="tab:blue", alpha=0.6, zorder=3, label="one seed" if i == 0 else None)
        mu = summary["test_acc"][m]["mean"]
        ax1.hlines(mu, i - 0.3, i + 0.3, color="black", zorder=4, label="mean" if i == 0 else None)
    ax1.set_xticks(range(len(MODELS)), [m.replace("_", "\n") for m in MODELS], fontsize=8)
    ax1.set_ylabel("test accuracy")
    ax1.set_title(f"Test accuracy over {len(per_seed)} seeds (split + training)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend()

    frac = str(max(SHIFT_FRACTIONS))
    width = 0.35
    for j, (net, label) in enumerate((("mlp", "MLP"), ("resnet18_layer4", "ResNet-18"))):
        s = summary["shift"][net][frac]
        ax2.bar(j - width / 2, s["acc"]["mean"], width, yerr=s["acc"]["std"], color="tab:blue", label="original" if j == 0 else None)
        ax2.bar(j + width / 2, s["acc_shifted"]["mean"], width, yerr=s["acc_shifted"]["std"], color="tab:orange", label="shifted" if j == 0 else None)
    ax2.set_xticks([0, 1], ["MLP", "ResNet-18"])
    ax2.set_ylabel("test accuracy")
    ax2.set_title(f"Circular shift by {float(frac):.1%} of the width")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0 .. N-1); seed 0 is the README split")
    args = p.parse_args()

    dev, df = device(), load_metadata()
    feats = {p: features(DATA_DIR / p) for p in df["path"]}
    per_seed, seed0 = [], None
    for seed in range(args.seeds):
        t = time.time()
        result, correct, groups = run_seed(seed, df, feats, dev)
        per_seed.append({"seed": seed, **result})
        if seed == 0:
            seed0 = (correct, groups)
        print(f"seed {seed} ({time.time() - t:.0f}s): " + "  ".join(f"{m} {a:.3f}" for m, a in result["test_acc"].items()), flush=True)

    correct, groups = seed0
    ci = {m: dict(zip(("acc", "low", "high"), bootstrap_accuracy(correct[m], groups))) for m in MODELS}
    diffs = {
        f"{a}_minus_{b}": dict(zip(("diff", "low", "high"), bootstrap_difference(correct[a], correct[b], groups)))
        for a, b in COMPARISONS
    }
    summary = summarise(per_seed)
    results = {
        "n_seeds": args.seeds,
        "n_test_seed0": len(groups),
        "per_seed": per_seed,
        "summary": summary,
        "seed0_confidence_intervals_95": ci,
        "seed0_paired_differences_95": diffs,
        "ci_method": "percentile bootstrap, 2000 resamples of artists (each Unknown Artist image is its own group)",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "robustness.json").write_text(json.dumps(results, indent=2))
    plot(per_seed, summary, REPORTS / "figures" / "robustness.png")

    print("\nTest accuracy over seeds (mean +- std):")
    for m in MODELS:
        s = summary["test_acc"][m]
        print(f"  {m:22s} {s['mean']:.3f} +- {s['std']:.3f}   seed 0: {ci[m]['acc']:.3f} [{ci[m]['low']:.3f}, {ci[m]['high']:.3f}]")
    print("\nSeed 0 paired differences (95% interval):")
    for k, v in diffs.items():
        print(f"  {k:48s} {v['diff']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]")
    print("\nCircular shift (mean over seeds):")
    for net, fr in summary["shift"].items():
        for frac, s in fr.items():
            print(f"  {net:16s} {float(frac):.2%} ({s['pixels']:2d}px): acc {s['acc']['mean']:.3f} -> {s['acc_shifted']['mean']:.3f}, "
                  f"predictions changed {s['prediction_change_rate']['mean']:.1%}")


if __name__ == "__main__":
    main()
