"""Does the fine-tuned ResNet-18 survive camera-like conditions, and does photo-style augmentation help?

The paintings are clean WikiArt scans, but the guide will see phone photos. With no labelled photos, we
test a proxy: nine deterministic perturbations (tilt, blur, oblique view, light, close-up, compression...)
applied to the test images, on a ResNet-18 (layer4 + head fine-tuned, as in `train_resnet.py`) trained with

  - "standard": the augmentation of `train_resnet.py` (random crop 70-100% + horizontal flip);
  - "photo":    `photo_train_transform` (crop 50-100%, flip, rotation +-20 deg, perspective, colour jitter, blur).

Same seeds, same splits, same hyper-parameters; only the training augmentation differs.

    python scripts/photo_robustness.py --seeds 5

Writes `reports/photo_robustness.json` and `reports/figures/photo_robustness.png`.
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from museum_guide.data import DATA_DIR, device, load_metadata, loaders, make_transforms, split
from museum_guide.models import pretrained_resnet18
from museum_guide.perturb import PERTURBATIONS, loaders_with_train_transform, photo_train_transform
from museum_guide.robustness import mean_std
from museum_guide.train import REPORTS, fit

SIZE = 128
CONDITIONS = ("clean", *PERTURBATIONS)
VARIANTS = ("standard", "photo")


def test_tensors(test_df, size=SIZE):
    """The test images under every condition, as normalised tensors."""
    tf = make_transforms(size, augment=False)
    imgs = [Image.open(DATA_DIR / p).convert("RGB") for p in test_df["path"]]
    out = {"clean": torch.stack([tf(i) for i in imgs])}
    for name, fn in PERTURBATIONS.items():
        out[name] = torch.stack([tf(fn(i)) for i in imgs])
    return out


@torch.no_grad()
def accuracy(model, x, y, dev, batch=64):
    model.eval().to(dev)
    pred = torch.cat([model(x[i : i + batch].to(dev)).argmax(1).cpu() for i in range(0, len(x), batch)])
    return float((pred == y).float().mean())


def train(variant, parts, seed, n_classes, dev):
    if variant == "standard":
        data = loaders(parts, SIZE, augment=True, batch_size=32)
    else:
        data = loaders_with_train_transform(parts, SIZE, photo_train_transform(SIZE), batch_size=32)
    torch.manual_seed(seed)
    model = pretrained_resnet18(n_classes, train_layer4=True)
    fit(model, data, dev, epochs=20, lr=3e-4, weight_decay=1e-4, patience=8, verbose=False)
    return model


def summarise(per_seed):
    acc = {v: {c: [s[v][c] for s in per_seed] for c in CONDITIONS} for v in VARIANTS}
    stats = {v: {c: dict(zip(("mean", "std"), mean_std(acc[v][c]))) for c in CONDITIONS} for v in VARIANTS}
    perturbed = {v: [float(np.mean([s[v][c] for c in PERTURBATIONS])) for s in per_seed] for v in VARIANTS}
    comparison = {}
    for c in CONDITIONS:
        diff = np.array(acc["photo"][c]) - np.array(acc["standard"][c])
        comparison[c] = {
            "mean_diff": float(diff.mean()),
            "std_diff": mean_std(diff)[1],
            "seeds_photo_better": int((diff > 0).sum()),
            "seeds_standard_better": int((diff < 0).sum()),
        }
    diff = np.array(perturbed["photo"]) - np.array(perturbed["standard"])
    comparison["mean_over_perturbations"] = {
        "standard": dict(zip(("mean", "std"), mean_std(perturbed["standard"]))),
        "photo": dict(zip(("mean", "std"), mean_std(perturbed["photo"]))),
        "mean_diff": float(diff.mean()),
        "std_diff": mean_std(diff)[1],
        "seeds_photo_better": int((diff > 0).sum()),
    }
    return {"accuracy": stats, "photo_minus_standard": comparison}


def plot(summary, n_seeds, path):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x, w = np.arange(len(CONDITIONS)), 0.38
    for k, (v, colour) in enumerate(zip(VARIANTS, ("tab:blue", "tab:orange"))):
        s = summary["accuracy"][v]
        ax.bar(x + (k - 0.5) * w, [s[c]["mean"] for c in CONDITIONS], w, yerr=[s[c]["std"] for c in CONDITIONS],
               color=colour, label=f"{v} augmentation", capsize=2)
    ax.set_xticks(x, [c.replace("_", "\n") for c in CONDITIONS], fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"ResNet-18 under camera-like perturbations (mean ± std over {n_seeds} seeds)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0 .. N-1); seed 0 is the README split")
    args = p.parse_args()

    dev, df = device(), load_metadata()
    n_classes = df["label_idx"].nunique()
    per_seed = []
    for seed in range(args.seeds):
        t = time.time()
        parts = split(df, seed)
        x, y = test_tensors(parts["test"]), torch.tensor(parts["test"]["label_idx"].values)
        row = {"seed": seed, "n_test": len(y)}
        for variant in VARIANTS:
            model = train(variant, parts, seed, n_classes, dev)
            row[variant] = {c: accuracy(model, x[c], y, dev) for c in CONDITIONS}
        per_seed.append(row)
        print(f"seed {seed} ({time.time() - t:.0f}s)  clean {row['standard']['clean']:.3f} -> {row['photo']['clean']:.3f}  "
              f"mean perturbed {np.mean([row['standard'][c] for c in PERTURBATIONS]):.3f} -> "
              f"{np.mean([row['photo'][c] for c in PERTURBATIONS]):.3f}", flush=True)

    summary = summarise(per_seed)
    results = {"n_seeds": args.seeds, "perturbations": {k: (f.__doc__ or "").strip() for k, f in PERTURBATIONS.items()},
               "per_seed": per_seed, **summary}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "photo_robustness.json").write_text(json.dumps(results, indent=2))
    plot(summary, args.seeds, REPORTS / "figures" / "photo_robustness.png")

    print(f"\nTest accuracy, mean ± std over {args.seeds} seeds (standard -> photo augmentation):")
    for c in CONDITIONS:
        a, b, d = summary["accuracy"]["standard"][c], summary["accuracy"]["photo"][c], summary["photo_minus_standard"][c]
        print(f"  {c:16s} {a['mean']:.3f} ± {a['std']:.3f} -> {b['mean']:.3f} ± {b['std']:.3f}   "
              f"{d['mean_diff']:+.3f}   photo better on {d['seeds_photo_better']}/{args.seeds} seeds")
    m = summary["photo_minus_standard"]["mean_over_perturbations"]
    print(f"  {'mean perturbed':16s} {m['standard']['mean']:.3f} -> {m['photo']['mean']:.3f}   {m['mean_diff']:+.3f}   "
          f"photo better on {m['seeds_photo_better']}/{args.seeds} seeds")


if __name__ == "__main__":
    main()
