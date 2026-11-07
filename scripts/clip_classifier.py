"""CLIP as a recognition model: zero-shot and linear probe, on the same splits as the other models.

CLIP (the model already used for search) maps images and sentences into one space. Two ways to use it
to recognise a genre, neither of which trains a network:

- zero-shot: compare the image with a sentence per class ("a painting of a portrait"), no training;
- linear probe: a logistic regression on the frozen image vectors, trained on the training part
  (the regularisation is chosen on the validation part).

For each seed (the split of `scripts/robustness.py`) the test accuracy is compared with the fine-tuned
ResNet-18 of `reports/robustness.json`. For seed 0 the ResNet is retrained to get artist-level bootstrap
intervals on the paired difference.

    python scripts/clip_classifier.py --seeds 10

Writes `reports/clip_classifier.json` and `reports/figures/clip_classifier.png`.
The image vectors are cached in `data/wikiart_genre5/clip_embeddings.npz` (not versioned).
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

from museum_guide.clip import (
    MODEL,
    PROMPT_TEMPLATES,
    cached_embeddings,
    class_prompts,
    embed_images,
    embed_texts,
    fit_linear_probe,
    load_clip,
    zero_shot_predict,
)
from museum_guide.data import DATA_DIR, class_names, device, load_metadata, loaders, split
from museum_guide.models import pretrained_resnet18
from museum_guide.robustness import bootstrap_accuracy, bootstrap_difference, mean_std
from museum_guide.train import REPORTS, fit, predict

METHODS = ("zero_shot_simple", "zero_shot_painting", "linear_probe")


def resnet_seed0_correct(parts, n_classes):
    """ResNet-18 (layer4 + head) retrained exactly as `scripts/robustness.py` does for seed 0."""
    dev = device()
    data = loaders(parts, 128, augment=True, batch_size=32)
    torch.manual_seed(0)
    model = pretrained_resnet18(n_classes, train_layer4=True)
    fit(model, data, dev, epochs=20, lr=3e-4, weight_decay=1e-4, patience=8, verbose=False)
    y, pred = predict(model, data["test"], dev)
    return (y == pred).astype(int)


def size_and_latency(model, processor, image_path):
    """Parameters and fp32 size of the image tower vs the full checkpoint; an indicative 1-thread latency."""
    from PIL import Image

    n_full = sum(p.numel() for p in model.parameters())
    tower = list(model.vision_model.parameters()) + list(model.visual_projection.parameters())
    n_image = sum(p.numel() for p in tower)
    with Image.open(image_path) as img:
        x = processor(images=[img.convert("RGB")], return_tensors="pt")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.no_grad():
        for _ in range(3):
            model.get_image_features(**x)
        times = []
        for _ in range(20):
            t = time.perf_counter()
            model.get_image_features(**x)
            times.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(threads)
    return {
        "params_full_checkpoint": n_full,
        "params_image_tower": n_image,
        "fp32_mb_full_checkpoint": round(n_full * 4 / 1e6, 1),
        "fp32_mb_image_tower": round(n_image * 4 / 1e6, 1),
        "image_encoder_latency_ms_1thread_indicative": round(float(np.median(times)), 1),
        "latency_note": "forward pass of the image encoder only (no image loading), measured while other jobs may run: indicative, not a benchmark",
    }


def plot(per_seed, path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    seeds = [s["seed"] for s in per_seed]
    for key, label, style in (
        ("linear_probe", "CLIP linear probe", "o-"),
        ("zero_shot_painting", "CLIP zero-shot ('a painting of a ...')", "s--"),
        ("zero_shot_simple", "CLIP zero-shot ('a ...')", "^:"),
        ("resnet18_layer4", "ResNet-18 fine-tuned", "d-"),
    ):
        ax.plot(seeds, [s["test_acc"][key] for s in per_seed], style, label=label)
    ax.set_xlabel("seed (split and training)")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(seeds)
    ax.set_ylim(0.4, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    ax.set_title("5 genres: CLIP vs fine-tuned ResNet-18, same splits")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=10, help="number of seeds (0 .. N-1); seed 0 is the README split")
    p.add_argument("--cache", default=DATA_DIR / "clip_embeddings.npz", help="cache of the image vectors")
    args = p.parse_args()

    df = load_metadata()
    classes, paths = class_names(df), [DATA_DIR / q for q in df["path"]]
    n_classes = len(classes)
    model, processor = load_clip()
    t = time.time()
    images = cached_embeddings(args.cache, MODEL, paths, lambda: embed_images(model, processor, paths))
    print(f"image vectors {images.shape} ({time.time() - t:.0f}s)", flush=True)
    texts = {k: embed_texts(model, processor, class_prompts(classes, tpl)) for k, tpl in PROMPT_TEMPLATES.items()}
    row_of = {i: k for k, i in enumerate(df["id"])}
    y_all = df["label_idx"].values

    # ResNet numbers of the same seeds, from scripts/robustness.py.
    robustness = json.loads((REPORTS / "robustness.json").read_text())["per_seed"]

    per_seed, seed0 = [], None
    for seed in range(args.seeds):
        parts = split(df, seed)
        rows = {k: np.array([row_of[i] for i in v["id"]]) for k, v in parts.items()}
        y_te = y_all[rows["test"]]
        preds = {
            f"zero_shot_{k}": zero_shot_predict(images[rows["test"]], texts[k]) for k in ("simple", "painting")
        }
        probe, C = fit_linear_probe(
            images[rows["train"]], y_all[rows["train"]], images[rows["val"]], y_all[rows["val"]]
        )
        preds["linear_probe"] = probe.predict(images[rows["test"]])
        entry = {
            "seed": seed,
            "n_test": len(y_te),
            "probe_C": C,
            "test_acc": {m: float((preds[m] == y_te).mean()) for m in METHODS},
            "test_macro_f1": {m: float(f1_score(y_te, preds[m], average="macro")) for m in METHODS},
        }
        ref = robustness[seed] if seed < len(robustness) else None
        if ref is not None:
            assert ref["n_test"] == len(y_te), "robustness.json was produced with another split"
            entry["test_acc"]["resnet18_layer4"] = ref["test_acc"]["resnet18_layer4"]
        if seed == 0:
            seed0 = (preds, y_te, parts)
        per_seed.append(entry)
        print(f"seed {seed}: " + "  ".join(f"{m} {a:.3f}" for m, a in entry["test_acc"].items()), flush=True)

    keys = METHODS + ("resnet18_layer4",)
    summary = {
        "test_acc": {m: dict(zip(("mean", "std"), mean_std([s["test_acc"][m] for s in per_seed]))) for m in keys},
        "test_macro_f1": {m: dict(zip(("mean", "std"), mean_std([s["test_macro_f1"][m] for s in per_seed]))) for m in METHODS},
        "seeds_where_clip_beats_resnet": {
            m: int(sum(s["test_acc"][m] > s["test_acc"]["resnet18_layer4"] for s in per_seed)) for m in METHODS
        },
    }

    # Seed 0: artist-level bootstrap, paired with a freshly trained ResNet (reproduces the README 74.6%).
    preds0, y0, parts0 = seed0
    groups = parts0["test"]["group"].values
    resnet_ok = resnet_seed0_correct(parts0, n_classes)
    correct0 = {m: (preds0[m] == y0).astype(int) for m in METHODS}
    correct0["resnet18_layer4"] = resnet_ok
    seed0_ci = {m: dict(zip(("acc", "low", "high"), bootstrap_accuracy(c, groups))) for m, c in correct0.items()}
    seed0_diff = {
        f"{m}_minus_resnet18_layer4": dict(zip(("diff", "low", "high"), bootstrap_difference(correct0[m], resnet_ok, groups)))
        for m in METHODS
    }
    print(f"seed 0 ResNet retrained: {resnet_ok.mean():.4f} (README: 0.7455)")

    cost = size_and_latency(model, processor, paths[0])
    results = {
        "model": MODEL,
        "n_seeds": args.seeds,
        "prompts": {k: class_prompts(classes, tpl) for k, tpl in PROMPT_TEMPLATES.items()},
        "per_seed": per_seed,
        "summary": summary,
        "seed0_confidence_intervals_95": seed0_ci,
        "seed0_paired_differences_95": seed0_diff,
        "ci_method": "percentile bootstrap, 2000 resamples of artists (each Unknown Artist image is its own group)",
        "cost": cost,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "clip_classifier.json").write_text(json.dumps(results, indent=2))
    plot(per_seed, REPORTS / "figures" / "clip_classifier.png")

    print("\nTest accuracy over seeds (mean +- std):")
    for m in keys:
        s = summary["test_acc"][m]
        f1 = summary["test_macro_f1"].get(m)
        print(f"  {m:20s} {s['mean']:.3f} +- {s['std']:.3f}" + (f"   macro-F1 {f1['mean']:.3f}" if f1 else ""))
    print("\nSeed 0, paired difference with the ResNet (95% interval):")
    for k, v in seed0_diff.items():
        print(f"  {k:42s} {v['diff']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]")
    print("\nSeeds where CLIP beats the ResNet:", summary["seeds_where_clip_beats_resnet"])
    print("Cost:", json.dumps(cost, indent=1))


if __name__ == "__main__":
    main()
