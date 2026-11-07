"""Can we beat CLIP's frozen linear probe by adapting the encoder? Richer features, partial fine-tuning, LoRA.

The proposed model is CLIP's image encoder, frozen, with a linear head (90.2% genre, 55.4% style over 10 seeds). The
style task has the most headroom, so this compares, on the same artist-disjoint splits:

- `probe`: the baseline, logistic regression on CLIP's normalised 512-d vectors (C chosen on validation);
- `frozen_last_layer`: standardised logistic regression on the 768-d class token before the projection (a control);
- `frozen_last4`: the same on the class tokens of the last 4 blocks concatenated (3072-d), lower layers keep more
  colour and brushwork information;
- `finetune_N`: the last N transformer blocks, the final layer norm and the projection are trained together with a
  linear head warm-started from the probe (so epoch 0 IS the probe); N = 1 and 2; AdamW, 1e-5 for the encoder,
  1e-4 for the head, weight decay 0.01, RandomResizedCrop + flip, at most 6 epochs, early stopping on VALIDATION accuracy;
- `lora_last4`: LoRA (rank 8) on the attention queries and values of the last 4 blocks, plus the warm-started head.

Settings are fixed in advance, not searched; the validation set picks the epoch (and C for the probes); the test set is
read once per run, at the chosen epoch.

    python scripts/clip_adapt.py                       # style task (data/wikiart_style6), 5 seeds
    python scripts/clip_adapt.py --task genre          # genre task (data/wikiart_genre5)

Needs the datasets (README) and CLIP's weights. Writes `reports/clip_adapt_<task>.json` and a figure. Uses the Apple GPU
(MPS) or the CPU, whichever `museum_guide.data.device()` finds.
"""
import argparse
import copy
import json
import os
import time

_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--task", choices=("style", "genre"), default="style")
_TASK = _pre.parse_known_args()[0].task
os.environ["MUSEUM_GUIDE_DATA"] = {"style": "data/wikiart_style6", "genre": "data/wikiart_genre5"}[_TASK]  # before museum_guide.data

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from museum_guide.clip import MODEL, cached_embeddings, embed_images, fit_linear_probe, load_clip
from museum_guide.clip_adapt import (
    ClipClassifier,
    MultiLayerFeatures,
    add_lora,
    clip_transforms,
    count_parameters,
    fit_adapt,
    fit_standardised_probe,
    freeze,
    last_blocks,
    model_size_mb,
    paired_differences,
    parameter_groups,
    unfreeze,
    warm_start_head,
)
from museum_guide.data import DATA_DIR, WikiArt, class_names, device, load_metadata, split
from museum_guide.robustness import mean_std
from museum_guide.train import REPORTS, predict

LR_ENCODER, LR_HEAD, LR_LORA, WEIGHT_DECAY = 1e-5, 1e-4, 3e-4, 0.01
BATCH = 32
VARIANTS = ("probe", "frozen_last_layer", "frozen_last4", "finetune_1", "finetune_2", "lora_last4")


def extract_last4(base_model, paths, dev, batch=64):
    """Class tokens of the last 4 blocks (through the final layer norm), concatenated: one 3072-d row per image."""
    from PIL import Image

    tower = MultiLayerFeatures(base_model.vision_model, base_model.vision_model.post_layernorm, 4).to(dev).eval()
    transform, out = clip_transforms(augment=False), []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            images = []
            for p in paths[i : i + batch]:
                with Image.open(p) as im:
                    images.append(transform(im.convert("RGB")))
            out.append(tower(torch.stack(images).to(dev)).cpu().numpy())
    tower.cpu()
    return np.concatenate(out)


def make_loaders(parts, workers=0):
    sets = {k: WikiArt(v, clip_transforms(augment=(k == "train"))) for k, v in parts.items()}
    return {k: DataLoader(ds, batch_size=BATCH, shuffle=(k == "train"), num_workers=workers) for k, ds in sets.items()}


def run_adaptation(variant, base_model, probe, loaders, n_classes, dev, seed, epochs, patience):
    """Train one adapted model; returns the measurements (the test set is read once, after the best epoch is restored)."""
    model = copy.deepcopy(base_model)
    vision, proj = model.vision_model, model.visual_projection
    head = nn.Linear(proj.out_features, n_classes)
    warm_start_head(head, probe)
    classifier = ClipClassifier(vision, proj, head)
    freeze(classifier)
    if variant.startswith("finetune_"):
        n = int(variant.split("_")[1])
        backbone = [*last_blocks(vision.encoder.layers, n), vision.post_layernorm, proj]
        unfreeze([*backbone, head])
        groups = parameter_groups([(backbone, LR_ENCODER), ([head], LR_HEAD)], WEIGHT_DECAY)
        shipped = count_parameters(backbone)  # what a kiosk would have to ship besides the frozen rest, plus the head
    else:  # lora_last4
        unfreeze([head])
        lora = add_lora(last_blocks(vision.encoder.layers, 4), r=8, alpha=16.0)
        groups = parameter_groups([(lora, LR_LORA), ([head], LR_HEAD)], WEIGHT_DECAY)
        shipped = count_parameters(lora)
    trainable = count_parameters(classifier)
    torch.manual_seed(seed)
    start = time.perf_counter()
    hist = fit_adapt(classifier, loaders, dev, groups, epochs=epochs, patience=patience)
    seconds = time.perf_counter() - start
    y, pred = predict(classifier, loaders["test"], dev)  # the single read of the test set
    return {
        "test_acc": float((y == pred).mean()),
        "test_macro_f1": float(f1_score(y, pred, average="macro")),
        "best_val_acc": hist["best_val_acc"],
        "epoch0_val_acc": hist["val_acc"][0],
        "best_epoch": hist["best_epoch"],
        "epochs_run": hist["epochs_run"],
        "trainable_params": trainable,
        "shipped_params_besides_head": shipped,
        "shipped_mb_besides_head_fp32": model_size_mb(shipped),
        "train_seconds": seconds,
        "seconds_per_epoch": float(np.mean(hist["epoch_seconds"])) if hist["epoch_seconds"] else 0.0,
        "val_acc_curve": hist["val_acc"],  # index 0 = the starting model (the probe)
        "train_loss_curve": hist["train_loss"],
    }


def plot(per_seed, summary, path, task):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    names = [v for v in VARIANTS if v in summary]
    for i, v in enumerate(names):
        accs = [s["variants"][v]["test_acc"] for s in per_seed]
        ax.scatter([i] * len(accs), accs, color="tab:blue", alpha=0.6, zorder=3, label="one seed" if i == 0 else None)
        ax.hlines(summary[v]["test_acc"]["mean"], i - 0.3, i + 0.3, color="black", zorder=4, label="mean" if i == 0 else None)
    ax.set_xticks(range(len(names)), [n.replace("_", "\n") for n in names], fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"CLIP adaptation, {task} task, {len(per_seed)} seeds (artist-disjoint splits)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    global LR_ENCODER, LR_HEAD, LR_LORA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=("style", "genre"), default="style")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    ap.add_argument("--lr-encoder", type=float, default=LR_ENCODER)
    ap.add_argument("--lr-head", type=float, default=LR_HEAD)
    ap.add_argument("--lr-lora", type=float, default=LR_LORA)
    ap.add_argument("--tag", default="", help="suffix of the output files, e.g. _lr3e-5")
    ap.add_argument("--replot", action="store_true", help="only redraw the figure from the saved JSON (no training)")
    args = ap.parse_args()
    if args.replot:
        saved = json.loads((REPORTS / f"clip_adapt_{args.task}{args.tag}.json").read_text())
        plot(saved["per_seed"], saved["summary"], REPORTS / "figures" / f"clip_adapt_{args.task}{args.tag}.png", args.task)
        return
    LR_ENCODER, LR_HEAD, LR_LORA = args.lr_encoder, args.lr_head, args.lr_lora
    torch.set_num_threads(4)

    dev, df = device(), load_metadata()
    classes = class_names(df)
    n_classes = len(classes)
    paths = [DATA_DIR / q for q in df["path"]]
    y_all = df["label_idx"].to_numpy()
    t0 = time.time()
    base_model, processor = load_clip()
    clip_emb = cached_embeddings(DATA_DIR / "clip_embeddings.npz", MODEL, paths, lambda: embed_images(base_model, processor, paths))
    last4 = cached_embeddings(DATA_DIR / "clip_last4.npz", MODEL + "|last4-cls-postln", paths, lambda: extract_last4(base_model, paths, dev))
    print(f"{args.task}: {len(df)} images, {n_classes} classes, features ready ({time.time() - t0:.0f}s), device {dev}", flush=True)

    pos = pd.Series(np.arange(len(df)), index=df["id"].to_numpy())
    per_seed = []
    for seed in range(args.seeds):
        parts = split(df, seed)
        idx = {k: pos[v["id"]].to_numpy() for k, v in parts.items()}
        y = {k: y_all[i] for k, i in idx.items()}
        probe, c_probe = fit_linear_probe(clip_emb[idx["train"]], y["train"], clip_emb[idx["val"]], y["val"])
        out = {}
        if "probe" in args.variants:
            pred = probe.predict(clip_emb[idx["test"]])
            out["probe"] = {"test_acc": float((pred == y["test"]).mean()), "test_macro_f1": float(f1_score(y["test"], pred, average="macro")),
                            "val_acc": float(probe.score(clip_emb[idx["val"]], y["val"])), "C": c_probe}
        for name, feats in (("frozen_last_layer", last4[:, -768:]), ("frozen_last4", last4)):
            if name in args.variants:
                clf, c = fit_standardised_probe(feats[idx["train"]], y["train"], feats[idx["val"]], y["val"])
                pred = clf.predict(feats[idx["test"]])
                out[name] = {"test_acc": float((pred == y["test"]).mean()), "test_macro_f1": float(f1_score(y["test"], pred, average="macro")),
                             "val_acc": float(clf.score(feats[idx["val"]], y["val"])), "C": c, "dim": int(feats.shape[1])}
        loaders = make_loaders(parts)
        for variant in (v for v in args.variants if v.startswith(("finetune_", "lora_"))):
            out[variant] = run_adaptation(variant, base_model, probe, loaders, n_classes, dev, seed, args.epochs, args.patience)
            r = out[variant]
            print(f"  seed {seed} {variant}: test {r['test_acc']:.3f} (val {r['best_val_acc']:.3f}, epoch {r['best_epoch']}/{r['epochs_run']}, "
                  f"{r['train_seconds']:.0f}s)", flush=True)
        per_seed.append({"seed": seed, "n_test": len(y["test"]), "variants": out})
        print(f"seed {seed}: " + "  ".join(f"{v} {r['test_acc']:.3f}" for v, r in out.items()), flush=True)

    names = [v for v in VARIANTS if v in per_seed[0]["variants"]]
    summary = {v: {"test_acc": dict(zip(("mean", "std"), mean_std([s["variants"][v]["test_acc"] for s in per_seed]))),
                   "test_macro_f1": dict(zip(("mean", "std"), mean_std([s["variants"][v]["test_macro_f1"] for s in per_seed])))}
               for v in names}
    for v in names:
        if v.startswith(("finetune_", "lora_")):
            runs = [s["variants"][v] for s in per_seed]
            summary[v].update({
                "trainable_params": runs[0]["trainable_params"],
                "shipped_mb_besides_head_fp32": runs[0]["shipped_mb_besides_head_fp32"],
                "mean_train_seconds": float(np.mean([r["train_seconds"] for r in runs])),
                "seeds_where_fine_tuning_changed_the_model": int(sum(r["best_epoch"] > 0 for r in runs)),
                "mean_best_epoch": float(np.mean([r["best_epoch"] for r in runs])),
                "epoch0_val_acc_mean": float(np.mean([r["epoch0_val_acc"] for r in runs])),
            })
    paired = {f"{v}_minus_probe": paired_differences([s["variants"][v]["test_acc"] for s in per_seed], [s["variants"]["probe"]["test_acc"] for s in per_seed])
              for v in names if v != "probe"} if "probe" in names else {}
    results = {
        "task": args.task, "data_dir": str(DATA_DIR), "classes": classes, "n_images": len(df), "n_seeds": args.seeds,
        "settings": {"lr_encoder": args.lr_encoder, "lr_head": args.lr_head, "lr_lora": args.lr_lora, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH,
                     "max_epochs": args.epochs, "patience": args.patience, "device": str(dev),
                     "chosen": "fixed in advance; the epoch (and C for the probes) is chosen on validation only; the test set is read once per run"},
        "chance": 1 / n_classes, "per_seed": per_seed, "summary": summary, "paired_vs_probe": paired,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / f"clip_adapt_{args.task}{args.tag}.json").write_text(json.dumps(results, indent=2))
    plot(per_seed, summary, REPORTS / "figures" / f"clip_adapt_{args.task}{args.tag}.png", args.task)

    print("\nTest accuracy (mean +- std over seeds), paired difference with the probe:")
    for v in names:
        s = summary[v]["test_acc"]
        p = paired.get(f"{v}_minus_probe")
        extra = f"   {p['mean'] * 100:+.1f} pts, higher on {p['seeds_a_higher']}/{p['n_seeds']} seeds" if p else ""
        print(f"  {v:18s} {s['mean'] * 100:5.1f} +- {s['std'] * 100:.1f}{extra}")


if __name__ == "__main__":
    main()
