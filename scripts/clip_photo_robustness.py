"""Does the recommended model (CLIP image encoder + linear head) survive camera-like conditions?

`photo_robustness.py` measured the fine-tuned ResNet-18 only. This script measures the CLIP head on the SAME
seeds, the SAME splits and the SAME perturbations (nine deterministic ones, `museum_guide.perturb`), plus the
circular shift of `robustness.py` (6.25% and 12.5% of the width).

Method (vectors are computed once per condition and cached under `data/wikiart_genre5/clip_photo/`):

  - every image of the collection is perturbed (at its 128 px size, as for the ResNet) and embedded by CLIP;
    the shift is applied to the 224 x 224 tensor CLIP actually sees;
  - for each seed, a "standard" probe is trained on CLEAN train vectors, as deployed (`train_clip_probe.py`):
    C on the clean validation set, temperature on the clean validation set, answer threshold chosen on the clean
    validation set for 95% accuracy; it is then tested on the perturbed TEST vectors;
  - an "augmented" probe is trained on clean + photo-augmented train vectors (`--aug-draws` random draws of
    `perturb.photo_augment_pil` per train image; the encoder stays frozen), everything else identical.

For the kiosk it also reports how often the head answers (instead of "unsure") on each condition and how
accurate the answers are: a good head should lose confidence on a bad photo.

    python scripts/clip_photo_robustness.py --seeds 5

Writes `reports/clip_photo_robustness.json` and `reports/figures/clip_photo_robustness.png`.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from museum_guide.calibration import apply_temperature, confidence_threshold_for_accuracy, fit_temperature
from museum_guide.clip import MODEL, cached_embeddings, embed_pil_images, fit_linear_probe, load_clip
from museum_guide.clip_robustness import (
    answer_stats,
    change_rate,
    embed_shifted,
    mean_over_seeds,
    paired_summary,
)
from museum_guide.data import DATA_DIR, load_metadata, split
from museum_guide.perturb import PERTURBATIONS, photo_augment_pil
from museum_guide.train import REPORTS

SIZE = 128  # size of the source images and of the augmentation output, as for the ResNet
SHIFTS = {"shift_6.25pct": 0.0625, "shift_12.5pct": 0.125}
CONDITIONS = ("clean", *PERTURBATIONS)  # the nine perturbations, comparable with the ResNet
ALL_CONDITIONS = (*CONDITIONS, *SHIFTS)
VARIANTS = ("standard", "augmented")
TARGET_ACCURACY = 0.95  # same rule as train_clip_probe.py
CACHE_VERSION = "v1"


class Collection:
    """The 1500 images, and the cached CLIP vectors of each condition (CLIP is loaded only if a cache is missing)."""

    def __init__(self, df, cache_dir: Path):
        self.paths = [DATA_DIR / p for p in df["path"]]
        self.cache_dir = cache_dir
        self._clip = None

    def clip(self):
        if self._clip is None:
            self._clip = load_clip()
        return self._clip

    def images(self):
        for p in self.paths:
            with Image.open(p) as img:
                yield img.convert("RGB")

    def vectors(self, condition: str, compute) -> np.ndarray:
        name = f"{MODEL}|{condition}|{CACHE_VERSION}"
        return cached_embeddings(self.cache_dir / f"{condition}.npz", name, self.paths, compute)

    def clean(self):
        # The clean vectors are the ones every other script uses (data/wikiart_genre5/clip_embeddings.npz).
        def compute():
            model, processor = self.clip()
            return embed_pil_images(model, processor, list(self.images()))

        return cached_embeddings(DATA_DIR / "clip_embeddings.npz", MODEL, self.paths, compute)

    def perturbed(self, name: str):
        def compute():
            model, processor = self.clip()
            return embed_pil_images(model, processor, [PERTURBATIONS[name](i) for i in self.images()])

        return self.vectors(name, compute)

    def shifted(self, name: str):
        def compute():
            model, processor = self.clip()
            return embed_shifted(model, processor, list(self.images()), SHIFTS[name])

        return self.vectors(name, compute)

    def augmented(self, draw: int):
        def compute():
            model, processor = self.clip()
            torch.manual_seed(1000 + draw)  # the same draw for every seed: the train rows are just a subset
            augment = photo_augment_pil(SIZE)
            return embed_pil_images(model, processor, [augment(i) for i in self.images()])

        return self.vectors(f"aug_{draw}", compute)


def fit_head(X_train, y_train, X_val, y_val):
    """Probe, temperature and answer threshold, all chosen on the clean validation set."""
    clf, C = fit_linear_probe(X_train, y_train, X_val, y_val)
    val_logits = clf.decision_function(X_val)
    temperature = fit_temperature(val_logits, y_val)
    val_probs = apply_temperature(val_logits, temperature)
    threshold = confidence_threshold_for_accuracy(val_probs.max(1), val_probs.argmax(1) == y_val, TARGET_ACCURACY)
    return {"clf": clf, "C": C, "temperature": temperature, "threshold": threshold}


def evaluate(head, vectors, y_test, clean_pred):
    probs = apply_temperature(head["clf"].decision_function(vectors), head["temperature"])
    pred, conf = probs.argmax(1), probs.max(1)
    return {
        "accuracy": float((pred == y_test).mean()),
        "prediction_change_rate": change_rate(pred, clean_pred),
        **answer_stats(conf, pred == y_test, head["threshold"]),
    }, pred


def run_seed(seed, df, V, y, aug_draws):
    parts = split(df, seed)
    idx = {k: df.index[df["id"].isin(v["id"])].to_numpy() for k, v in parts.items()}
    tr, va, te = idx["train"], idx["val"], idx["test"]
    X_aug = np.concatenate([V["clean"][tr]] + [V[f"aug_{k}"][tr] for k in range(aug_draws)])
    y_aug = np.tile(y[tr], 1 + aug_draws)
    heads = {
        "standard": fit_head(V["clean"][tr], y[tr], V["clean"][va], y[va]),
        "augmented": fit_head(X_aug, y_aug, V["clean"][va], y[va]),
    }
    row = {"seed": seed, "n_test": int(len(te))}
    for variant, head in heads.items():
        clean_pred = head["clf"].predict(V["clean"][te])
        row[variant] = {"C": head["C"], "temperature": float(head["temperature"]), "threshold": head["threshold"]}
        for cond in ALL_CONDITIONS:
            row[variant][cond], _ = evaluate(head, V[cond][te], y[te], clean_pred)
    return row


def summarise(per_seed, resnet):
    acc = {v: {c: [s[v][c]["accuracy"] for s in per_seed] for c in ALL_CONDITIONS} for v in VARIANTS}
    perturbed_mean = {v: [float(np.mean([s[v][c]["accuracy"] for c in PERTURBATIONS])) for s in per_seed] for v in VARIANTS}
    out = {"accuracy": {}, "augmented_minus_standard": {}, "drop_from_clean": {}, "kiosk_behaviour": {}}
    for v in VARIANTS:
        rows = [{c: s[v][c]["accuracy"] for c in ALL_CONDITIONS} for s in per_seed]
        out["accuracy"][v] = mean_over_seeds(rows, ALL_CONDITIONS)
        out["accuracy"][v]["mean_over_perturbations"] = mean_over_seeds(
            [{"m": x} for x in perturbed_mean[v]], ["m"]
        )["m"]
        out["drop_from_clean"][v] = {
            c: float(np.mean(np.array(acc[v][c]) - np.array(acc[v]["clean"]))) for c in ALL_CONDITIONS if c != "clean"
        }
        out["kiosk_behaviour"][v] = {
            c: {
                k: float(np.mean([s[v][c][k] for s in per_seed]))
                for k in ("mean_confidence", "answered_share", "accuracy_when_answered", "prediction_change_rate")
            }
            for c in ALL_CONDITIONS
        }
    out["augmented_minus_standard"] = {c: paired_summary(acc["augmented"][c], acc["standard"][c]) for c in ALL_CONDITIONS}
    out["augmented_minus_standard"]["mean_over_perturbations"] = paired_summary(
        perturbed_mean["augmented"], perturbed_mean["standard"]
    )
    if resnet is not None:  # same seeds, same splits: paired comparison of the two recommended-model candidates
        r_std = {c: [s["standard"][c] for s in resnet["per_seed"]] for c in CONDITIONS}
        r_perturbed = [float(np.mean([s["standard"][c] for c in PERTURBATIONS])) for s in resnet["per_seed"]]
        out["clip_standard_minus_resnet_standard"] = {
            c: paired_summary(acc["standard"][c], r_std[c]) for c in CONDITIONS
        }
        out["clip_standard_minus_resnet_standard"]["mean_over_perturbations"] = paired_summary(
            perturbed_mean["standard"], r_perturbed
        )
    return out


def plot(summary, resnet, n_seeds, path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    series = [("CLIP head, standard", summary["accuracy"]["standard"], "tab:blue"),
              ("CLIP head, photo-augmented", summary["accuracy"]["augmented"], "tab:cyan")]
    if resnet is not None:
        series = [("ResNet-18, standard", resnet["accuracy"]["standard"], "tab:red"),
                  ("ResNet-18, photo-augmented", resnet["accuracy"]["photo"], "tab:orange")] + series
    x, w = np.arange(len(CONDITIONS)), 0.8 / len(series)
    for k, (label, stats, colour) in enumerate(series):
        ax.bar(x + (k - (len(series) - 1) / 2) * w, [stats[c]["mean"] for c in CONDITIONS], w,
               yerr=[stats[c]["std"] for c in CONDITIONS], color=colour, label=label, capsize=1.5)
    ax.set_xticks(x, [c.replace("_", "\n") for c in CONDITIONS], fontsize=8)
    ax.set_ylabel("test accuracy")
    ax.set_title(f"Genre accuracy under camera-like perturbations (mean ± std, {n_seeds} seeds)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=8, frameon=False)

    ax = axes[1]
    kb = summary["kiosk_behaviour"]["standard"]
    conds = ("clean", "rotation_15deg", "blur", "perspective", "darker", "shift_12.5pct")
    x = np.arange(len(conds))
    ax.bar(x - 0.2, [kb[c]["answered_share"] for c in conds], 0.4, label="share answered", color="tab:green")
    ax.bar(x + 0.2, [kb[c]["accuracy_when_answered"] for c in conds], 0.4, label="accuracy when answered", color="tab:purple")
    ax.set_xticks(x, [c.replace("_", "\n") for c in conds], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("CLIP head with its 'unsure' threshold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seeds", type=int, default=5, help="number of seeds (0 .. N-1), as in photo_robustness.py")
    p.add_argument("--aug-draws", type=int, default=2, help="photo-augmented copies of each train image")
    p.add_argument("--cache-dir", type=Path, default=DATA_DIR / "clip_photo", help="where the vectors are cached")
    args = p.parse_args()

    df = load_metadata()
    y = df["label_idx"].values
    coll = Collection(df, args.cache_dir)
    t = time.time()
    V = {"clean": coll.clean()}
    for name in PERTURBATIONS:
        V[name] = coll.perturbed(name)
    for name in SHIFTS:
        V[name] = coll.shifted(name)
    for k in range(args.aug_draws):
        V[f"aug_{k}"] = coll.augmented(k)
    print(f"vectors ready ({time.time() - t:.0f}s): {len(V)} sets of {len(df)} images", flush=True)

    per_seed = [run_seed(seed, df, V, y, args.aug_draws) for seed in range(args.seeds)]
    for row in per_seed:
        print(f"seed {row['seed']}: clean {row['standard']['clean']['accuracy']:.3f}  "
              f"mean perturbed {np.mean([row['standard'][c]['accuracy'] for c in PERTURBATIONS]):.3f} -> "
              f"{np.mean([row['augmented'][c]['accuracy'] for c in PERTURBATIONS]):.3f} (augmented)", flush=True)

    ref = REPORTS / "photo_robustness.json"
    resnet = json.loads(ref.read_text()) if ref.exists() else None
    if resnet is not None:
        n_resnet = [s["n_test"] for s in resnet["per_seed"][: args.seeds]]
        assert n_resnet == [r["n_test"] for r in per_seed[: len(n_resnet)]], "different test sets than the ResNet study"

    # Cross-check with the deployed head (seed 0 split): the standard probe must reproduce its test accuracy.
    check = {}
    deployed = REPORTS / "clip_probe.json"
    if deployed.exists():
        d = json.loads(deployed.read_text())
        mine = per_seed[0]["standard"]["clean"]["accuracy"]
        check = {"deployed_test_accuracy": d["test_accuracy"], "this_script_seed0": mine,
                 "same": bool(abs(mine - d["test_accuracy"]) < 1e-9)}

    summary = summarise(per_seed, resnet)
    results = {
        "n_seeds": args.seeds,
        "aug_draws": args.aug_draws,
        "model": MODEL,
        "conventions": {
            "perturbations": "applied to the 128 px image, then CLIP's own resize/crop to 224 (as the ResNet study applied them at 128 px)",
            "shift": "circular shift of the preprocessed 224 x 224 tensor, down and right, by 6.25% (14 px) or 12.5% (28 px) of the width",
            "standard_probe": "trained on clean vectors; C, temperature and the 95%-accuracy answer threshold chosen on the clean validation set",
            "augmented_probe": f"trained on clean + {args.aug_draws} photo-augmented draws of each train image (encoder frozen)",
            "answered": "top probability (after temperature) >= the validation threshold",
        },
        "seed0_check_against_clip_probe_json": check,
        "per_seed": per_seed,
        **summary,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "clip_photo_robustness.json").write_text(json.dumps(results, indent=2))
    plot(summary, resnet, args.seeds, REPORTS / "figures" / "clip_photo_robustness.png")

    print(f"\nTest accuracy, mean over {args.seeds} seeds (CLIP standard -> augmented | ResNet standard):")
    for c in ALL_CONDITIONS:
        a, b = summary["accuracy"]["standard"][c], summary["accuracy"]["augmented"][c]
        r = f"{resnet['accuracy']['standard'][c]['mean']:.3f}" if resnet is not None and c in CONDITIONS else "  -  "
        print(f"  {c:16s} {a['mean']:.3f} ± {a['std']:.3f} -> {b['mean']:.3f} ± {b['std']:.3f} | {r}")


if __name__ == "__main__":
    main()
