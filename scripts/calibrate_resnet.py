"""Calibrate the fine-tuned ResNet-18 and measure what "I am not sure" buys.

Temperature scaling: one number T, fitted on the *validation* set, divides the logits. The test set is used
once, to report. Needs `reports/resnet18.pt` (written by `python scripts/train_resnet.py`); the split is the
README one (seed 0), so the test accuracy is the README's 74.6%.

Writes `reports/calibration.json` (read by `scripts/predict.py`) and `reports/figures/calibration.png`.
"""
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from museum_guide.calibration import (
    apply_temperature,
    confidence_threshold_for_accuracy,
    expected_calibration_error,
    fit_temperature,
    nll,
    reliability_bins,
    selective_accuracy,
)
from museum_guide.data import class_names, load_metadata, loaders, split
from museum_guide.predict import DEFAULT_WEIGHTS, SIZE, load_classifier
from museum_guide.train import REPORTS

COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.5)
TARGET_ACCURACIES = (0.80, 0.85, 0.90)


@torch.no_grad()
def logits_and_labels(model, loader):
    out, ys = [], []
    for x, y in loader:
        out.append(model(x))
        ys.append(y)
    return torch.cat(out).numpy(), torch.cat(ys).numpy()


def summary(logits, y, temperature):
    probs = apply_temperature(logits, temperature)
    conf, correct = probs.max(1), probs.argmax(1) == y
    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "nll": nll(logits, y, temperature),
        "ece": expected_calibration_error(probs, y),
        "mean_confidence_when_right": float(conf[correct].mean()),
        "mean_confidence_when_wrong": float(conf[~correct].mean()),
    }


def plot(bins_before, bins_after, curve, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, bins, title in ((axes[0], bins_before, "Before (T = 1)"), (axes[1], bins_after, "After temperature scaling")):
        centres = [(b["low"] + b["high"]) / 2 for b in bins]
        acc = [b["accuracy"] if b["accuracy"] is not None else 0 for b in bins]
        ax.bar(centres, acc, width=0.09, color="tab:blue", alpha=0.8, label="accuracy in bin")
        for b, c in zip(bins, centres):
            if b["n"]:
                ax.text(c, 0.02, str(b["n"]), ha="center", fontsize=7, color="white")
        ax.plot([0, 1], [0, 1], "k--", label="perfect calibration")
        ax.set(xlabel="confidence", ylabel="accuracy", title=title, xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left", fontsize=8)
    axes[2].plot([c["coverage"] for c in curve], [c["accuracy"] for c in curve], color="tab:green")
    axes[2].set(xlabel="coverage (share of photos answered)", ylabel="accuracy on answered photos",
                title="Answering only when sure", ylim=(0.6, 1.0))
    axes[2].invert_xaxis()
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    if not DEFAULT_WEIGHTS.exists():
        sys.exit(f"{DEFAULT_WEIGHTS} not found: train the model first with `python scripts/train_resnet.py`.")
    torch.set_num_threads(1)
    df = load_metadata()
    parts = split(df)
    n_classes = len(class_names(df))
    data = loaders(parts, SIZE, augment=False, batch_size=64)
    model = load_classifier(DEFAULT_WEIGHTS, n_classes)
    val_logits, val_y = logits_and_labels(model, data["val"])
    test_logits, test_y = logits_and_labels(model, data["test"])

    temperature = fit_temperature(val_logits, val_y)  # validation only
    before, after = summary(test_logits, test_y, 1.0), summary(test_logits, test_y, temperature)
    val_before, val_after = summary(val_logits, val_y, 1.0), summary(val_logits, val_y, temperature)

    test_probs = apply_temperature(test_logits, temperature)
    conf, correct = test_probs.max(1), test_probs.argmax(1) == test_y
    risk_coverage = selective_accuracy(conf, correct, COVERAGES)
    curve = selective_accuracy(conf, correct, np.linspace(1.0, 0.1, 37))

    # Threshold chosen on validation to reach a target accuracy, then applied once to the test set.
    val_probs = apply_temperature(val_logits, temperature)
    val_conf, val_correct = val_probs.max(1), val_probs.argmax(1) == val_y
    thresholds = {}
    for target in TARGET_ACCURACIES:
        t = confidence_threshold_for_accuracy(val_conf, val_correct, target)
        if t is None:
            thresholds[str(target)] = None
            continue
        answered = conf >= t
        thresholds[str(target)] = {
            "threshold": t,
            "val_coverage": float((val_conf >= t).mean()),
            "val_accuracy": float(val_correct[val_conf >= t].mean()),
            "test_coverage": float(answered.mean()),
            "test_accuracy": float(correct[answered].mean()) if answered.any() else None,
        }

    results = {
        "temperature": temperature,
        "fitted_on": "validation set",
        "n_fitted": int(len(val_y)),
        "n_test": int(len(test_y)),
        "split_seed": 0,
        "test_before": before,
        "test_after": after,
        "validation_before": val_before,
        "validation_after": val_after,
        "test_risk_coverage": risk_coverage,
        "threshold_chosen_on_validation": thresholds,
        "reliability_test_before": reliability_bins(apply_temperature(test_logits, 1.0), test_y),
        "reliability_test_after": reliability_bins(test_probs, test_y),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "calibration.json").write_text(json.dumps(results, indent=2))
    plot(results["reliability_test_before"], results["reliability_test_after"], curve, REPORTS / "figures" / "calibration.png")

    print(f"temperature T = {temperature:.2f} (fitted on {len(val_y)} validation images)")
    for name, s in (("before", before), ("after ", after)):
        print(f"test {name}: acc {s['accuracy']:.3f}  mean conf {s['mean_confidence']:.3f}  ECE {s['ece']:.3f}  NLL {s['nll']:.3f}")
    print("coverage -> accuracy on answered photos (test):")
    for r in risk_coverage:
        print(f"  {r['coverage']:.0%}: {r['accuracy']:.3f}")
    print("threshold chosen on validation -> test:", json.dumps(thresholds, indent=1))


if __name__ == "__main__":
    import argparse

    # This script takes no options. Parsing the command line gives `--help` (the docstring above) and rejects
    # typos, instead of silently running a long experiment.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
