"""Fit, calibrate and save the linear head on CLIP image vectors: the recognition model of the guide.

CLIP's image encoder maps a painting to a 512-number vector; a logistic regression on these vectors
recognises the genre (90.2% over 10 seeds, `scripts/clip_classifier.py`). This script fits that head once,
on the TRAIN part of the README split (seed 0), choosing the regularisation C on the validation part, then:

- fits a temperature on the VALIDATION set (the same method as `scripts/calibrate_resnet.py`) and measures,
  once, on the TEST set: accuracy, calibration error, and the accuracy kept when only the most confident
  answers are given;
- chooses on the validation set the confidence above which the guide answers (`--target-accuracy`);
- saves the head, its temperature and its threshold in `reports/clip_probe.npz` (about 10 KB, versioned, so
  that `scripts/predict.py --model clip` and `scripts/guide.py` work without refitting) and the numbers in
  `reports/clip_probe.json`.

The 1,500 image vectors come from the cache `data/wikiart_genre5/clip_embeddings.npz` (computed by
`scripts/clip_classifier.py` or `scripts/search.py`; computed here, about 30 s, if missing).

    python scripts/train_clip_probe.py
"""
import argparse
import json

import numpy as np
from calibrate_resnet import COVERAGES, TARGET_ACCURACIES, plot, summary

from museum_guide.calibration import (
    apply_temperature,
    confidence_threshold_for_accuracy,
    fit_temperature,
    reliability_bins,
    selective_accuracy,
)
from museum_guide.clip import MODEL, cached_embeddings, embed_images, fit_linear_probe, load_clip
from museum_guide.data import DATA_DIR, class_names, load_metadata, split
from museum_guide.predict import DEFAULT_PROBE, ClipProbe, save_probe
from museum_guide.train import REPORTS

SEED = 0  # the split of the README


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target-accuracy", type=float, default=0.95,
                   help="the guide answers only above the confidence that reaches this accuracy on the validation set")
    p.add_argument("--out", default=DEFAULT_PROBE, help="where to save the head")
    args = p.parse_args()

    df = load_metadata()
    classes = tuple(class_names(df))
    paths = [DATA_DIR / q for q in df["path"]]

    def compute():
        model, processor = load_clip(MODEL)
        return embed_images(model, processor, paths)

    vectors = cached_embeddings(DATA_DIR / "clip_embeddings.npz", MODEL, paths, compute)
    y = df["label_idx"].values
    parts = split(df, SEED)
    idx = {k: df.index[df["id"].isin(v["id"])].to_numpy() for k, v in parts.items()}

    clf, C = fit_linear_probe(vectors[idx["train"]], y[idx["train"]], vectors[idx["val"]], y[idx["val"]])
    probe = ClipProbe(coef=clf.coef_, intercept=clf.intercept_, classes=classes, C=float(C), model=MODEL, split_seed=SEED)
    # The saved head must give the same answers as scikit-learn's own.
    assert np.allclose(probe.logits(vectors[idx["test"]]), clf.decision_function(vectors[idx["test"]]), atol=1e-4)

    val_logits, test_logits = probe.logits(vectors[idx["val"]]), probe.logits(vectors[idx["test"]])
    val_y, test_y = y[idx["val"]], y[idx["test"]]
    temperature = fit_temperature(val_logits, val_y)  # validation only
    before, after = summary(test_logits, test_y, 1.0), summary(test_logits, test_y, temperature)
    val_before, val_after = summary(val_logits, val_y, 1.0), summary(val_logits, val_y, temperature)

    test_probs = apply_temperature(test_logits, temperature)
    conf, correct = test_probs.max(1), test_probs.argmax(1) == test_y
    risk_coverage = selective_accuracy(conf, correct, COVERAGES)
    curve = selective_accuracy(conf, correct, np.linspace(1.0, 0.1, 37))

    val_probs = apply_temperature(val_logits, temperature)
    val_conf, val_correct = val_probs.max(1), val_probs.argmax(1) == val_y
    thresholds = {}
    for target in sorted({*TARGET_ACCURACIES, 0.95, args.target_accuracy}):
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
    chosen = thresholds[str(args.target_accuracy)]
    probe.temperature = temperature
    probe.min_confidence = None if chosen is None else chosen["threshold"]
    save_probe(probe, args.out)

    reference = None
    ref_file = REPORTS / "clip_classifier.json"
    if ref_file.exists():  # the seed-0 linear probe of scripts/clip_classifier.py must be reproduced
        seed0 = json.loads(ref_file.read_text())["per_seed"][0]
        reference = {"probe_C": seed0["probe_C"], "test_acc": seed0["test_acc"]["linear_probe"],
                     "reproduced": bool(seed0["probe_C"] == C and abs(seed0["test_acc"]["linear_probe"] - before["accuracy"]) < 1e-9)}

    results = {
        "model": MODEL,
        "classes": list(classes),
        "split_seed": SEED,
        "n_train": int(len(idx["train"])),
        "n_val": int(len(idx["val"])),
        "n_test": int(len(idx["test"])),
        "C": float(C),
        "val_accuracy": val_before["accuracy"],
        "test_accuracy": before["accuracy"],
        "temperature": temperature,
        "temperature_fitted_on": "validation set",
        "target_accuracy": args.target_accuracy,
        "min_confidence": probe.min_confidence,
        "test_before": before,
        "test_after": after,
        "validation_before": val_before,
        "validation_after": val_after,
        "test_risk_coverage": risk_coverage,
        "threshold_chosen_on_validation": thresholds,
        "reliability_test_before": reliability_bins(apply_temperature(test_logits, 1.0), test_y),
        "reliability_test_after": reliability_bins(test_probs, test_y),
        "clip_classifier_seed0_reference": reference,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "clip_probe.json").write_text(json.dumps(results, indent=2))
    plot(results["reliability_test_before"], results["reliability_test_after"], curve, REPORTS / "figures" / "clip_probe_calibration.png")

    print(f"C = {C}; validation accuracy {val_before['accuracy']:.3f}, test accuracy {before['accuracy']:.3f}"
          f" ({len(idx['train'])} train / {len(idx['val'])} val / {len(idx['test'])} test images)")
    if reference:
        print(f"seed-0 reference in reports/clip_classifier.json: C={reference['probe_C']}, test {reference['test_acc']:.3f}"
              f" -> reproduced: {reference['reproduced']}")
    print(f"temperature T = {temperature:.2f} (fitted on {len(val_y)} validation images)")
    for name, s in (("before", before), ("after ", after)):
        print(f"test {name}: acc {s['accuracy']:.3f}  mean conf {s['mean_confidence']:.3f}  ECE {s['ece']:.3f}  NLL {s['nll']:.3f}")
    print("coverage -> accuracy on answered photos (test):")
    for r in risk_coverage:
        print(f"  {r['coverage']:.0%}: {r['accuracy']:.3f}")
    print("threshold chosen on validation -> test:", json.dumps(thresholds, indent=1))
    print(f"saved {args.out} (min_confidence = {probe.min_confidence})")


if __name__ == "__main__":
    main()
