"""Beyond accuracy: per-class metrics and error analysis for the four models of the README.

For the split of seed 0 (the one of the README table), the four models are retrained exactly as in their own
scripts (`baseline_classical.py`, `train_mlp.py`, `train_resnet.py`), so the test accuracies are the README ones.
For each model: precision / recall / F1 per genre and macro-F1, accuracy for "Unknown Artist" vs named artists,
and accuracy per painting style. For the ResNet: the most confident mistakes and the least confident correct
predictions, as images.

    python scripts/error_analysis.py

Writes `reports/error_analysis.json` and `reports/figures/error_*.png`.
"""
import json

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

from museum_guide.data import DATA_DIR, class_names, device, load_metadata, loaders, split
from museum_guide.metrics import (
    accuracy_by_group,
    least_confident_correct,
    most_confident_errors,
    per_class_report,
)
from museum_guide.models import FeedForward, pretrained_resnet18
from museum_guide.train import REPORTS, fit

FIG = REPORTS / "figures"
MIN_STYLE_IMAGES = 10  # styles with fewer test images are too noisy to report


@torch.no_grad()
def predict_probs(model, loader, dev):
    model.eval().to(dev)
    return torch.cat([model(x.to(dev)).softmax(1).cpu() for x, _ in loader]).numpy()


def train_models(parts, classes, dev):
    """Returns {model name: (predictions on test, probabilities on test or None)} for the split `parts`."""
    y_train = parts["train"]["label_idx"].values
    out = {}

    # Classical models, as in `baseline_classical.py`.
    X = {k: np.stack([features(DATA_DIR / p) for p in v["path"]]) for k, v in parts.items()}
    classical = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
    }
    for name, model in classical.items():
        model.fit(X["train"], y_train)
        out[name] = (model.predict(X["test"]), model.predict_proba(X["test"]))

    # Feed-forward network, final configuration of `train_mlp.py`.
    plain = loaders(parts, MLP_SIZE, augment=False)
    torch.manual_seed(0)
    mlp = FeedForward(3 * MLP_SIZE * MLP_SIZE, HIDDEN, len(classes), dropout=0.5)
    fit(mlp, plain, dev, epochs=60, lr=1e-4, weight_decay=1e-4, verbose=False)
    probs = predict_probs(mlp, plain["test"], dev)
    out["mlp"] = (probs.argmax(1), probs)

    # Fine-tuned ResNet-18, as in `train_resnet.py`.
    data = loaders(parts, RESNET_SIZE, augment=True, batch_size=32)
    torch.manual_seed(0)
    resnet = pretrained_resnet18(len(classes), train_layer4=True)
    fit(resnet, data, dev, epochs=20, lr=3e-4, weight_decay=1e-4, patience=8, verbose=False)
    probs = predict_probs(resnet, data["test"], dev)
    out["resnet18_layer4"] = (probs.argmax(1), probs)
    return out


def gallery(test, idx, probs, classes, title, path):
    """Grid of test images with their true genre, predicted genre and confidence."""
    cols = 4
    rows = int(np.ceil(len(idx) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), idx):
        row = test.iloc[i]
        ax.imshow(plt.imread(DATA_DIR / row["path"]))
        pred = probs[i].argmax()
        ax.set_title(f"true: {row['label']}\npred: {classes[pred]} ({probs[i].max():.0%})", fontsize=8.5)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    dev, df = device(), load_metadata()
    parts, classes = split(df, 0), class_names(df)
    test = parts["test"]
    y = test["label_idx"].values
    artist = np.where(test["artist"] == "Unknown Artist", "unknown artist", "named artist")
    style = test["style"].values

    models = train_models(parts, classes, dev)
    results = {"seed": 0, "n_test": len(y), "classes": classes, "models": {}}
    for name, (pred, _) in models.items():
        results["models"][name] = {
            **per_class_report(y, pred, classes),
            "accuracy_by_artist_type": accuracy_by_group(y, pred, artist),
            "accuracy_by_style": accuracy_by_group(y, pred, style, min_count=MIN_STYLE_IMAGES),
        }
        r = results["models"][name]
        print(f"{name:20s} acc {r['accuracy']:.3f}  macro-F1 {r['macro_f1']:.3f}  "
              + "  ".join(f"{k} {v['acc']:.3f} (n={v['n']})" for k, v in r["accuracy_by_artist_type"].items()), flush=True)

    # Image galleries for the ResNet.
    probs = models["resnet18_layer4"][1]
    wrong = most_confident_errors(probs, y, k=12)
    right = least_confident_correct(probs, y, k=12)
    FIG.mkdir(parents=True, exist_ok=True)
    gallery(test, wrong, probs, classes, "ResNet-18: the 12 most confident mistakes (test, seed 0)", FIG / "error_confident_mistakes.png")
    gallery(test, right, probs, classes, "ResNet-18: the 12 least confident correct predictions (test, seed 0)", FIG / "error_hesitant_correct.png")
    results["resnet18_layer4"] = {
        "most_confident_mistakes": [
            {"id": test["id"].iloc[i], "artist": test["artist"].iloc[i], "style": test["style"].iloc[i],
             "true": classes[y[i]], "pred": classes[probs[i].argmax()], "confidence": float(probs[i].max())}
            for i in wrong
        ],
        "mean_confidence_wrong": float(probs.max(1)[probs.argmax(1) != y].mean()),
        "mean_confidence_right": float(probs.max(1)[probs.argmax(1) == y].mean()),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "error_analysis.json").write_text(json.dumps(results, indent=2))

    print("\nResNet-18 per class (test, seed 0):")
    for c, v in results["models"]["resnet18_layer4"]["per_class"].items():
        print(f"  {c:20s} P {v['precision']:.3f}  R {v['recall']:.3f}  F1 {v['f1']:.3f}  n={v['support']}")


if __name__ == "__main__":
    import argparse

    # This script takes no options. Parsing the command line gives `--help` (the docstring above) and rejects
    # typos, instead of silently running a long experiment.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
