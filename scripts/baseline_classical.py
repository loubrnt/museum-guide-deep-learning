"""Classical baseline: hand-made colour and edge features + scikit-learn models.

This is the "simpler model" the director asks about. If a network cannot beat it clearly,
it is not worth its cost on a kiosk.
"""
import time

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from museum_guide.data import DATA_DIR, class_names, load_metadata, split
from museum_guide.train import REPORTS, plot_confusion, save_results


def features(path) -> np.ndarray:
    img = Image.open(path).convert("RGB").resize((128, 128))
    hsv = np.asarray(img.convert("HSV"), dtype=np.float32) / 255
    colour = np.concatenate([np.histogram(hsv[..., c], bins=16, range=(0, 1), density=True)[0] for c in range(3)])
    grey = np.asarray(img.convert("L"), dtype=np.float32) / 255
    gy, gx = np.gradient(grey)
    magnitude, angle = np.hypot(gx, gy), np.arctan2(gy, gx) % np.pi
    edges = np.histogram(angle, bins=12, range=(0, np.pi), weights=magnitude)[0]
    edges = edges / (edges.sum() + 1e-8)
    thumb = np.asarray(img.convert("L").resize((8, 8)), dtype=np.float32).ravel() / 255  # coarse layout
    return np.concatenate([colour, edges, [magnitude.mean(), grey.std()], thumb])


def main():
    df = load_metadata()
    parts, classes = split(df), class_names(df)
    X = {k: np.stack([features(DATA_DIR / p) for p in v["path"]]) for k, v in parts.items()}
    y = {k: v["label_idx"].values for k, v in parts.items()}

    models = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=-1),
    }
    results = {"n_features": int(X["train"].shape[1])}
    for name, model in models.items():
        model.fit(X["train"], y["train"])
        val_acc = accuracy_score(y["val"], model.predict(X["val"]))
        pred = model.predict(X["test"])
        sample = DATA_DIR / parts["test"]["path"].iloc[0]
        if name == "random_forest":
            model.set_params(n_jobs=1)
        times = []
        for _ in range(30):
            t = time.perf_counter()
            model.predict(features(sample)[None])
            times.append((time.perf_counter() - t) * 1000)
        results[name] = {
            "val_acc": val_acc,
            "test_acc": accuracy_score(y["test"], pred),
            "test_macro_f1": f1_score(y["test"], pred, average="macro"),
            "cpu_latency_ms": float(np.median(times)),
        }
        print(name, results[name])
        plot_confusion(y["test"], pred, classes, f"Classical baseline ({name}) - test", REPORTS / "figures" / f"cm_{name}.png")
    save_results("baseline_classical", results)


if __name__ == "__main__":
    main()
