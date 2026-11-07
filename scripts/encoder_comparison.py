"""Which image encoder? Accuracy against cost, on the same artist-disjoint splits.

CLIP ViT-B/32 + a linear head is the proposed recognition model. Is another FROZEN encoder a better trade-off?
For each encoder (CLIP B/32, B/16, L/14, SigLIP base, DINOv2 small and base, ImageNet ResNet-50 and ResNet-18, see
`museum_guide.encoders` for the conventions, fixed in advance) this script:

- embeds the 1500 genre images (`data/wikiart_genre5`) and the 1800 style images (`data/wikiart_style6`, PR #15);
- trains a logistic-regression probe on the frozen vectors for the same 10 splits `split(df, seed)` (C chosen on the
  validation set among 0.1, 1, 10, 100, 1000, the same grid for every encoder), reports accuracy and macro-F1 and, versus
  CLIP B/32, on how many seeds each encoder wins;
- for the encoders with a text tower, the zero-shot accuracy ("a painting of a {genre}" / "a painting in the style of
  {style}", never tuned on the test sets);
- measures the cost of the image tower: parameters, weight file, and the latency of one image on ONE thread, in a fresh
  process per encoder (3 repeats), with random weights of the right architecture (speed does not depend on the values).

    python scripts/encoder_comparison.py                  # everything (embeddings, probes, latency)
    python scripts/encoder_comparison.py --stage latency  # only the timings (leave the machine idle)
    python scripts/encoder_comparison.py --stage accuracy # only the probes

The source images are 128 px and are upsampled to each encoder's input size. Embeddings are cached under `data/`.
Writes `reports/encoder_comparison.json` (each stage updates its part) and `reports/figures/encoder_comparison.png`.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

from museum_guide.benchmark import n_params, serialized_mb, summarize_ms
from museum_guide.clip import MODEL as CLIP_B32, cached_embeddings, class_prompts, fit_linear_probe, zero_shot_predict
from museum_guide.data import class_names, device, load_metadata, split
from museum_guide.encoders import REGISTRY, image_tower, load_encoder, paired_wins
from museum_guide.robustness import mean_std
from museum_guide.train import REPORTS

DATASETS = {  # name -> (data dir, zero-shot template)
    "genre": ("data/wikiart_genre5", "a painting of a {name}"),
    "style": ("data/wikiart_style6", "a painting in the style of {name}"),
}
CS = (0.1, 1, 10, 100, 1000)
BASELINE = "clip_b32"
OUT = REPORTS / "encoder_comparison.json"


# ------------------------------------------------------------------ accuracy stage


def embeddings_for(name, dev, data):
    """{dataset: {"images": (n, d), "text": (n_classes, d) or None}} for one encoder, with caching."""
    spec, loaded, out = REGISTRY[name], [], {}

    def encoder():  # load the model only when something is missing from the caches (or for the text tower)
        if not loaded:
            loaded.append(load_encoder(name, dev))
        return loaded[0]

    for ds, (data_dir, template) in DATASETS.items():
        d = Path(data_dir)
        df = data[ds]["df"]
        paths = [d / p for p in df["path"]]
        if name == BASELINE:  # the vectors already computed by scripts/clip_classifier.py and style_task.py
            cache, key = d / "clip_embeddings.npz", CLIP_B32
        else:
            cache, key = d / f"enc_{name}.npz", f"{spec.hf_id}|{spec.convention}|{spec.input_size}"
        images = cached_embeddings(cache, key, paths, lambda p=paths: encoder().embed_images(p))
        text = encoder().embed_texts(class_prompts(data[ds]["classes"], template)) if spec.has_text else None
        out[ds] = {"images": images, "text": text}
    return out


def probe_runs(emb, df, n_seeds):
    """Per-seed probe accuracy, macro-F1, chosen C and zero-shot accuracy for one encoder on one dataset."""
    y = df["label_idx"].values
    res = {"probe_acc": [], "macro_f1": [], "chosen_C": [], "zero_shot_acc": []}
    for seed in range(n_seeds):
        parts = split(df, seed)
        idx = {k: df.index[df["id"].isin(v["id"])] for k, v in parts.items()}
        clf, C = fit_linear_probe(emb["images"][idx["train"]], y[idx["train"]], emb["images"][idx["val"]], y[idx["val"]], CS)
        pred = clf.predict(emb["images"][idx["test"]])
        res["probe_acc"].append(float((pred == y[idx["test"]]).mean()))
        res["macro_f1"].append(float(f1_score(y[idx["test"]], pred, average="macro")))
        res["chosen_C"].append(C)
        if emb["text"] is not None:
            zs = zero_shot_predict(emb["images"][idx["test"]], emb["text"])
            res["zero_shot_acc"].append(float((zs == y[idx["test"]]).mean()))
    return res


def accuracy_stage(names, n_seeds, results):
    dev = device()
    data = {}
    for ds, (data_dir, _) in DATASETS.items():
        if not Path(data_dir, "metadata.csv").exists():
            print(f"skipping the {ds} dataset: {data_dir} not found", flush=True)
            continue
        df = load_metadata(Path(data_dir))
        data[ds] = {"df": df, "classes": class_names(df)}
    results["datasets_used"] = {ds: DATASETS[ds][0] for ds in data}
    acc = results.setdefault("accuracy", {ds: {} for ds in data})
    skipped = results.setdefault("skipped", {})
    for name in names:
        t = time.time()
        try:
            emb = embeddings_for(name, dev, {ds: data[ds] for ds in data})
        except Exception as err:  # an encoder that cannot load or run is reported, not fatal
            skipped[name] = f"{type(err).__name__}: {str(err)[:300]}"
            print(f"{name}: SKIPPED ({skipped[name]})", flush=True)
            continue
        skipped.pop(name, None)
        for ds in data:
            res = probe_runs(emb[ds], data[ds]["df"], n_seeds)
            acc[ds][name] = {
                "probe_acc": dict(zip(("mean", "std"), mean_std(res["probe_acc"]))),
                "macro_f1": dict(zip(("mean", "std"), mean_std(res["macro_f1"]))),
                "zero_shot_acc": dict(zip(("mean", "std"), mean_std(res["zero_shot_acc"]))) if res["zero_shot_acc"] else None,
                "per_seed": res,
            }
        print(f"{name}: " + "  ".join(f"{ds} probe {acc[ds][name]['probe_acc']['mean']:.3f}" for ds in data) + f"  ({time.time() - t:.0f}s)", flush=True)
        if dev.type == "mps":
            torch.mps.empty_cache()
    for ds in acc:  # paired comparison with the baseline over the same seeds
        base = acc[ds].get(BASELINE)
        for name, r in acc[ds].items():
            if base is not None and name != BASELINE:
                r["vs_clip_b32"] = paired_wins(r["per_seed"]["probe_acc"], base["per_seed"]["probe_acc"])
                diff = np.array(r["per_seed"]["probe_acc"]) - np.array(base["per_seed"]["probe_acc"])
                r["vs_clip_b32"]["mean_difference"] = float(diff.mean())


# ------------------------------------------------------------------ latency stage


def latency_worker(name, repeats_images=100):
    """Time the image tower of one encoder in this fresh process, one thread, batch of one image."""
    torch.set_num_threads(1)
    tower, size = image_tower(name)
    x = torch.randn(1, 3, size, size)
    with torch.no_grad():
        for _ in range(10):
            tower(x)
        times = []
        for _ in range(repeats_images):
            t = time.perf_counter()
            tower(x)
            times.append((time.perf_counter() - t) * 1000)
    print(json.dumps({"params": n_params(tower), "weights_mb": serialized_mb(tower.state_dict()), "input_size": size,
                      "latency": summarize_ms(times), "load_average_after": os.getloadavg()[0]}))


def wait_until_quiet(max_load=3.0, timeout_s=900):
    """Wait for the 1-minute load average to fall below `max_load`; return (quiet, load average at the end)."""
    start = time.time()
    while os.getloadavg()[0] > max_load and time.time() - start < timeout_s:
        time.sleep(20)
    return os.getloadavg()[0] <= max_load, os.getloadavg()[0]


def latency_stage(names, repeats, results):
    quiet, load = wait_until_quiet()
    results["latency_conditions"] = {"quiet_when_started": quiet, "load_average_when_started": load, "threads": 1,
                                     "batch": 1, "timed_forward_passes_per_process": 100, "processes_per_encoder": repeats,
                                     "tower": "random weights of the right architecture"}
    cost = results.setdefault("cost", {})
    for name in names:
        runs = []
        for _ in range(repeats):
            out = subprocess.run([sys.executable, "-W", "ignore", __file__, "--worker", name], capture_output=True, text=True, check=True).stdout
            runs.append(json.loads(out.strip().splitlines()[-1]))
        med = [r["latency"]["median"] for r in runs]
        cost[name] = {
            "params": runs[0]["params"],
            "weights_mb": runs[0]["weights_mb"],
            "input_size": runs[0]["input_size"],
            "latency_ms": {"median_of_runs": float(np.median(med)), "min": float(min(med)), "max": float(max(med)),
                           "p95_median_of_runs": float(np.median([r["latency"]["p95"] for r in runs]))},
            "load_average_after": max(r["load_average_after"] for r in runs),
        }
        print(f"{name:16s} {cost[name]['params'] / 1e6:6.1f} M params  {cost[name]['weights_mb']:6.1f} MB  "
              f"{cost[name]['latency_ms']['median_of_runs']:7.1f} ms [{min(med):.1f}-{max(med):.1f}]", flush=True)


# ------------------------------------------------------------------ report


def plot(results, path):
    acc, cost = results.get("accuracy", {}), results.get("cost", {})
    sets = [ds for ds in acc if acc[ds]]
    if not sets or not cost:
        return
    fig, axes = plt.subplots(1, len(sets), figsize=(6.4 * len(sets), 4.6), squeeze=False)
    for ax, ds in zip(axes[0], sets):
        for name, r in acc[ds].items():
            if name not in cost:
                continue
            x, y, w = cost[name]["latency_ms"]["median_of_runs"], 100 * r["probe_acc"]["mean"], cost[name]["weights_mb"]
            ax.scatter(x, y, s=40 + w * 0.6, alpha=0.6, color="tab:red" if name == BASELINE else "tab:blue")
            ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("image tower, 1 thread, 1 image (ms, log scale)")
        ax.set_ylabel("linear-probe test accuracy (%, mean of 10 seeds)")
        ax.set_title(f"{ds}: accuracy against cost (bubble = weight file)")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summary_lines(results):
    cost, acc = results.get("cost", {}), results.get("accuracy", {})
    print("\nencoder           " + "  ".join(f"{ds:>14s}" for ds in acc) + "   params  weights   latency")
    for name in REGISTRY:
        if not any(name in acc[ds] for ds in acc):
            continue
        cells = "  ".join(f"{100 * acc[ds][name]['probe_acc']['mean']:6.1f}±{100 * acc[ds][name]['probe_acc']['std']:4.1f}%  " if name in acc[ds] else " " * 14 for ds in acc)
        c = cost.get(name)
        tail = f"{c['params'] / 1e6:6.1f}M {c['weights_mb']:6.0f}MB {c['latency_ms']['median_of_runs']:7.1f}ms" if c else ""
        print(f"{name:16s}  {cells} {tail}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", help="internal: time one encoder's image tower and print JSON")
    p.add_argument("--stage", choices=("all", "accuracy", "latency"), default="all")
    p.add_argument("--encoders", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--repeats", type=int, default=3, help="fresh processes per encoder for the latency")
    args = p.parse_args()
    if args.worker:
        return latency_worker(args.worker)

    results = json.loads(OUT.read_text()) if OUT.exists() else {}
    results["protocol"] = {
        "probe": "logistic regression on L2-normalised frozen vectors, C chosen on validation among " + str(list(CS)),
        "splits": f"split(df, seed) for seeds 0..{args.seeds - 1}",
        "source_images": "128 px, upsampled to each encoder's input size",
        "conventions": {n: REGISTRY[n].convention for n in REGISTRY},
        "encoders": {n: {"hf_id": s.hf_id, "input_size": s.input_size} for n, s in REGISTRY.items()},
    }
    if args.stage in ("all", "latency"):
        latency_stage(args.encoders, args.repeats, results)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(results, indent=2))
    if args.stage in ("all", "accuracy"):
        accuracy_stage(args.encoders, args.seeds, results)
        OUT.write_text(json.dumps(results, indent=2))
    plot(results, REPORTS / "figures" / "encoder_comparison.png")
    summary_lines(results)


if __name__ == "__main__":
    main()
