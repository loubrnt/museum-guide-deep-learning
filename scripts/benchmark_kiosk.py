"""What would each model cost on a kiosk? Weight size, peak memory, latency per image, int8 and fp16.

The README says kiosk memory "must be checked". This script measures, for the ResNet-18 and for CLIP's image
encoder (fp32, fp16 and int8-quantised):

- the size of the weight file (`torch.save` of the state dict);
- the peak memory of a process serving the model (each model runs in a FRESH process, so figures do not leak
  from one model to the next);
- the latency of one image end to end (read the JPEG, resize/normalise, forward pass), on 1 and on 4 threads,
  with median and 95th percentile over many different images, repeated in several processes;
- the cost of reading a full-size phone photo (12 megapixels), with and without JPEG draft decoding;
- what fp16 and int8 do to CLIP's accuracy (a linear probe trained on fp32 vectors, tested on quantised ones).

Models in the timing part have RANDOM weights: shapes, size, memory and speed do not depend on the values. Each model
is built and saved once, then loaded by fresh processes that only serve it (so memory excludes model construction).

    python scripts/benchmark_kiosk.py                  # about 15 minutes; leave the machine alone while it runs
    python scripts/benchmark_kiosk.py --skip-accuracy  # timings only

Numbers are specific to the machine (printed in the report): the ratios between models travel better than the
milliseconds. Writes `reports/kiosk_benchmark.json` and `reports/figures/kiosk_benchmark.png`.
"""
import argparse
import copy
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from museum_guide.benchmark import (
    clip_image_tower,
    n_params,
    peak_rss_mb,
    quantize_linear_int8,
    select_quantized_engine,
    serialized_mb,
    summarize_ms,
)
from museum_guide.clip import MODEL, cached_embeddings, embed_images, fit_linear_probe, load_clip
from museum_guide.data import DATA_DIR, load_metadata, make_transforms, split
from museum_guide.predict import SIZE as RESNET_SIZE
from museum_guide.robustness import mean_std
from museum_guide.train import REPORTS

CONFIGS = ("resnet18_fp32", "clip_fp32", "clip_fp16", "clip_int8")
WARMUP_IMAGES = 10


def build_model(config):
    """(model, dtype of the input) for one configuration; random weights."""
    if config == "resnet18_fp32":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 5)
        return model.eval(), torch.float32
    tower = clip_image_tower()
    if config == "clip_fp16":
        return tower.half().eval(), torch.float16
    if config == "clip_int8":
        return quantize_linear_int8(tower).eval(), torch.float32
    return tower.eval(), torch.float32


def build_transform(config):
    """Image -> (3,H,W) tensor, as the project does it for this model."""
    if config == "resnet18_fp32":
        return make_transforms(RESNET_SIZE, augment=False)
    from transformers import CLIPImageProcessor

    processor = CLIPImageProcessor()  # defaults: resize to 224, centre crop, CLIP mean and std
    return lambda img: processor(images=img, return_tensors="pt")["pixel_values"][0]


def worker(config, threads, n_images, model_file):
    """Load the saved model in this (fresh) process, serve images, print the measurements as JSON.

    Memory is therefore that of a process that only loads and serves the model: the construction of the model
    (random initialisation, fp16 conversion, quantisation) happened in another process.
    """
    torch.set_num_threads(threads)
    transform = build_transform(config)  # imports the image-processing libraries: part of the process, not of the model
    dtype = torch.float16 if config == "clip_fp16" else torch.float32
    if config == "clip_int8":
        select_quantized_engine()  # needed to unpickle int8 layers
    rss_before = peak_rss_mb()
    model = torch.load(model_file, weights_only=False).eval()
    paths = [DATA_DIR / p for p in load_metadata()["path"].iloc[: WARMUP_IMAGES + n_images]]

    def one(path):
        t0 = time.perf_counter()
        with Image.open(path) as im:
            img = im.convert("RGB")
        t1 = time.perf_counter()
        x = transform(img).unsqueeze(0).to(dtype)
        t2 = time.perf_counter()
        with torch.no_grad():
            model(x)
        t3 = time.perf_counter()
        return [(t1 - t0) * 1000, (t2 - t1) * 1000, (t3 - t2) * 1000]

    for p in paths[:WARMUP_IMAGES]:
        one(p)
    rows = np.array([one(p) for p in paths[WARMUP_IMAGES:]])
    print(json.dumps({
        "config": config,
        "threads": threads,
        "params": n_params(model),
        "weights_mb": serialized_mb(model.state_dict()),
        "peak_rss_mb": peak_rss_mb(),
        "rss_before_model_mb": rss_before,  # process with libraries imported, model not loaded yet

        "read_ms": summarize_ms(rows[:, 0]),
        "preprocess_ms": summarize_ms(rows[:, 1]),
        "forward_ms": summarize_ms(rows[:, 2]),
        "total_ms": summarize_ms(rows.sum(1)),
    }))


def camera_worker(repeats=10):
    """Cost of reading a 12-megapixel JPEG, in full or with draft decoding (JPEG decodes at 1/2, 1/4 or 1/8 scale)."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(0)
    smooth = Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).resize((4032, 3024), Image.BICUBIC)
    noisy = np.clip(np.asarray(smooth).astype(np.int16) + rng.integers(-6, 7, (3024, 4032, 3)), 0, 255).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "photo.jpg"
        Image.fromarray(noisy).save(path, quality=90)

        def full():
            with Image.open(path) as im:
                return im.convert("RGB")

        def draft():
            with Image.open(path) as im:
                im.draft("RGB", (448, 448))  # decode at the largest reduction that stays >= 448 px
                return im.convert("RGB")

        out = {"image_pixels": "4032x3024", "file_mb": path.stat().st_size / 1e6}
        for name, fn in (("full_decode_ms", full), ("draft_decode_ms", draft)):
            fn()
            times = []
            for _ in range(repeats):
                t = time.perf_counter()
                fn()
                times.append((time.perf_counter() - t) * 1000)
            out[name] = summarize_ms(times)
        out["draft_size_px"] = list(draft().size)
    print(json.dumps(out))


def run_worker(*args):
    """Run this script as a fresh subprocess in worker mode and parse its JSON line."""
    cmd = [sys.executable, "-W", "ignore", __file__, *map(str, args)]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def hardware():
    info = {"machine": platform.machine(), "system": f"{platform.system()} {platform.release()}", "cpu_count": os.cpu_count(),
            "torch": torch.__version__, "python": platform.python_version(), "date": time.strftime("%Y-%m-%d")}
    if platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        info["chip"] = chip
    return info


def aggregate(runs):
    """Across the repeated processes of one (config, threads): median of the medians, and their range."""
    def over(key, stat="median"):
        v = [r[key][stat] for r in runs]
        return {"median_of_runs": float(np.median(v)), "min": float(min(v)), "max": float(max(v))}

    return {
        "params": runs[0]["params"],
        "weights_mb": runs[0]["weights_mb"],
        "peak_rss_mb": float(np.median([r["peak_rss_mb"] for r in runs])),
        "rss_added_by_model_mb": float(np.median([r["peak_rss_mb"] - r["rss_before_model_mb"] for r in runs])),
        "read_ms": over("read_ms"),
        "preprocess_ms": over("preprocess_ms"),
        "forward_ms": over("forward_ms"),
        "total_ms": over("total_ms"),
        "total_p95_ms": over("total_ms", "p95"),
        "n_runs": len(runs),
    }


def accuracy_under_quantisation(n_seeds):
    """CLIP linear probe trained on fp32 vectors and tested on fp32, fp16 and int8 vectors (10 splits)."""
    model, processor = load_clip()
    df = load_metadata()
    paths = [DATA_DIR / q for q in df["path"]]
    fp32 = cached_embeddings(DATA_DIR / "clip_embeddings.npz", MODEL, paths, lambda: embed_images(model, processor, paths))
    print("embedding with int8 weights...", flush=True)
    int8 = embed_images(quantize_linear_int8(model), processor, paths)
    print("embedding with fp16 weights...", flush=True)
    half = copy.deepcopy(model).half()
    vecs = []
    with torch.no_grad():
        for i in range(0, len(paths), 32):
            images = []
            for p in paths[i : i + 32]:
                with Image.open(p) as im:
                    images.append(im.convert("RGB"))
            pv = processor(images=images, return_tensors="pt")["pixel_values"].half()
            out = half.get_image_features(pixel_values=pv)
            vecs.append((out if hasattr(out, "shape") else out.pooler_output).float().numpy())
    fp16 = np.concatenate(vecs)
    fp16 /= np.linalg.norm(fp16, axis=1, keepdims=True)

    y = df["label_idx"].values
    variants = {"fp32": fp32, "fp16": fp16, "int8": int8}
    accs = {k: [] for k in variants}
    for seed in range(n_seeds):
        parts = split(df, seed)
        idx = {k: df.index[df["id"].isin(v["id"])] for k, v in parts.items()}
        probe, _ = fit_linear_probe(fp32[idx["train"]], y[idx["train"]], fp32[idx["val"]], y[idx["val"]])
        for name, emb in variants.items():
            accs[name].append(float(probe.score(emb[idx["test"]], y[idx["test"]])))
    out = {
        "n_seeds": n_seeds,
        "note": "probe trained on fp32 vectors only; the quantised encoder is used at inference time",
        "test_accuracy": {k: dict(zip(("mean", "std"), mean_std(v))) for k, v in accs.items()},
        "per_seed": accs,
    }
    for name in ("fp16", "int8"):
        diff = np.array(accs[name]) - np.array(accs["fp32"])
        out[f"{name}_minus_fp32"] = {"mean": float(diff.mean()), "min": float(diff.min()), "max": float(diff.max())}
        out[f"{name}_cosine_to_fp32"] = float(np.mean(np.sum(variants[name] * fp32, axis=1)))
    return out


def plot(results, path):
    one = {c: results["configs"][c]["threads_1"] for c in CONFIGS}
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    labels = [c.replace("_", "\n") for c in CONFIGS]
    axes[0].bar(labels, [one[c]["weights_mb"] for c in CONFIGS], color="tab:blue")
    axes[0].set_title("weight file (MB)")
    axes[1].bar(labels, [one[c]["rss_added_by_model_mb"] for c in CONFIGS], color="tab:green")
    axes[1].set_title("memory added by serving the model (MB)")
    bottom = np.zeros(len(CONFIGS))
    for key, name, color in (("read_ms", "read JPEG", "tab:gray"), ("preprocess_ms", "preprocess", "tab:orange"), ("forward_ms", "forward", "tab:red")):
        vals = np.array([one[c][key]["median_of_runs"] for c in CONFIGS])
        axes[2].bar(labels, vals, bottom=bottom, color=color, label=name)
        bottom += vals
    axes[2].set_title("latency per image, 1 thread (ms)")
    axes[2].legend()
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--worker", help="internal: measure one configuration (or 'camera') and print JSON")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--model-file", help="internal: saved model to load in worker mode")
    p.add_argument("--images", type=int, default=100, help="different images timed per process")
    p.add_argument("--repeats", type=int, default=3, help="fresh processes per (configuration, thread count)")
    p.add_argument("--skip-accuracy", action="store_true")
    args = p.parse_args()
    if args.worker == "camera":
        return camera_worker()
    if args.worker:
        return worker(args.worker, args.threads, args.images, args.model_file)

    results = {"hardware": hardware(), "load_average_start": os.getloadavg(), "configs": {}}
    results["quantised_engine"] = select_quantized_engine()
    tmp = tempfile.TemporaryDirectory()
    for config in CONFIGS:
        model_file = Path(tmp.name) / f"{config}.pt"
        torch.save(build_model(config)[0], model_file)  # built once here, loaded by each fresh worker
        results["configs"][config] = {}
        for threads in (1, 4):
            runs = [run_worker("--worker", config, "--threads", threads, "--images", args.images, "--model-file", model_file)
                    for _ in range(args.repeats)]
            agg = results["configs"][config][f"threads_{threads}"] = aggregate(runs)
            print(f"{config:14s} {threads} thread(s): total {agg['total_ms']['median_of_runs']:7.1f} ms "
                  f"(p95 {agg['total_p95_ms']['median_of_runs']:.1f}), forward {agg['forward_ms']['median_of_runs']:.1f} ms, "
                  f"weights {agg['weights_mb']:.0f} MB, memory +{agg['rss_added_by_model_mb']:.0f} MB", flush=True)
    results["camera_photo_read"] = run_worker("--worker", "camera")
    results["load_average_end"] = os.getloadavg()
    if not args.skip_accuracy:
        results["clip_precision_fp16_int8"] = accuracy_under_quantisation(10)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "kiosk_benchmark.json").write_text(json.dumps(results, indent=2))
    plot(results, REPORTS / "figures" / "kiosk_benchmark.png")
    print(json.dumps({k: v for k, v in results.items() if k not in ("configs",)}, indent=2))


if __name__ == "__main__":
    main()
