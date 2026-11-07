"""What does the guide do when the photo is not a painting, and can it tell? (a wall, a face, an object...)

    python scripts/fetch_non_paintings.py     # once: about 400 MB of ordinary photos in data/ood/
    python scripts/non_painting.py            # a few minutes

1. Shows how the classifiers react: the share of non-paintings that the ResNet (threshold 0.71) and the CLIP head
   (threshold 0.83) would still ANSWER, i.e. would not call "unsure".
2. Compares ways to tell a painting from a non-painting, by AUROC and by the share of non-paintings accepted when
   95% of the paintings are accepted: the top probability of each classifier, CLIP zero-shot sentence pairs, the
   similarity to the collection, and a combination of the two best signals.
3. Chooses the method and the threshold on a VALIDATION draw (other photos than the test draw; paintings of the
   validation split), then reports the result on the TEST draw. The chosen rule is saved to
   `reports/non_painting_rule.json` and can be used with `--reject-non-paintings` (predict.py, guide.py).

Photos: Caltech-101 objects and faces, Imagenette, DTD textures (see `fetch_non_paintings.py`), plus synthetic
degenerate frames (uniform wall, plaster, black frame, noise). Paintings: the seed-0 splits of the genre data.
A visitor's photo is not cropped to the canvas, so the validation "paintings" are the cropped ones AND the same number
of crude compositions of a painting in a thin frame on a wall; the test reports both. A heavily blurred painting is
reported separately: it is still a painting, only a bad photo of one.
Writes `reports/non_painting.json`, the score figure in `reports/figures/`, and a grid of failure cases in `data/ood/`
(thumbnails of other people's photos and of WikiArt paintings: not versioned).
"""
import argparse
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageFilter

from museum_guide.calibration import load_temperature, softmax
from museum_guide.clip import MODEL, cached_embeddings, embed_images, load_encoder
from museum_guide.data import DATA_DIR, device, load_metadata, loaders, make_transforms, split
from museum_guide.models import pretrained_resnet18
from museum_guide.ood import (
    PROMPT_PAIRS,
    SAME_PAINTING,
    NonPaintingRule,
    accepted_share,
    auroc,
    collection_similarity,
    fit_linear_score,
    fpr_at_tpr,
    max_probability,
    save_rule,
    threshold_for_tpr,
    zero_shot_margin,
)
from museum_guide.predict import DEFAULT_PROBE, SIZE, load_probe
from museum_guide.train import REPORTS, fit

OOD = Path("data/ood")  # where scripts/fetch_non_paintings.py wrote the photos
RESNET_THRESHOLD, CLIP_HEAD_THRESHOLD = 0.71, 0.83  # the "answer" thresholds documented with PR #9 and #18
SYNTHETIC = ("uniform_wall", "plaster_wall", "black_frame", "gaussian_noise")
N_SYNTH_TEST, N_SYNTH_VAL = 60, 30


# ------------------------------------------------------------------------------------------------
# Inputs
# ------------------------------------------------------------------------------------------------


def jpeg_round_trip(img: Image.Image, quality: int = 90) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def synthetic_images(kind: str, n: int, seed: int) -> list[Image.Image]:
    """Degenerate frames a camera can produce, 128 x 128, with the same JPEG compression as the other photos."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        if kind == "uniform_wall":
            level = rng.uniform(90, 245) + rng.normal(0, 3, 3)  # a plain wall with a slight tint
            gradient = np.linspace(-8, 8, 128)[:, None, None] * rng.choice([-1, 1])
            arr = level[None, None, :] + gradient + rng.normal(0, 2, (128, 128, 3))
        elif kind == "plaster_wall":
            base = rng.uniform(120, 230) + rng.normal(0, 6, 3)
            coarse = np.asarray(Image.fromarray(rng.integers(0, 255, (8, 8), dtype=np.uint8)).resize((128, 128), Image.Resampling.BICUBIC), float)
            arr = base[None, None, :] + 0.12 * (coarse[..., None] - 128) + rng.normal(0, 3, (128, 128, 3))
        elif kind == "black_frame":
            arr = rng.uniform(0, 14) + rng.normal(0, 3, (128, 128, 3))
        elif kind == "gaussian_noise":
            arr = rng.normal(128, 50, (128, 128, 3))
        else:
            raise ValueError(kind)
        out.append(jpeg_round_trip(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))))
    return out


def on_wall(painting: Image.Image, wall: Image.Image, rng) -> Image.Image:
    """A crude stand-in for a visitor's photo: the painting in a thin frame, slightly rotated, on a wall (128 x 128).

    A proxy only: no glare, no perspective, no neighbouring pictures. It checks that a painting surrounded by wall is
    still accepted, which a photo cropped to the canvas (the test paintings) cannot show.
    """
    size = int(rng.uniform(0.55, 0.75) * 128)
    art = painting.resize((size, size), Image.Resampling.BICUBIC)
    frame_colour = [(170, 130, 50), (60, 40, 30), (30, 30, 30)][int(rng.integers(3))]
    framed = Image.new("RGB", (size + 6, size + 6), frame_colour)
    framed.paste(art, (3, 3))
    framed = framed.rotate(float(rng.uniform(-4, 4)), expand=True, resample=Image.Resampling.BICUBIC, fillcolor=tuple(wall.getpixel((0, 0))))
    canvas = wall.copy()
    x = (128 - framed.width) // 2 + int(rng.integers(-6, 7))
    y = (128 - framed.height) // 2 + int(rng.integers(-6, 7))
    canvas.paste(framed, (x, y))
    return jpeg_round_trip(canvas)


def open_rgb(path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def resnet_probabilities(model, images, temperature, dev, batch=64):
    """Calibrated softmax of the ResNet on PIL images, with the transform of `predict.classify`."""
    transform = make_transforms(SIZE, augment=False)
    out = []
    model.eval().to(dev)
    with torch.no_grad():
        for i in range(0, len(images), batch):
            x = torch.stack([transform(im) for im in images[i : i + batch]]).to(dev)
            out.append((model(x) / temperature).softmax(1).cpu().numpy())
    return np.concatenate(out)


def trained_resnet(parts, dev, cache: Path):
    """The seed-0 ResNet-18 (layer4 + head), as in scripts/robustness.py; cached in data/ood/ (not versioned)."""
    data = loaders(parts, SIZE, augment=True, batch_size=32)
    torch.manual_seed(0)
    model = pretrained_resnet18(5, train_layer4=True)
    if cache.exists():
        model.load_state_dict(torch.load(cache, map_location="cpu", weights_only=True))
    else:
        fit(model, data, dev, epochs=20, lr=3e-4, weight_decay=1e-4, patience=8, verbose=False)
        cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), cache)
    return model.eval()


# ------------------------------------------------------------------------------------------------
# Main experiment
# ------------------------------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpr", type=float, default=0.95, help="share of paintings the rule must accept (on validation)")
    args = p.parse_args()
    manifest_file = OOD / "manifest.json"
    if not manifest_file.exists():
        raise SystemExit("no photos: run `python scripts/fetch_non_paintings.py` first")
    manifest = json.loads(manifest_file.read_text())
    sources = list(manifest["sources"])

    dev, df = device(), load_metadata()
    parts = split(df, 0)
    encoder = load_encoder()
    painting_paths = [DATA_DIR / q for q in df["path"]]
    all_vectors = cached_embeddings(DATA_DIR / "clip_embeddings.npz", MODEL, painting_paths,
                                    lambda: embed_images(encoder.model, encoder.processor, painting_paths))
    idx = {k: np.where(df["id"].isin(v["id"]).to_numpy())[0] for k, v in parts.items()}
    labels = df["label_idx"].to_numpy()

    # --- images: paintings (val, test), photos per source (val, test), synthetic frames, blurred paintings ---
    vec, img = {}, {}  # keyed by (group, subset); groups: painting, <source>, <synthetic kind>, blurred_painting
    for subset in ("val", "test"):
        vec["painting", subset] = all_vectors[idx[subset]]
        img["painting", subset] = [open_rgb(painting_paths[i]) for i in idx[subset]]
    for src in sources:
        for subset in ("val", "test"):
            files = manifest["sources"][src][subset]
            img[src, subset] = [open_rgb(f) for f in files]
            vec[src, subset] = cached_embeddings(OOD / f"clip_{src}_{subset}.npz", MODEL, files, lambda s=src, k=subset: encoder.images(img[s, k]))
    for kind in SYNTHETIC:
        for subset, n, seed in (("val", N_SYNTH_VAL, 11), ("test", N_SYNTH_TEST, 12)):
            img[kind, subset] = synthetic_images(kind, n, seed + 100 * SYNTHETIC.index(kind))
            vec[kind, subset] = encoder.images(img[kind, subset])
    blurred = [im.filter(ImageFilter.GaussianBlur(8)) for im in img["painting", "test"][:100]]
    img["blurred_painting", "test"] = [jpeg_round_trip(b) for b in blurred]
    vec["blurred_painting", "test"] = encoder.images(img["blurred_painting", "test"])
    rng_wall = np.random.default_rng(77)
    walls = synthetic_images("plaster_wall", 100, 99)
    walls_val = synthetic_images("plaster_wall", 100, 199)
    img["painting_on_wall", "test"] = [on_wall(p, w, rng_wall) for p, w in zip(img["painting", "test"][:100], walls)]
    img["painting_on_wall", "val"] = [on_wall(p, w, rng_wall) for p, w in zip(img["painting", "val"][:100], walls_val)]
    for subset in ("test", "val"):
        vec["painting_on_wall", subset] = encoder.images(img["painting_on_wall", subset])
    non_painting_groups = sources + list(SYNTHETIC)

    # --- classifier outputs ---
    resnet = trained_resnet(parts, dev, OOD / "resnet18_seed0.pt")
    t_resnet = load_temperature(Path("reports/calibration.json"))
    probe = load_probe(DEFAULT_PROBE)
    test_acc = float((np.argmax(resnet_probabilities(resnet, img["painting", "test"], t_resnet, dev), 1) == labels[idx["test"]]).mean())
    print(f"ResNet seed-0 test accuracy {test_acc:.4f} (README: 0.7455)", flush=True)
    resnet_msp, head_msp = {}, {}
    for key in img:
        resnet_msp[key] = max_probability(resnet_probabilities(resnet, img[key], t_resnet, dev))
        head_msp[key] = max_probability(softmax(probe.logits(vec[key]), probe.temperature))

    # --- zero-shot and collection signals ---
    prompt_vecs = {name: (encoder.texts(list(pos)), encoder.texts(list(neg))) for name, (pos, neg) in PROMPT_PAIRS.items()}
    margin = {name: {k: zero_shot_margin(v, *prompt_vecs[name]) for k, v in vec.items()} for name in PROMPT_PAIRS}
    index_vectors = all_vectors
    # the original of a blurred or framed painting is in the collection: leave it out
    own = {("blurred_painting", "test"): idx["test"][:100], ("painting_on_wall", "test"): idx["test"][:100],
           ("painting_on_wall", "val"): idx["val"][:100]}
    sim = {k: {n: collection_similarity(v, index_vectors, n, SAME_PAINTING, own.get(k)) for n in (1, 5)} for k, v in vec.items()}
    sim1 = {k: s[1] for k, s in sim.items()}
    sim5 = {k: s[5] for k, s in sim.items()}

    def pooled(score, subset, groups):
        return np.concatenate([score[g, subset] for g in groups])

    def val_positives(score):
        """Validation paintings: the cropped ones and the ones framed on a wall (a visitor's photo is not cropped to the canvas)."""
        return np.concatenate([score["painting", "val"], score["painting_on_wall", "val"]])

    methods = {"resnet_max_probability": resnet_msp, "clip_head_max_probability": head_msp, "collection_similarity_top1": sim1, "collection_similarity_top5": sim5}
    methods.update({f"zero_shot:{n}": m for n, m in margin.items()})

    # --- choose the best zero-shot pair on VALIDATION, then a combination fitted on VALIDATION ---
    val_auc = {n: auroc(val_positives(methods[n]), pooled(methods[n], "val", non_painting_groups)) for n in methods}
    best_pair = max((n for n in methods if n.startswith("zero_shot:")), key=val_auc.get)
    feats = {k: np.stack([methods[best_pair][k], sim5[k]], axis=1) for k in vec}
    w, bias = fit_linear_score(np.vstack([feats["painting", "val"], feats["painting_on_wall", "val"]]), np.vstack([feats[g, "val"] for g in non_painting_groups]))
    methods["combination:" + best_pair.split(":")[1] + "+similarity_top5"] = {k: feats[k] @ w + bias for k in vec}
    val_auc = {n: auroc(val_positives(methods[n]), pooled(methods[n], "val", non_painting_groups)) for n in methods}

    # --- report on TEST ---
    report = {"n_paintings_test": len(idx["test"]), "n_paintings_val": len(idx["val"]), "resnet_test_accuracy": test_acc,
              "non_painting_test_counts": {g: len(img[g, "test"]) for g in non_painting_groups},
              "answered_at_documented_thresholds": {}, "methods": {}, "validation_auroc_pooled": val_auc,
              "tpr_target": args.tpr}
    for name, thr, score in (("resnet", RESNET_THRESHOLD, resnet_msp), ("clip_head", CLIP_HEAD_THRESHOLD, head_msp)):
        report["answered_at_documented_thresholds"][name] = {
            "threshold": thr, "paintings": accepted_share(score["painting", "test"], thr),
            **{g: accepted_share(score[g, "test"], thr) for g in non_painting_groups},
            "all_real_photos": accepted_share(pooled(score, "test", sources), thr),
            "blurred_painting": accepted_share(score["blurred_painting", "test"], thr),
            "painting_on_wall": accepted_share(score["painting_on_wall", "test"], thr),
        }
    real = sources
    for name, score in methods.items():
        pos_t = score["painting", "test"]
        thr_val = threshold_for_tpr(val_positives(score), args.tpr)  # chosen on validation paintings (cropped and framed)
        row = {"validation_auroc_all_non_paintings": val_auc[name], "threshold_from_validation": thr_val,
               "test_tpr_at_validation_threshold": accepted_share(pos_t, thr_val),
               "test_accepted_blurred_painting": accepted_share(score["blurred_painting", "test"], thr_val),
               "test_accepted_painting_on_wall": accepted_share(score["painting_on_wall", "test"], thr_val),
               "test_auroc_painting_on_wall_vs_real_photos": auroc(score["painting_on_wall", "test"], pooled(score, "test", real)), "per_group": {}}
        for label, groups in (("all_real_photos", real), ("all_synthetic", list(SYNTHETIC)), ("all_non_paintings", non_painting_groups)):
            neg = pooled(score, "test", groups)
            fpr, _ = fpr_at_tpr(pos_t, neg, args.tpr)
            row[label] = {"auroc": auroc(pos_t, neg), "fpr_at_tpr": fpr, "accepted_at_validation_threshold": accepted_share(neg, thr_val)}
        for g in non_painting_groups:
            neg = score[g, "test"]
            fpr, _ = fpr_at_tpr(pos_t, neg, args.tpr)
            row["per_group"][g] = {"auroc": auroc(pos_t, neg), "fpr_at_tpr": fpr, "accepted_at_validation_threshold": accepted_share(neg, thr_val)}
        report["methods"][name] = row

    # --- the rule for the guide: the best method on validation among those that can be deployed ---
    deployable = [n for n in methods if not n.startswith(("resnet", "clip_head"))]
    chosen = max(deployable, key=lambda n: val_auc[n])
    pair = best_pair.split(":")[1]
    if chosen.startswith("combination:"):
        wm, ws, b0 = float(w[0]), float(w[1]), float(bias)
    elif chosen.startswith("zero_shot:"):
        wm, ws, b0, pair = 1.0, 0.0, 0.0, chosen.split(":")[1]
    else:  # a similarity-only method: no sentences involved; keep a rule shape that reproduces it
        wm, ws, b0 = 0.0, 1.0, 0.0
    top_k = 1 if chosen == "collection_similarity_top1" else 5
    rule = NonPaintingRule(
        painting_prompts=PROMPT_PAIRS[pair][0], other_prompts=PROMPT_PAIRS[pair][1], threshold=float(threshold_for_tpr(val_positives(methods[chosen]), args.tpr)),
        w_margin=wm, w_similarity=ws, bias=b0, top_k=top_k, target_tpr=args.tpr,
        info={"method": chosen, "validation_auroc": val_auc[chosen], "n_validation_paintings": len(idx["val"]),
              "n_validation_non_paintings": int(sum(len(img[g, "val"]) for g in non_painting_groups)),
              "fitted_on": "validation draw only (other photos than the test draw)"},
    )
    save_rule(rule, REPORTS / "non_painting_rule.json")
    report["chosen_rule"] = {"method": chosen, "pair": pair, "weights": {"margin": wm, "similarity": ws, "bias": b0},
                             "threshold": rule.threshold, "top_k": top_k}
    tradeoff = {}
    for tpr in (0.90, 0.95, 0.98, 0.99):
        thr = threshold_for_tpr(val_positives(methods[chosen]), tpr)
        tradeoff[str(tpr)] = {
            "threshold": float(thr),
            "paintings_accepted_test": accepted_share(methods[chosen]["painting", "test"], thr),
            "real_photos_accepted_test": accepted_share(pooled(methods[chosen], "test", real), thr),
            "synthetic_accepted_test": accepted_share(pooled(methods[chosen], "test", list(SYNTHETIC)), thr),
            "painting_on_wall_accepted_test": accepted_share(methods[chosen]["painting_on_wall", "test"], thr),
        }
    report["chosen_method_tradeoff_validation_thresholds"] = tradeoff
    (REPORTS / "non_painting.json").write_text(json.dumps(report, indent=2))

    # --- print ---
    print("\nShare of each group ANSWERED (not 'unsure') at the documented thresholds:")
    for name, row in report["answered_at_documented_thresholds"].items():
        print(f"  {name:10s} (>= {row['threshold']}): paintings {row['paintings']:.0%} | " + " ".join(f"{g} {row[g]:.0%}" for g in non_painting_groups) + f" | blurred painting {row['blurred_painting']:.0%} | painting on a wall {row['painting_on_wall']:.0%}")
    print(f"\nAUROC (paintings vs all real photos) and share of real photos accepted when {args.tpr:.0%} of paintings are:")
    for name, row in report["methods"].items():
        r = row["all_real_photos"]
        print(f"  {name:48s} val AUROC {row['validation_auroc_all_non_paintings']:.3f} | test AUROC {r['auroc']:.3f}  FPR {r['fpr_at_tpr']:.1%}  (val threshold: TPR {row['test_tpr_at_validation_threshold']:.1%}, accepted {r['accepted_at_validation_threshold']:.1%})")
    print("\nChosen rule:", json.dumps(report["chosen_rule"]))
    print("Painting on a wall accepted, per method (validation threshold):", {n: round(r["test_accepted_painting_on_wall"], 2) for n, r in report["methods"].items()})
    print("Trade-off of the chosen method:", json.dumps(tradeoff))

    # --- figures ---
    best_name = "combination:" + best_pair.split(":")[1] + "+similarity_top5" if chosen.startswith("combination:") else chosen
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, name, title in ((axes[0], "resnet_max_probability", "ResNet top probability (baseline)"), (axes[1], best_name, f"chosen score: {best_name}")):
        sc = methods[name]
        bins = np.linspace(min(np.min(v) for v in sc.values()), max(np.max(v) for v in sc.values()), 40)
        ax.hist(sc["painting", "test"], bins=bins, alpha=0.6, density=True, label="paintings", color="tab:blue")
        ax.hist(pooled(sc, "test", real), bins=bins, alpha=0.5, density=True, label="real photos", color="tab:orange")
        ax.hist(pooled(sc, "test", list(SYNTHETIC)), bins=bins, alpha=0.4, density=True, label="synthetic frames", color="tab:green")
        ax.axvline(threshold_for_tpr(val_positives(sc), args.tpr), color="black", ls="--", label=f"threshold ({args.tpr:.0%} of val paintings)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS / "figures" / "non_painting_scores.png", dpi=120)
    plt.close(fig)

    # failure cases of the chosen method: non-paintings that look most like paintings, paintings that look least like one
    sc = methods[best_name]
    thr = threshold_for_tpr(val_positives(sc), args.tpr)
    fooling = sorted(((s, g, i) for g in real + list(SYNTHETIC) for i, s in enumerate(sc[g, "test"])), reverse=True)[:12]
    rejected = sorted((s, i) for i, s in enumerate(sc["painting", "test"]))[:12]
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (s, g, i) in zip(axes[:2].ravel(), fooling):
        ax.imshow(img[g, "test"][i])
        ax.set_title(f"{g}\nscore {s:.2f}", fontsize=7)
    for ax, (s, i) in zip(axes[2:].ravel(), rejected):
        ax.imshow(img["painting", "test"][i])
        ax.set_title(f"painting ({df['label'].iloc[idx['test'][i]]})\nscore {s:.2f}", fontsize=7)
    fig.suptitle(f"top 2 rows: the 12 non-paintings that score highest (threshold {thr:.2f}); bottom 2 rows: the 12 paintings that score lowest", fontsize=9)
    fig.tight_layout()
    # Thumbnails of other people's photos and of WikiArt paintings: kept next to the data, never committed.
    fig.savefig(OOD / "non_painting_failures.png", dpi=110)
    plt.close(fig)
    print(f"failure cases (thumbnails, not versioned): {OOD / 'non_painting_failures.png'}")


if __name__ == "__main__":
    main()
