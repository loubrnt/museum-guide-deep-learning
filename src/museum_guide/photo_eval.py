"""Evaluate the guide on real photos taken by the team.

Nothing here has been run on a real camera photo yet: this module is the machinery for it. A photo folder has one
sub-folder per label (the five genres, and optionally `not_a_painting/`); every photo goes through the same
decisions as the guide (answer, "unsure", or "this does not look like a painting"), and the results are summarised
with intervals that resample *paintings*, not photos (several shots of one painting are not independent).

The functions are pure or take their models as arguments, so they are tested without CLIP or any real photo.
"""
from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix

from museum_guide.metrics import per_class_report
from museum_guide.predict import CLASSES
from museum_guide.robustness import bootstrap_accuracy

NOT_A_PAINTING = "not_a_painting"
VALID_LABELS = (*CLASSES, NOT_A_PAINTING)
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".heic", ".heif"})
SHOT_SEPARATOR = "__"  # `<paintingid>__<shot>.jpg`: photos with the same painting id are one cluster
ANSWERED, UNSURE, REJECTED = "answered", "unsure", "not_a_painting"
CSV_COLUMNS = ("path", "true_label", "painting_id", "predicted", "probability", "status", "painting_score")


@dataclass(frozen=True)
class Photo:
    path: Path
    label: str  # one of VALID_LABELS
    painting_id: str  # cluster key: photos with the same id show the same painting


# ------------------------------------------------------------------------------------------------
# Finding and opening the photos
# ------------------------------------------------------------------------------------------------


def normalise_label(name: str) -> str:
    """'Religious painting' or 'religious-painting' -> 'religious_painting'."""
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def painting_key(path: Path, label: str, root: Path | None = None) -> str:
    """Cluster key of a photo: `<paintingid>__<shot>.jpg` -> `label/paintingid`; any other name is its own cluster.

    Without the naming convention every photo counts as a different painting, so intervals are too narrow if
    several shots show the same one: that is why the protocol asks for the convention. The key of an unnamed photo
    is its path relative to `root`, so nothing about the disk it lives on is written out.
    """
    path = Path(path)
    if SHOT_SEPARATOR in path.stem:
        return f"{label}/{path.stem.split(SHOT_SEPARATOR, 1)[0]}"
    return str(path.relative_to(root)) if root is not None else str(path)


def scan_photo_dir(photo_dir: Path | str) -> tuple[list[Photo], list[str]]:
    """The photos of a folder with one sub-folder per label, and notes about what was ignored."""
    root = Path(photo_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} is not a folder: expected one sub-folder per label ({', '.join(VALID_LABELS)}).")
    photos: list[Photo] = []
    notes: list[str] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        label = normalise_label(sub.name)
        if label not in VALID_LABELS:
            notes.append(f"ignored folder {sub.name!r}: not one of {', '.join(VALID_LABELS)}")
            continue
        files = sorted(f for f in sub.rglob("*") if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            notes.append(f"folder {sub.name!r} has no photo")
        photos += [Photo(f, label, painting_key(f, label, root)) for f in files]
    if not photos:
        raise FileNotFoundError(f"no photo found in {root}")
    return photos, notes


def _register_heif() -> None:
    try:  # optional: `pip install pillow-heif` lets PIL open iPhone HEIC photos
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass


def open_photo(path: Path | str, max_side: int = 1024) -> Image.Image:
    """Open a photo as RGB with its EXIF rotation; large JPEGs are decoded at reduced size to save time.

    `max_side` (0 = off) is the longest side the photo is brought down to: CLIP and the ResNet shrink it to 128-224 px
    anyway, and a 12-megapixel phone photo takes about 30 ms to decode in full (see the kiosk budget). Raises OSError
    for a file PIL cannot read.
    """
    _register_heif()
    with Image.open(path) as img:
        if max_side:
            img.draft("RGB", (max_side, max_side))  # JPEG only: decodes at 1/2, 1/4 or 1/8 scale, never below max_side
        out = ImageOps.exif_transpose(img).convert("RGB")
    if max_side and max(out.size) > max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


# ------------------------------------------------------------------------------------------------
# The guide's decisions
# ------------------------------------------------------------------------------------------------


def decide(top_probability: float, min_confidence: float | None = None, rejected: bool = False) -> str:
    """What the guide does with a photo: refuse it as a non-painting, say 'unsure', or answer.

    Same order as `scripts/predict.py`: a rejected photo is not a painting whatever its probability; otherwise the
    answer is 'unsure' below `min_confidence` (None: always answer).
    """
    if rejected:
        return REJECTED
    if min_confidence is not None and top_probability < min_confidence:
        return UNSURE
    return ANSWERED


# images -> (the top-k [(class, probability), ...] of each image, painting scores or None)
PredictBatch = Callable[[Sequence[Image.Image]], tuple]


def run_photos(
    photos: Sequence[Photo],
    predict_batch: PredictBatch,
    *,
    min_confidence: float | None = None,
    rule=None,
    max_side: int = 1024,
    batch_size: int = 16,
    progress: Callable[[int, int], None] | None = None,
    root: Path | str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Run photos through the guide. Returns (one record per photo, the photos that could not be read).

    `predict_batch(images)` returns the top-k list `[(class, probability), ...]` of each image (best first) and, when a
    non-painting rule is in use, one painting score per image (else None). `rule` is a `museum_guide.ood.NonPaintingRule`.
    With `root`, paths in the records are relative to it (nothing about where the photos live on a disk is written out).
    """

    def shown(path: Path) -> str:
        try:
            return str(Path(path).relative_to(root)) if root is not None else str(path)
        except ValueError:
            return str(path)

    records: list[dict] = []
    skipped: list[dict] = []
    for start in range(0, len(photos), batch_size):
        chunk: list[Photo] = []
        images: list[Image.Image] = []
        for photo in photos[start : start + batch_size]:
            try:
                images.append(open_photo(photo.path, max_side))
                chunk.append(photo)
            except (OSError, ValueError, SyntaxError) as err:
                skipped.append({"path": shown(photo.path), "reason": f"{type(err).__name__}: {err}"})
        if images:
            tops, scores = predict_batch(images)
            for i, (photo, top) in enumerate(zip(chunk, tops)):
                score = None if scores is None else float(scores[i])
                rejected = rule is not None and score is not None and not bool(rule.accepts(score))
                name, probability = top[0]
                records.append(
                    {
                        "path": shown(photo.path),
                        "true_label": photo.label,
                        "painting_id": photo.painting_id,
                        "predicted": name,
                        "probability": float(probability),
                        "status": decide(probability, min_confidence, rejected),
                        "painting_score": score,
                    }
                )
        if progress:
            progress(min(start + batch_size, len(photos)), len(photos))
    return records, skipped


# ------------------------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------------------------


def _interval(correct: Sequence[bool], groups: Sequence[str], n_boot: int, seed: int) -> dict | None:
    if len(correct) == 0:
        return None
    acc, low, high = bootstrap_accuracy(np.asarray(correct, dtype=int), np.asarray(groups), n_boot=n_boot, seed=seed)
    return {"accuracy": acc, "low": low, "high": high, "n_photos": len(correct), "n_paintings": len(set(groups))}


def evaluate(
    records: Sequence[dict],
    classes: Sequence[str] = CLASSES,
    *,
    rejection_enabled: bool = False,
    min_confidence: float | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Summarise per-photo records (see `run_photos`) into the numbers the README reports.

    Genre photos: coverage (share the guide answers), accuracy on the photos it answers, accuracy of the forced top-1
    guess on all of them, per-class precision/recall/F1 and a confusion matrix (forced guesses). Non-paintings: how
    often the guide answers, says 'unsure' or refuses. Intervals are 95% percentile bootstraps that resample
    paintings (the ids of `Photo.painting_id`).
    """
    classes = list(classes)
    genre = [r for r in records if r["true_label"] in classes]
    other = [r for r in records if r["true_label"] == NOT_A_PAINTING]
    summary: dict = {
        "n_photos": len(records),
        "n_genre_photos": len(genre),
        "n_non_painting_photos": len(other),
        "rejection_enabled": rejection_enabled,
        "min_confidence": min_confidence,
        "interval_method": f"percentile bootstrap, {n_boot} resamples of paintings (photos with the same painting id are one cluster)",
    }
    if genre:
        answered = [r for r in genre if r["status"] == ANSWERED]
        n = len(genre)
        summary["genre"] = {
            "coverage": len(answered) / n,
            "unsure_rate": sum(r["status"] == UNSURE for r in genre) / n,
            "wrongly_rejected_rate": sum(r["status"] == REJECTED for r in genre) / n,
            "accuracy_on_answered": _interval(
                [r["predicted"] == r["true_label"] for r in answered], [r["painting_id"] for r in answered], n_boot, seed
            ),
            "accuracy_forced_top1": _interval(
                [r["predicted"] == r["true_label"] for r in genre], [r["painting_id"] for r in genre], n_boot, seed
            ),
        }
        y = np.array([classes.index(r["true_label"]) for r in genre])
        pred = np.array([classes.index(r["predicted"]) if r["predicted"] in classes else -1 for r in genre])
        summary["genre"]["per_class_forced_top1"] = per_class_report(y, pred, classes)
        summary["genre"]["confusion_forced_top1"] = {
            "labels": classes,
            "counts": confusion_matrix(y, pred, labels=range(len(classes))).tolist(),
            "note": "rows: true genre, columns: guessed genre, every genre photo counted (no 'unsure', no rejection)",
        }
    if other:
        m = len(other)
        counts = {s: sum(r["status"] == s for r in other) for s in (ANSWERED, UNSURE, REJECTED)}
        summary["non_paintings"] = {
            "answered_rate": counts[ANSWERED] / m,
            "unsure_rate": counts[UNSURE] / m,
            "rejected_rate": counts[REJECTED] / m,
            # With rejection off no photo can be refused, so the false-accept rate is only meaningful when it is on.
            "false_accept_rate": (counts[ANSWERED] + counts[UNSURE]) / m if rejection_enabled else None,
            "confidently_answered_interval": _interval(
                [r["status"] == ANSWERED for r in other], [r["painting_id"] for r in other], n_boot, seed
            ),
        }
    return summary


def format_summary(summary: dict, skipped: Sequence[dict] = (), notes: Sequence[str] = ()) -> str:
    """The summary as plain text for the terminal."""

    def pct(x):
        return "n/a" if x is None else f"{100 * x:.1f}%"

    def interval(d):
        if d is None:
            return "n/a (no photo)"
        return f"{pct(d['accuracy'])} [{pct(d['low'])}, {pct(d['high'])}] ({d['n_photos']} photos of {d['n_paintings']} paintings)"

    lines = [f"{summary['n_photos']} photos read ({summary['n_genre_photos']} paintings, {summary['n_non_painting_photos']} non-paintings)"]
    lines += [f"note: {n}" for n in notes]
    if skipped:
        lines.append(f"{len(skipped)} photo(s) skipped (unreadable), the first: {skipped[0]['path']} ({skipped[0]['reason']})")
    mc = summary["min_confidence"]
    lines.append(
        f"rules: answer 'unsure' below {mc:.2f}" if mc is not None else "rules: always answer (no confidence threshold)"
    )
    lines[-1] += "; refuse non-paintings" if summary["rejection_enabled"] else "; no refusal of non-paintings"
    g = summary.get("genre")
    if g:
        lines += [
            "paintings:",
            f"  answered {pct(g['coverage'])}, unsure {pct(g['unsure_rate'])}, wrongly refused {pct(g['wrongly_rejected_rate'])}",
            f"  accuracy on the photos it answers: {interval(g['accuracy_on_answered'])}",
            f"  accuracy of the best guess on every photo: {interval(g['accuracy_forced_top1'])}",
            f"  macro-F1 of the best guess: {g['per_class_forced_top1']['macro_f1']:.3f}",
        ]
    o = summary.get("non_paintings")
    if o:
        far = o["false_accept_rate"]
        lines += [
            "non-paintings:",
            f"  answered {pct(o['answered_rate'])}, unsure {pct(o['unsure_rate'])}, refused {pct(o['rejected_rate'])}"
            + ("" if far is None else f"  -> false-accept rate {pct(far)}"),
        ]
    return "\n".join(lines)


def write_records(records: Sequence[dict], path: Path | str) -> None:
    """The per-photo table as a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow({k: ("" if r.get(k) is None else r[k]) for k in CSV_COLUMNS})


def plot_confusion(summary: dict, path: Path | str, title: str = "Real photos: best guess for every photo") -> None:
    """Confusion matrix of the best guesses (counts) as a figure; nothing is drawn without genre photos."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conf = summary.get("genre", {}).get("confusion_forced_top1")
    if conf is None:
        return
    counts, labels = np.asarray(conf["counts"]), conf["labels"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(counts, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", color="white" if counts[i, j] > counts.max() / 2 else "black")
    ax.set_xlabel("guessed genre")
    ax.set_ylabel("true genre")
    ax.set_title(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# Demo set: perturbed dataset images, NOT real photos
# ------------------------------------------------------------------------------------------------


def build_demo_set(out_dir: Path | str, data_dir: Path | str, per_class: int = 6, seed: int = 0) -> Path:
    """Write a small fake photo folder so the harness can be tried without real photos.

    Paintings: images of the seed-`seed` TEST split (the saved CLIP head never saw them) damaged by camera-like
    perturbations and named `<imageid>__<perturbation>.jpg`; non-paintings: a few synthetic pictures (grey wall, black
    frame, noise, soft gradient). The numbers from such a set are NOT results on real photos.
    """
    from museum_guide.data import load_metadata, split
    from museum_guide.perturb import PERTURBATIONS

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    df = load_metadata(data_dir)
    test = split(df, seed)["test"]
    names: list[str] = [n for n in ("rotation_15deg", "blur", "perspective", "darker") if n in PERTURBATIONS]
    for label in CLASSES:
        rows = test[test["label"] == label].head(per_class)
        (out_dir / label).mkdir(parents=True, exist_ok=True)
        for k, (_, row) in enumerate(rows.iterrows()):
            name = names[k % len(names)]
            with Image.open(data_dir / row["path"]) as img:
                damaged = PERTURBATIONS[name](img.convert("RGB"))
            damaged.save(out_dir / label / f"{row['id']}{SHOT_SEPARATOR}{name}.jpg", quality=92)
    rng = np.random.default_rng(seed)
    gradient = np.tile(np.linspace(90, 160, 224, dtype=np.uint8)[None, :, None], (224, 1, 3))
    fakes = {
        "grey_wall": np.full((224, 224, 3), 128, dtype=np.uint8),
        "black_frame": np.zeros((224, 224, 3), dtype=np.uint8),
        "noise": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "soft_gradient": gradient,
    }
    (out_dir / NOT_A_PAINTING).mkdir(parents=True, exist_ok=True)
    for name, arr in fakes.items():
        Image.fromarray(arr).save(out_dir / NOT_A_PAINTING / f"{name}{SHOT_SEPARATOR}synthetic.png")
    return out_dir

