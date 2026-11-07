"""Evaluate the guide on real photos taken by the team.

    python scripts/evaluate_photos.py my_photos/ --model clip
    python scripts/evaluate_photos.py my_photos/ --model clip --reject-non-paintings
    python scripts/evaluate_photos.py --demo                       # try it on perturbed dataset images (NOT real photos)

`my_photos/` has one sub-folder per label: cityscape, landscape, portrait, religious_painting, still_life, and
optionally not_a_painting. Name the files `<paintingid>__<shot>.jpg` so that several shots of one painting are
treated as one painting in the confidence intervals. The step-by-step protocol is in docs/REAL_PHOTO_PROTOCOL.md.

Each photo goes through the same decisions as the guide: answer, "unsure" (below the confidence threshold), or "this does
not look like a painting" (with --reject-non-paintings). The threshold defaults to the one saved with the CLIP head
(like scripts/guide.py) and to "always answer" for the ResNet (like scripts/predict.py); change it with --min-confidence,
or turn it off with --always-answer. `--model clip` needs reports/clip_probe.npz and downloads CLIP on first use
(about 600 MB); `--model resnet` needs reports/resnet18.pt (python scripts/train_resnet.py); `--reject-non-paintings`
also needs CLIP and the saved collection index (built by the first scripts/search.py), see scripts/predict.py.

Results go to --out (default data/photo_eval/, not versioned: photos may be copyrighted or show people):
predictions.csv (one row per photo), summary.json and confusion_matrix.png. Nothing is ever written to reports/.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

from museum_guide.calibration import load_temperature
from museum_guide.clip import load_encoder
from museum_guide.data import DATA_DIR
from museum_guide.ood import DEFAULT_RULE, load_rule, painting_scores
from museum_guide.photo_eval import (
    build_demo_set,
    evaluate,
    format_summary,
    plot_confusion,
    run_photos,
    scan_photo_dir,
    write_records,
)
from museum_guide.predict import CLASSES, DEFAULT_PROBE, DEFAULT_WEIGHTS, classify, classify_vectors, load_classifier, load_probe
from museum_guide.search import load_index

DEFAULT_OUT = Path("data/photo_eval")


def build_predictor(args, encoder=None):
    """(predict_batch, rule, min_confidence, temperature) for the chosen model, loading what is needed.

    `encoder` (a `museum_guide.clip.ClipEncoder`, or a fake with `images` and `texts`) is for tests: it replaces CLIP.
    """
    probe = model = None
    if args.model == "clip":
        probe = load_probe(args.probe)
        encoder = encoder or load_encoder(probe.model)
        temperature = probe.temperature if args.temperature is None else args.temperature
        default_threshold = probe.min_confidence  # the threshold chosen on validation data, as in scripts/guide.py
    else:
        model = load_classifier(args.weights)
        temperature = load_temperature(Path(args.weights).with_name("calibration.json")) if args.temperature is None else args.temperature
        default_threshold = None  # as in scripts/predict.py
    min_confidence = None if args.always_answer else (default_threshold if args.min_confidence is None else args.min_confidence)

    rule = reference = None
    if args.reject_non_paintings:
        rule = load_rule(args.rule)
        if encoder is None:
            print("--reject-non-paintings loads CLIP (about 600 MB on first use).", file=sys.stderr)
            encoder = load_encoder()
        if rule.needs_collection:
            if not Path(args.index).exists():
                raise FileNotFoundError(
                    f"{args.index} not found: build the collection index once with `python scripts/search.py \"a snowy village\"`."
                )
            reference = load_index(args.index).vectors

    def predict_batch(images):
        vectors = encoder.images(images) if encoder is not None else None
        if probe is not None:
            tops = classify_vectors(probe, vectors, 1, temperature)
        else:
            tops = classify(model, images, CLASSES, 1, temperature)
        scores = painting_scores(rule, encoder, vectors, reference) if rule is not None else None
        return tops, scores

    return predict_batch, rule, min_confidence, temperature


def main(argv=None, encoder=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photo_dir", nargs="?", help="folder with one sub-folder per label (not needed with --demo)")
    p.add_argument("--model", choices=("resnet", "clip"), default="clip", help="clip: CLIP image encoder + linear head (default); resnet: fine-tuned ResNet-18")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="trained ResNet-18 weights (--model resnet)")
    p.add_argument("--probe", default=DEFAULT_PROBE, help="saved CLIP head (--model clip)")
    p.add_argument("--temperature", type=float, default=None, help="divide the logits by this before the softmax (default: the saved calibration)")
    p.add_argument("--min-confidence", type=float, default=None, help="answer 'unsure' below this probability (default: the CLIP head's saved threshold; none for the ResNet)")
    p.add_argument("--always-answer", action="store_true", help="never say 'unsure' (ignores any threshold)")
    p.add_argument("--reject-non-paintings", action="store_true", help="refuse a wall, a face, an object... as in scripts/predict.py (loads CLIP)")
    p.add_argument("--rule", default=DEFAULT_RULE, help="saved rule for --reject-non-paintings")
    p.add_argument("--index", default=DATA_DIR / "search_index.npz", help="saved collection index (for --reject-non-paintings)")
    p.add_argument("--out", default=DEFAULT_OUT, help="where to write predictions.csv, summary.json and the figure")
    p.add_argument("--max-side", type=int, default=1024, help="longest side photos are brought down to before the models (0: full size, slower)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--demo", action="store_true", help="try the harness on perturbed dataset images and synthetic non-paintings: NOT real photos")
    p.add_argument("--demo-data", default=DATA_DIR, help="dataset folder used by --demo")
    args = p.parse_args(argv)
    if not args.demo and not args.photo_dir:
        p.error("give a photo folder, or use --demo")

    try:
        predict_batch, rule, min_confidence, temperature = build_predictor(args, encoder)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    demo_dir = None
    try:
        if args.demo:
            if not (Path(args.demo_data) / "metadata.csv").exists():
                print(f"{args.demo_data}/metadata.csv not found: --demo needs the dataset (see the README, section Data).", file=sys.stderr)
                return 1
            demo_dir = tempfile.TemporaryDirectory(prefix="museum_guide_demo_photos_")
            photo_dir = build_demo_set(demo_dir.name, args.demo_data)
            print("DEMO: perturbed dataset images and synthetic non-paintings. These numbers are NOT results on real photos.\n")
            out = Path(args.out) / "demo"
        else:
            photo_dir, out = Path(args.photo_dir), Path(args.out)
        try:
            photos, notes = scan_photo_dir(photo_dir)
        except FileNotFoundError as err:
            print(err, file=sys.stderr)
            return 1

        def progress(done, total):
            print(f"\r{done}/{total} photos", end="" if done < total else "\n", file=sys.stderr, flush=True)

        records, skipped = run_photos(
            photos, predict_batch, min_confidence=min_confidence, rule=rule, max_side=args.max_side, batch_size=args.batch_size, progress=progress,
            root=photo_dir,
        )
    finally:
        if demo_dir is not None:
            demo_dir.cleanup()
    if not records:
        print("no photo could be read.", file=sys.stderr)
        return 1

    summary = evaluate(records, rejection_enabled=rule is not None, min_confidence=min_confidence)
    summary.update(
        {
            "demo": args.demo,
            "model": args.model,
            "temperature": temperature,
            "max_side": args.max_side,
            "photo_folder_name": None if args.demo else Path(photo_dir).name,  # never the full path: it can contain a user name
            "notes": notes,
            "skipped": skipped,
        }
    )
    out.mkdir(parents=True, exist_ok=True)
    write_records(records, out / "predictions.csv")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_confusion(summary, out / "confusion_matrix.png", "DEMO (not real photos): best guess for every photo" if args.demo else "Real photos: best guess for every photo")
    print(format_summary(summary, skipped, notes))
    print(f"\nwritten to {out}/ (predictions.csv, summary.json, confusion_matrix.png)")
    if args.demo:
        print("Reminder: demo numbers are NOT real-photo results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
