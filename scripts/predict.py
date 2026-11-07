"""Classify photos of paintings: genre and confidence, from the fine-tuned ResNet-18 or from CLIP.

    python scripts/predict.py photo1.jpg photo2.jpg
    python scripts/predict.py photo.jpg --model clip             # CLIP image encoder + linear head (more accurate)
    python scripts/predict.py data/wikiart_genre5/portrait/*.jpg --top-k 2 --json
    python scripts/predict.py photo.jpg --min-confidence 0.6     # "unsure" below 60% confidence
    python scripts/predict.py photo.jpg --story                  # add the story of the genre (French)
    python scripts/predict.py photo.jpg --story --lang en        # ... in English
    python scripts/predict.py photo.jpg --reject-non-paintings   # "this does not look like a painting" for a wall, a face...

The default model (`--model resnet`) needs `reports/resnet18.pt`, written by `python scripts/train_resnet.py`. If
`reports/calibration.json` (from `python scripts/calibrate_resnet.py`) exists next to the weights, the probabilities
are calibrated with it. `--model clip` needs `reports/clip_probe.npz` (versioned; refit with
`python scripts/train_clip_probe.py`) and downloads CLIP's weights on first use (about 600 MB); its probabilities
are calibrated with the temperature saved in that file.

`--reject-non-paintings` (off by default) first asks CLIP whether the photo looks like a painting at all, with the rule in
`reports/non_painting_rule.json` (written by `python scripts/non_painting.py`). It needs CLIP even with `--model resnet`
(about 600 MB on first use) and the saved collection index `data/search_index.npz` (built by the first `scripts/search.py`).
"""
import argparse
import json
import sys
from pathlib import Path

from museum_guide.calibration import load_temperature
from museum_guide.notices import LANGS, notice_for, story_entry
from museum_guide.clip import load_encoder
from museum_guide.data import DATA_DIR
from museum_guide.ood import DEFAULT_RULE, load_rule, painting_scores
from museum_guide.predict import (
    CLASSES,
    DEFAULT_PROBE,
    DEFAULT_WEIGHTS,
    classify,
    classify_clip,
    classify_vectors,
    load_classifier,
    load_probe,
    open_image,
)
from museum_guide.search import load_index


def main(argv=None, encoder=None) -> int:
    """`encoder` (a `museum_guide.clip.ClipEncoder`, or a fake with the same `images` method) is for tests: it replaces CLIP."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="image files to classify")
    p.add_argument("--model", choices=("resnet", "clip"), default="resnet", help="resnet: fine-tuned ResNet-18; clip: CLIP image encoder + linear head")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="trained ResNet-18 weights (--model resnet)")
    p.add_argument("--probe", default=DEFAULT_PROBE, help="saved CLIP head (--model clip)")
    p.add_argument("--top-k", type=int, default=3, help="number of classes to show per image")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    p.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="divide the logits by this before the softmax (default: the value in calibration.json next to the ResNet weights, else 1; the value saved in the CLIP head)",
    )
    p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="answer 'unsure' when the top probability is below this value (default: always answer)",
    )
    p.add_argument(
        "--story",
        action="store_true",
        help="after the genre, add a short notice on that tradition (draft texts, see museum_guide.notices)",
    )
    p.add_argument("--lang", choices=LANGS, default=None, help="language of the story (with --story; default: fr)")
    p.add_argument(
        "--reject-non-paintings",
        action="store_true",
        help="answer 'this does not look like a painting' for a wall, a face, an object... (loads CLIP; off by default)",
    )
    p.add_argument("--rule", default=DEFAULT_RULE, help="saved rule for --reject-non-paintings")
    p.add_argument("--index", default=DATA_DIR / "search_index.npz", help="saved collection index (for --reject-non-paintings)")
    args = p.parse_args(argv)
    if args.lang is not None and not args.story:
        p.error("--lang only applies together with --story")
    lang = args.lang or "fr"

    try:
        if args.model == "clip":
            probe = load_probe(args.probe)
            encoder = encoder or load_encoder(probe.model)
        else:
            model = load_classifier(args.weights)
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
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1
    temperature = args.temperature
    if temperature is None and args.model == "resnet":
        temperature = load_temperature(Path(args.weights).with_name("calibration.json"))
    elif temperature is None:
        temperature = probe.temperature

    results, judged, failed = {}, {}, False
    for path in args.images:
        try:
            photo = open_image(path)
            vector = None
            if rule is not None:
                vector = encoder.images([photo])
                judged[path] = float(painting_scores(rule, encoder, vector, reference)[0])
            if args.model == "clip" and vector is not None:  # the photo is already embedded: do not run CLIP twice
                results[path] = classify_vectors(probe, vector, args.top_k, temperature)[0]
            elif args.model == "clip":
                results[path] = classify_clip(encoder, probe, [photo], args.top_k, temperature)[0]
            else:
                results[path] = classify(model, [photo], CLASSES, args.top_k, temperature)[0]
        except OSError as err:  # unreadable or not an image: report it and go on with the others
            print(f"{path}: cannot read image ({err})", file=sys.stderr)
            failed = True

    def unsure(top):
        return args.min_confidence is not None and top[0][1] < args.min_confidence

    def not_a_painting(path):
        return rule is not None and not rule.accepts(judged[path])

    if args.json:
        def entry(top):
            preds = [{"class": c, "probability": round(pr, 4)} for c, pr in top]
            # The shape only changes when a flag is given, so existing consumers keep working.
            if not args.story:
                return preds if args.min_confidence is None else {"unsure": unsure(top), "predictions": preds}
            out = {"predictions": preds}
            if args.min_confidence is not None:
                out = {"unsure": unsure(top), **out}
            # No story for a guess the guide is not sure about.
            out["story"] = None if unsure(top) else story_entry(top[0][0], lang)
            return out

        def with_painting_check(path, top):
            base = entry(top)
            if rule is None:
                return base
            base = {"predictions": base} if isinstance(base, list) else dict(base)
            if not_a_painting(path) and "story" in base:
                base["story"] = None  # no story for something that is not a painting
            return {"not_a_painting": bool(not_a_painting(path)), "painting_score": round(judged[path], 4), **base}

        print(json.dumps({k: with_painting_check(k, v) for k, v in results.items()}, indent=2))
    else:
        for path, top in results.items():
            if not_a_painting(path):
                print(f"{path}: this does not look like a painting (score {judged[path]:.2f}, threshold {rule.threshold:.2f})")
            elif unsure(top):
                print(f"{path}: unsure (best guess: {top[0][0]} {top[0][1]:.0%})")
            else:
                print(f"{path}: " + ", ".join(f"{c} {pr:.0%}" for c, pr in top))
                if args.story:
                    print(f"  {notice_for(top[0][0], lang)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
