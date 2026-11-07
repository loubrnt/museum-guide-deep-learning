"""Classify photos of paintings: genre and confidence, from the fine-tuned ResNet-18.

    python scripts/predict.py photo1.jpg photo2.jpg
    python scripts/predict.py data/wikiart_genre5/portrait/*.jpg --top-k 2 --json

Needs `reports/resnet18.pt`, written by `python scripts/train_resnet.py`.
"""
import argparse
import json
import sys

from museum_guide.predict import CLASSES, DEFAULT_WEIGHTS, classify, load_classifier, open_image


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+", help="image files to classify")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="trained ResNet-18 weights")
    p.add_argument("--top-k", type=int, default=3, help="number of classes to show per image")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    args = p.parse_args(argv)

    try:
        model = load_classifier(args.weights)
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1

    results, failed = {}, False
    for path in args.images:
        try:
            results[path] = classify(model, [open_image(path)], CLASSES, args.top_k)[0]
        except OSError as err:  # unreadable or not an image: report it and go on with the others
            print(f"{path}: cannot read image ({err})", file=sys.stderr)
            failed = True

    if args.json:
        print(json.dumps({k: [{"class": c, "probability": round(pr, 4)} for c, pr in v] for k, v in results.items()}, indent=2))
    else:
        for path, top in results.items():
            print(f"{path}: " + ", ".join(f"{c} {pr:.0%}" for c, pr in top))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
