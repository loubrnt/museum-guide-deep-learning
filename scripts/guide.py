"""The guide in one command: genre, story, similar paintings and, if you ask, the paintings that answer a question.

    python scripts/guide.py photo.jpg
    python scripts/guide.py photo.jpg --ask "a stormy sea" --k 3
    python scripts/guide.py photo.jpg --lang en --json
    python scripts/guide.py photo.jpg --reject-non-paintings      # "this does not look like a painting" for a wall, a face...

One CLIP model does it all: its image vector gives the genre (a linear head, `reports/clip_probe.npz`), the
nearest paintings of the collection and the answers to the question. The 1,500 collection vectors are computed
once and saved (`data/search_index.npz`, built on first use, about 30 s), so a photo costs one image encoding and
a few dot products. The first call downloads CLIP's weights (about 600 MB).

The guide answers "unsure" (and tells no story) below the confidence saved with the head: the threshold that
reaches the target accuracy on the validation set (see `scripts/train_clip_probe.py`). Use --min-confidence to change it.
"""
import argparse
import json
import sys
from pathlib import Path

from museum_guide.clip import load_encoder
from museum_guide.data import DATA_DIR
from museum_guide.guide import build_answer, format_text
from museum_guide.notices import LANGS
from museum_guide.ood import DEFAULT_RULE, load_rule
from museum_guide.predict import DEFAULT_PROBE, load_probe, open_image
from museum_guide.search import load_index, load_or_build_index


def main(argv=None, encoder=None, index=None) -> int:
    """`encoder` and `index` are for tests: they replace the CLIP model and the saved collection index."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("photo", help="image file of the painting the visitor points at")
    p.add_argument("--ask", help="a question in natural language: also return the paintings that best answer it")
    p.add_argument("--k", type=int, default=5, help="number of paintings to return for the photo and for the question")
    p.add_argument("--lang", choices=LANGS, default="fr", help="language of the story")
    p.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    p.add_argument("--min-confidence", type=float, default=None,
                   help="answer 'unsure' below this probability (default: the threshold saved with the head; 0 = always answer)")
    p.add_argument("--temperature", type=float, default=None, help="override the temperature saved with the head")
    p.add_argument("--reject-non-paintings", action="store_true",
                   help="say 'this does not look like a painting' for a wall, a face or an object (needs reports/non_painting_rule.json)")
    p.add_argument("--rule", default=DEFAULT_RULE, help="saved rule for --reject-non-paintings")
    p.add_argument("--probe", default=DEFAULT_PROBE, help="saved CLIP head")
    p.add_argument("--index", default=DATA_DIR / "search_index.npz", help="saved collection index")
    args = p.parse_args(argv)

    try:
        probe = load_probe(args.probe)
        if index is None:
            index = _load_index(Path(args.index))
        if index.model != probe.model:
            print(f"the index was built with {index.model} but the head expects {probe.model}: "
                  "rebuild the index with `python scripts/search.py --rebuild \"x\"`.", file=sys.stderr)
            return 1
        photo = open_image(args.photo)
        rule = load_rule(args.rule) if args.reject_non_paintings else None
    except OSError as err:  # includes a missing probe, index or photo
        print(err, file=sys.stderr)
        return 1
    encoder = encoder or load_encoder(probe.model)

    threshold = probe.min_confidence if args.min_confidence is None else args.min_confidence
    answer = build_answer(
        photo, encoder, probe, index, lang=args.lang, k=args.k, ask=args.ask, min_confidence=threshold,
        temperature=args.temperature, photo_path=args.photo, data_dir=DATA_DIR,
        reject_rule=rule,
    )
    print(json.dumps({args.photo: answer}, indent=2, ensure_ascii=False) if args.json else format_text(answer, args.photo))
    return 0


def _load_index(path: Path):
    """The saved index; built once from the data if it is missing, with a clear error if there is no data either."""
    if path.exists():
        return load_index(path)
    from museum_guide.data import load_metadata

    if not (DATA_DIR / "metadata.csv").exists():
        raise FileNotFoundError(
            f"{path} not found and no data in {DATA_DIR}: build the data first (README, section Data), "
            "then run `python scripts/search.py \"a snowy village\"` once to build the index."
        )
    print(f"building the search index, once (about 30 s): {path}", file=sys.stderr)
    return load_or_build_index(load_metadata(), DATA_DIR, path, cache_file=DATA_DIR / "clip_embeddings.npz")


if __name__ == "__main__":
    sys.exit(main())
