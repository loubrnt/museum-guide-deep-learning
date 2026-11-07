"""Search the collection with a question in natural language.

    python scripts/search.py "a stormy sea"
    python scripts/search.py "paintings by Claude Monet" --k 8 --grid monet.png

The image vectors are computed once and saved in `data/search_index.npz` (not versioned); the first call builds
the index (about 30 s on CPU), later calls only encode the question. Use --rebuild after changing the data.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from museum_guide.clip import embed_texts, load_clip
from museum_guide.data import DATA_DIR, load_metadata
from museum_guide.search import load_or_build_index, search


def save_grid(results, path):
    fig, axes = plt.subplots(1, len(results), figsize=(2.6 * len(results), 3.2), squeeze=False)
    for ax, (_, r) in zip(axes[0], results.iterrows()):
        with Image.open(DATA_DIR / r["path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"#{r['rank']}  {r['score']:.3f}\n{r['label']}, {r['style']}", fontsize=7)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", help="the question, e.g. 'a snowy winter village'")
    p.add_argument("--k", type=int, default=5, help="number of paintings to return")
    p.add_argument("--grid", help="also save an image grid of the results to this file")
    p.add_argument("--rebuild", action="store_true", help="recompute the image vectors")
    p.add_argument("--index", default=DATA_DIR / "search_index.npz", help="where the index is saved")
    args = p.parse_args(argv)

    index = load_or_build_index(
        load_metadata(), DATA_DIR, args.index, args.rebuild, cache_file=DATA_DIR / "clip_embeddings.npz"
    )
    model, processor = load_clip(index.model)
    results = search(index, embed_texts(model, processor, [args.query])[0], args.k)
    for _, r in results.iterrows():
        print(f"{r['rank']:2d}  {r['score']:.3f}  {r['path']}  [{r['label']}, {r['style']}, {r['artist']}]")
    if args.grid:
        save_grid(results, args.grid)
        print(f"grid saved to {args.grid}")


if __name__ == "__main__":
    main()
