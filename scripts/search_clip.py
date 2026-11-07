"""Natural-language search over the collection with a pretrained CLIP dual encoder.

Images and questions are mapped to the same vector space; the answer is the k paintings whose
vectors are closest to the question's. Image vectors are computed once, offline, so a query costs
one text encoding plus a dot product.

Evaluation: with no hand-labelled questions, we use the genres as relevance labels. For each genre,
a visitor-style question ("I would like to see portraits") must return paintings of that genre.

Writes `reports/search_clip.json`, and `search_examples.png` under `data/figures_local/` (it shows paintings, which are
not published: see `museum_guide.figures`).
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from museum_guide.data import DATA_DIR, load_metadata
from museum_guide.figures import painting_figure_path
from museum_guide.train import REPORTS

MODEL = "openai/clip-vit-base-patch32"
K = 10
GENRE_QUERIES = {
    "cityscape": "I would like to see views of a city with streets and buildings",
    "landscape": "a painting of nature, fields, trees and mountains",
    "portrait": "I would like to see portraits of people",
    "religious_painting": "paintings showing saints, angels or scenes from the Bible",
    "still_life": "a still life with fruit, flowers and objects on a table",
}
FREE_QUERIES = [
    "Which painting shows a stormy sea?",
    "portraits of children",
    "a snowy winter village",
    "the crucifixion of Christ",
    "a vase of flowers",
]


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    model, proc = CLIPModel.from_pretrained(MODEL).eval(), CLIPProcessor.from_pretrained(MODEL)
    df = load_metadata()

    def embed_text(texts):
        out = model.get_text_features(**proc(text=texts, return_tensors="pt", padding=True))
        out = out if isinstance(out, torch.Tensor) else out.pooler_output
        return torch.nn.functional.normalize(out, dim=-1)

    vecs = []
    for i in range(0, len(df), 64):
        imgs = [Image.open(DATA_DIR / p).convert("RGB") for p in df["path"].iloc[i : i + 64]]
        out = model.get_image_features(**proc(images=imgs, return_tensors="pt"))
        out = out if isinstance(out, torch.Tensor) else out.pooler_output
        vecs.append(torch.nn.functional.normalize(out, dim=-1))
    collection = torch.cat(vecs)

    results = {"model": MODEL, "collection_size": len(df), "genre_queries": {}}
    for genre, q in GENRE_QUERIES.items():
        top = (collection @ embed_text([q]).T).squeeze(1).topk(K).indices.numpy()
        results["genre_queries"][genre] = {"query": q, f"precision@{K}": float((df["label"].values[top] == genre).mean())}
    results[f"mean_precision@{K}"] = float(np.mean([v[f"precision@{K}"] for v in results["genre_queries"].values()]))

    times = []
    for _ in range(20):
        t = time.perf_counter()
        (collection @ embed_text([FREE_QUERIES[0]]).T).squeeze(1).topk(K)
        times.append((time.perf_counter() - t) * 1000)
    results["cpu_latency_ms_per_query"] = float(np.median(times))
    print(json.dumps({k: v for k, v in results.items() if k != "genre_queries"}, indent=2))

    fig, axes = plt.subplots(len(FREE_QUERIES), 5, figsize=(11, 2.4 * len(FREE_QUERIES)))
    for row, q in enumerate(FREE_QUERIES):
        top = (collection @ embed_text([q]).T).squeeze(1).topk(5).indices.numpy()
        for col, idx in enumerate(top):
            ax = axes[row, col]
            ax.imshow(Image.open(DATA_DIR / df["path"].iloc[idx]))
            ax.axis("off")
            if col == 0:
                ax.set_title(q, loc="left", fontsize=10)
    fig.tight_layout()
    # The figure shows paintings: it goes to data/figures_local/ (not versioned), see museum_guide.figures.
    figure = painting_figure_path("search_examples.png")
    fig.savefig(figure, dpi=110)
    print(f"figure with paintings saved to {figure} (not versioned)")
    (REPORTS / "search_clip.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    import argparse

    # This script takes no options. Parsing the command line gives `--help` (the docstring above) and rejects
    # typos, instead of silently running a long experiment.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
