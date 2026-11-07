"""Measure the natural-language search on a query set generated from the WikiArt metadata.

Relevance is objective: a painting is relevant to "a Cubist still life" if its style is a Cubism label and its
genre is still_life; to "paintings by Claude Monet" if its artist is claude-monet. Every query has at least 10
relevant paintings, so precision@10 can reach 1.0. We report precision@10, average precision and nDCG@10 next to
the score of a random ranking (the base rate), overall and by query type, for two phrasings of each question.

    python scripts/search_eval.py

Writes `reports/search_queries.json` (the queries and their relevance rules), `reports/search_eval.json` and
`reports/figures/search_eval.png`. This measures what the metadata knows (genre, style, artist), not fine content
such as "a stormy sea".
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from museum_guide.clip import embed_texts, load_clip
from museum_guide.data import DATA_DIR, load_metadata
from museum_guide.search import (
    average_precision,
    build_query_set,
    load_or_build_index,
    ndcg_at_k,
    precision_at_k,
    random_baseline,
    rank_all,
    relevance_in_rank_order,
)
from museum_guide.train import REPORTS

K = 10
VARIANTS = {"bare": "{q}", "prefixed": "a painting: {q}"}  # the question as written, and with a short prefix
TYPES = ("genre", "style_genre", "artist")
METRICS = (f"precision@{K}", "average_precision", f"ndcg@{K}")


def evaluate(index, model, processor, queries, template):
    texts = [template.format(q=q["text"]) for q in queries]
    emb = embed_texts(model, processor, texts)
    rows = []
    for q, e in zip(queries, emb):
        rel = relevance_in_rank_order(rank_all(index, e), q["relevant"], len(index))
        rows.append(
            {
                "id": q["id"], "type": q["type"], "text": q["text"], "n_relevant": len(q["relevant"]),
                f"precision@{K}": precision_at_k(rel, K), "average_precision": average_precision(rel), f"ndcg@{K}": ndcg_at_k(rel, K),
            }
        )
    return rows


def rounded(rows):
    return [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]


def summarise(rows, chance):
    out = {}
    for name in ("all", *TYPES):
        sel = [i for i, r in enumerate(rows) if name == "all" or r["type"] == name]
        out[name] = {
            "n_queries": len(sel),
            **{m: float(np.mean([rows[i][m] for i in sel])) for m in METRICS},
            "chance": {m: float(np.mean([chance[i][m] for i in sel])) for m in METRICS},
            "queries_with_precision_0": int(sum(rows[i][f"precision@{K}"] == 0 for i in sel)),
        }
    return out


def plot(summary, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ("all", *TYPES)
    x = np.arange(len(names))
    for ax, metric, title in zip(axes, (f"precision@{K}", "average_precision"), (f"precision@{K}", "average precision")):
        for j, (label, series) in enumerate(
            [(v, [summary[v][n][metric] for n in names]) for v in VARIANTS]
            + [("random ranking", [summary["bare"][n]["chance"][metric] for n in names])]
        ):
            ax.bar(x + (j - 1) * 0.27, series, 0.27, label=label, color=("tab:blue", "tab:cyan", "tab:gray")[j])
        ax.set_xticks(x, [f"{n}\n({summary['bare'][n]['n_queries']} queries)" for n in names], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    df = load_metadata()
    index = load_or_build_index(df, DATA_DIR, DATA_DIR / "search_index.npz", cache_file=DATA_DIR / "clip_embeddings.npz")
    assert index.meta["id"].tolist() == df["id"].astype(str).tolist(), "index and metadata are out of sync: use --rebuild"
    queries = build_query_set(df)
    model, processor = load_clip(index.model)

    chance = [random_baseline(len(q["relevant"]), len(index), K) for q in queries]
    per_variant = {v: evaluate(index, model, processor, queries, t) for v, t in VARIANTS.items()}
    summary = {v: summarise(rows, chance) for v, rows in per_variant.items()}

    ranked = sorted(per_variant["bare"], key=lambda r: r["average_precision"])
    results = {
        "model": index.model, "n_queries": len(queries), "collection_size": len(index), "k": K,
        "variants": VARIANTS, "summary": summary, "per_query": per_variant,
        "per_query_chance": [{"id": q["id"], **c} for q, c in zip(queries, chance)],
        "lowest_average_precision_bare": rounded(ranked[:6]), "highest_average_precision_bare": rounded(ranked[-6:][::-1]),
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "search_queries.json").write_text(json.dumps(
        [{k: v for k, v in q.items() if k != "relevant"} | {"n_relevant": len(q["relevant"]), "base_rate": len(q["relevant"]) / len(index)} for q in queries],
        indent=2))
    (REPORTS / "search_eval.json").write_text(json.dumps(results, indent=2))
    plot(summary, REPORTS / "figures" / "search_eval.png")

    for v in VARIANTS:
        print(f"\nvariant '{v}' ({VARIANTS[v]!r}):")
        for name, s in summary[v].items():
            print(f"  {name:12s} n={s['n_queries']:2d}  P@{K} {s[f'precision@{K}']:.3f} (chance {s['chance'][f'precision@{K}']:.3f})  "
                  f"AP {s['average_precision']:.3f} (chance {s['chance']['average_precision']:.3f})  nDCG@{K} {s[f'ndcg@{K}']:.3f}  "
                  f"P@{K}=0 on {s['queries_with_precision_0']}")


if __name__ == "__main__":
    import argparse

    # This script takes no options. Parsing the command line gives `--help` (the docstring above) and rejects
    # typos, instead of silently running a long experiment.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
