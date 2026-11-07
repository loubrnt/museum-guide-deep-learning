"""Questions in French: what does the English-only CLIP lose, and does a multilingual text encoder fix it?

The search index holds the CLIP ViT-B/32 image vectors. Two text encoders can query it:

- `clip`: the original CLIP text encoder, trained mostly on English text;
- `multilingual`: `sentence-transformers/clip-ViT-B-32-multilingual-v1`, a student encoder trained to put French (and
  other languages) where CLIP puts the English translation. Its image side is the same CLIP, so the index is reused.

Each encoder is scored on the 66 English questions of `scripts/search_eval.py` and on their French translations
(`museum_guide.french`), which have the same ids and the same relevant paintings. A third French version keeps the
artists' names as spelled in the metadata ('Albrecht Durer'), to separate the effect of the language from the
spelling ('Albrecht Dürer'), and a fourth strips the accents of the French text ('Édouard Cortès' -> 'Edouard Cortes').
The 5 free questions, judged by eye in the README, are compared by how much the
top 10 changes between languages.

    pip install -e ".[multilingual]"
    python scripts/search_french.py

Writes `reports/search_french.json`, `reports/search_queries_fr.json` (the translations, to be read by a French
speaker) and `reports/figures/search_french.png`. The French texts were written by an AI assistant.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from search_eval import K, METRICS, TYPES, summarise

from museum_guide.clip import embed_texts, load_clip
from museum_guide.data import DATA_DIR, load_metadata
from museum_guide.french import FREE_QUERIES, french_query_set, strip_accents
from museum_guide.multilingual import MODEL as MULTILINGUAL_MODEL
from museum_guide.multilingual import embed_texts_multilingual, load_multilingual
from museum_guide.robustness import bootstrap_difference
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

CONDITIONS = {  # name -> (encoder, query set)
    "clip_en": ("clip", "en"),
    "clip_fr": ("clip", "fr"),
    "clip_fr_original_names": ("clip", "fr_original_names"),
    "clip_fr_no_accents": ("clip", "fr_no_accents"),
    "multilingual_en": ("multilingual", "en"),
    "multilingual_fr": ("multilingual", "fr"),
    "multilingual_fr_original_names": ("multilingual", "fr_original_names"),
}
COMPARISONS = [  # (a, b): average precision of a minus b, paired over the 66 questions
    ("clip_fr", "clip_en"),
    ("clip_fr_no_accents", "clip_fr"),
    ("clip_fr_no_accents", "clip_en"),
    ("multilingual_en", "clip_en"),
    ("multilingual_fr", "clip_en"),
    ("multilingual_fr", "clip_fr"),
    ("multilingual_fr", "multilingual_en"),
]
# What a free question should bring back, to read the top 10 (None: no genre label says it).
FREE_EXPECTED_GENRES = [None, {"portrait"}, {"landscape", "cityscape"}, {"religious_painting"}, {"still_life"}]


def evaluate(index, embed, queries):
    emb = embed([q["text"] for q in queries])
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


def free_question_report(index, embeds):
    """Top 10 of each free question in English and French, per encoder: overlap and share of the expected genre."""
    out = {}
    for encoder, embed in embeds.items():
        en_vecs = embed([en for en, _ in FREE_QUERIES])
        fr_vecs = embed([fr for _, fr in FREE_QUERIES])
        rows = []
        for (en, fr), ve, vf, expected in zip(FREE_QUERIES, en_vecs, fr_vecs, FREE_EXPECTED_GENRES):
            top_en, top_fr = rank_all(index, ve)[:10], rank_all(index, vf)[:10]
            row = {"english": en, "french": fr, "top10_overlap": len(set(top_en) & set(top_fr))}
            if expected:
                labels_en, labels_fr = index.meta["label"].to_numpy()[top_en], index.meta["label"].to_numpy()[top_fr]
                row["expected_genres"] = sorted(expected)
                row["share_expected_genre_en"] = float(np.isin(labels_en, list(expected)).mean())
                row["share_expected_genre_fr"] = float(np.isin(labels_fr, list(expected)).mean())
            rows.append(row)
        out[encoder] = rows
    return out


def plot(summaries, path):
    names = ("all", *TYPES)
    shown = ["clip_en", "clip_fr", "clip_fr_no_accents", "multilingual_en", "multilingual_fr"]
    colours = ["tab:blue", "tab:cyan", "tab:green", "tab:orange", "tab:red"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(names))
    for ax, metric, title in zip(axes, (f"precision@{K}", "average_precision"), (f"precision@{K}", "average precision")):
        for j, (cond, colour) in enumerate(zip(shown, colours)):
            ax.bar(x + (j - 2) * 0.16, [summaries[cond][n][metric] for n in names], 0.16, label=cond, color=colour)
        ax.plot(x, [summaries["clip_en"][n]["chance"][metric] for n in names], "k_", markersize=18, label="random ranking")
        ax.set_xticks(x, [f"{n}\n({summaries['clip_en'][n]['n_queries']} queries)" for n in names], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    df = load_metadata()
    index = load_or_build_index(df, DATA_DIR, DATA_DIR / "search_index.npz", cache_file=DATA_DIR / "clip_embeddings.npz")
    assert index.meta["id"].tolist() == df["id"].astype(str).tolist(), "index and metadata are out of sync"
    english = build_query_set(df)
    french = french_query_set(english)
    query_sets = {
        "en": english,
        "fr": french,
        "fr_original_names": french_query_set(english, original_names=True),
        "fr_no_accents": [{**q, "text": strip_accents(q["text"])} for q in french],
    }

    model, processor = load_clip(index.model)
    multilingual = load_multilingual()
    embeds = {
        "clip": lambda texts: embed_texts(model, processor, texts),
        "multilingual": lambda texts: embed_texts_multilingual(multilingual, texts),
    }

    chance = [random_baseline(len(q["relevant"]), len(index), K) for q in english]
    per_condition = {name: evaluate(index, embeds[enc], query_sets[qs]) for name, (enc, qs) in CONDITIONS.items()}
    summaries = {name: summarise(rows, chance) for name, rows in per_condition.items()}

    paired = {}
    for a, b in COMPARISONS:
        ap_a = np.array([r["average_precision"] for r in per_condition[a]])
        ap_b = np.array([r["average_precision"] for r in per_condition[b]])
        diff, low, high = bootstrap_difference(ap_a, ap_b)
        by_type = {}
        for kind in TYPES:
            sel = np.array([r["type"] == kind for r in per_condition[a]])
            d, lo, hi = bootstrap_difference(ap_a[sel], ap_b[sel])
            by_type[kind] = {"mean_average_precision_difference": d, "low": lo, "high": hi, "n_queries": int(sel.sum())}
        paired[f"{a}_minus_{b}"] = {
            "mean_average_precision_difference": diff, "low": low, "high": high,
            "queries_where_first_is_better": int((ap_a > ap_b).sum()), "queries_where_first_is_worse": int((ap_a < ap_b).sum()),
            "n_queries": len(ap_a), "by_type": by_type,
        }

    results = {
        "k": K, "n_queries": len(english), "collection_size": len(index), "clip_model": index.model,
        "multilingual_model": MULTILINGUAL_MODEL, "conditions": {n: {"encoder": e, "queries": q} for n, (e, q) in CONDITIONS.items()},
        "summary": summaries, "paired_average_precision": paired,
        "paired_method": "percentile bootstrap over the 66 questions (2000 resamples), 95% interval",
        "free_questions": free_question_report(index, embeds),
        "per_query": per_condition,
        "translations_note": "written by an AI assistant: to be read by a French speaker",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "search_french.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    (REPORTS / "search_queries_fr.json").write_text(json.dumps(
        [{"id": e["id"], "type": e["type"], "english": e["text"], "french": f["text"]} for e, f in zip(english, query_sets["fr"])],
        indent=2, ensure_ascii=False))
    plot(summaries, REPORTS / "figures" / "search_french.png")

    print(f"{'condition':32s} " + "  ".join(f"{n:>22s}" for n in ("all", *TYPES)) + "     (P@10 / AP)")
    for name, s in summaries.items():
        print(f"{name:32s} " + "  ".join(f"{s[n][f'precision@{K}']:.3f} / {s[n]['average_precision']:.3f}".rjust(22) for n in ("all", *TYPES)))
    print("\npaired differences of average precision (95% interval):")
    for k, v in paired.items():
        print(f"  {k:42s} {v['mean_average_precision_difference']:+.3f} [{v['low']:+.3f}, {v['high']:+.3f}]  "
              f"better on {v['queries_where_first_is_better']}, worse on {v['queries_where_first_is_worse']} of {v['n_queries']}")
        for kind, t in v["by_type"].items():
            print(f"      {kind:12s} {t['mean_average_precision_difference']:+.3f} [{t['low']:+.3f}, {t['high']:+.3f}] ({t['n_queries']} queries)")
    print("\nfree questions, overlap of the top 10 between English and French:")
    for enc, rows in results["free_questions"].items():
        print(f"  {enc}: " + ", ".join(f"{r['english'][:22]!r} {r['top10_overlap']}/10" for r in rows))
    assert all(m in summaries["clip_en"]["all"] for m in METRICS)


if __name__ == "__main__":
    import argparse

    # This script takes no options. Parsing the command line gives `--help` (the docstring above) and rejects
    # typos, instead of silently running a long experiment.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
