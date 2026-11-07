"""Does using the catalogue fields in the question help? What the evaluation can and cannot show.

The search can read the artist, style and genre a visitor typed (`museum_guide.catalogue`) and use them as filters,
leaving CLIP to rank what the catalogue cannot say. This script reports three things, and keeps them apart because
they do not prove the same thing:

(a) **The parser.** For the 66 questions of the metadata-labelled query set, in English and in French, does it recover
    exactly the artist, style and genre the question was generated from? Precision, recall and the failures are listed,
    including the French spellings that broke plain CLIP. A list of questions that name NO catalogue field shows
    whether it invents filters.
(b) **Retrieval on the metadata-defined queries.** The relevant paintings of those 66 questions are DEFINED by the
    same metadata, so a metadata filter solves them by construction. These numbers are an upper bound that shows the
    filter works as intended; they are not evidence that retrieval got better.
(c) **A small manual check** where the filter cannot cheat: questions that combine a catalogue part with a visual
    part the catalogue does not hold ("a Baroque religious painting with angels"). The top 5 of plain CLIP and of the
    hybrid search are looked at by an AI reviewer against a criterion written BEFORE looking; a human should confirm.

    python scripts/search_hybrid_eval.py --grids   # first: save the image grids to look at (data/hybrid_manual, not versioned)
    python scripts/search_hybrid_eval.py           # then: everything, reading the judgments in reports/search_hybrid_manual.json

Writes `reports/search_hybrid.json` and `reports/figures/search_hybrid.png`. The grids contain WikiArt paintings and
stay out of git.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from search_eval import K, summarise

from museum_guide.catalogue import Catalogue
from museum_guide.clip import embed_texts, load_clip
from museum_guide.data import DATA_DIR, load_metadata
from museum_guide.french import FREE_QUERIES, french_query_set
from museum_guide.search import (
    average_precision,
    build_query_set,
    hybrid_search,
    load_or_build_index,
    ndcg_at_k,
    precision_at_k,
    random_baseline,
    rank_all,
    rank_with_filter,
    relevance_in_rank_order,
    search,
)
from museum_guide.train import REPORTS

BOOST = 0.1  # soft boost tried in (b): fixed in advance, not tuned
MANUAL_FILE = REPORTS / "search_hybrid_manual.json"
GRID_DIR = DATA_DIR.parent / "hybrid_manual"

# Questions that name NO artist, style or genre of the catalogue: any filter found on them is a false alarm.
# They include traps: 'romantic' and 'naive' are ordinary words, 'gris' and 'chase' are surnames and common words.
NEUTRAL_QUERIES = [
    "a stormy sea", "a snowy winter village", "a vase of flowers", "the crucifixion of Christ", "a girl reading a book by the window",
    "a cat sleeping on a sofa", "a golden sunset over the water", "a romantic evening by the sea", "a naive question",
    "a gris sky", "a chase through the forest", "a ship in a storm", "children playing in a garden",
    "un village enneigé en hiver", "la mer agitée", "une scène de marché", "un ciel gris au-dessus de la mer",
    "Quel tableau montre un chat ?", "un coucher de soleil sur l'eau", "un enfant qui lit",
]

# Written before looking at any result. `visual` is the part the catalogue does not hold; it must be coarse enough to
# be seen in a 128-pixel thumbnail. The last question is a deliberate empty case: Renoir has only city views here.
MANUAL_QUERIES = [
    ("baroque_angels", "a Baroque religious painting with angels", "angels, cherubs or other winged / floating heavenly figures"),
    ("impressionist_river", "an Impressionist landscape with a river", "a river, stream or lake clearly visible"),
    ("monet_snow", "a snowy street by Claude Monet", "snow visible (white ground or roofs)"),
    ("cubist_guitar", "a Cubist still life with a guitar", "a guitar or other stringed instrument visible"),
    ("postimp_flowers", "a Post-Impressionist still life with flowers", "flowers (a bouquet or a vase of flowers) visible"),
    ("rococo_woman", "a Rococo portrait of a woman", "the portrait shows a woman"),
    ("expressionist_woman", "an Expressionist portrait of a woman", "the portrait shows a woman"),
    ("romantic_sea", "a Romantic landscape with a stormy sea", "sea, waves or ships visible"),
    ("baroque_skull", "a Baroque still life with a skull", "a skull visible"),
    ("realist_fruit", "a Realist still life with fruit", "fruit visible"),
    ("renaissance_hat", "a Northern Renaissance portrait of a man with a hat", "a man wearing a hat or cap"),
    ("matisse_woman", "a work by Henri Matisse showing a woman", "a woman visible"),
    ("renoir_portrait", "a portrait by Renoir of a woman with a hat", "a woman wearing a hat (catalogue fits nothing: expected fallback)"),
]


# --------------------------------------------------------------------------------------------- (a) the parser
def expected_fields(query):
    rel = query["relevance"]
    out = {"artist": set(), "style": set(), "genre": set()}
    if query["type"] == "genre":
        out["genre"] = {rel["genre"]}
    elif query["type"] == "style_genre":
        out["style"], out["genre"] = set(rel["styles"]), {rel["genre"]}
    else:
        out["artist"] = {rel["artist"]}
    return out


def detected_fields(parsed):
    return {"artist": set(parsed.artists), "style": set(parsed.styles), "genre": set(parsed.genres)}


def parser_report(catalogue, queries):
    """Per-field precision and recall, exact-match rate and the failures for one query set."""
    fields = ("artist", "style", "genre")
    tp, fp, fn = ({f: 0 for f in fields} for _ in range(3))
    exact, failures = [], []
    by_type = {}
    for q in queries:
        want, got = expected_fields(q), detected_fields(catalogue.parse(q["text"]))
        for f in fields:
            tp[f] += len(want[f] & got[f])
            fp[f] += len(got[f] - want[f])
            fn[f] += len(want[f] - got[f])
        ok = all(want[f] == got[f] for f in fields)
        exact.append(ok)
        by_type.setdefault(q["type"], []).append(ok)
        if not ok:
            failures.append({"id": q["id"], "text": q["text"],
                             "expected": {f: sorted(want[f]) for f in fields if want[f]},
                             "detected": {f: sorted(got[f]) for f in fields if got[f]}})

    def ratio(a, b):
        return float(a / b) if b else None

    return {
        "n_queries": len(queries),
        "exact_match_rate": float(np.mean(exact)),
        "exact_match_by_type": {t: {"n": len(v), "rate": float(np.mean(v))} for t, v in by_type.items()},
        "per_field": {f: {"precision": ratio(tp[f], tp[f] + fp[f]), "recall": ratio(tp[f], tp[f] + fn[f]),
                          "true_positive": tp[f], "false_positive": fp[f], "false_negative": fn[f]} for f in fields},
        "failures": failures,
    }


def neutral_report(catalogue):
    hits = []
    for text in NEUTRAL_QUERIES + [en for en, _ in FREE_QUERIES] + [fr for _, fr in FREE_QUERIES]:
        parsed = catalogue.parse(text)
        if parsed.has_filters:
            hits.append({"text": text, "found": parsed.describe()})
    n = len(NEUTRAL_QUERIES) + 2 * len(FREE_QUERIES)
    return {"n_questions": n, "questions_with_a_filter_found": len(hits), "examples": hits}


# --------------------------------------------------------------------------------------------- (b) retrieval
def make_embedder(model, processor, texts):
    """A function text -> vector, with every text of `texts` embedded in one batch up front."""
    unique = sorted(set(texts))
    vecs = embed_texts(model, processor, unique)
    table = dict(zip(unique, vecs, strict=True))

    def embed(text):
        if text not in table:
            table[text] = embed_texts(model, processor, [text])[0]
        return table[text]

    return embed


def rank_for(index, catalogue, embed, text, boost=None, use_catalogue=True):
    """The full ranking (best first) of one question: plain CLIP, or the catalogue filter then CLIP."""
    parsed = catalogue.parse(text)
    if not use_catalogue or not parsed.has_filters:
        return rank_all(index, embed(text))
    match = catalogue.mask(index.meta, parsed)
    if not match.any():
        return rank_all(index, embed(text))
    return rank_with_filter(index, embed(parsed.visual_text), match, boost)


def evaluate(index, catalogue, embed, queries, **kw):
    rows = []
    for q in queries:
        rel = relevance_in_rank_order(rank_for(index, catalogue, embed, q["text"], **kw), q["relevant"], len(index))
        rows.append({"id": q["id"], "type": q["type"], "text": q["text"],
                     f"precision@{K}": precision_at_k(rel, K), "average_precision": average_precision(rel), f"ndcg@{K}": ndcg_at_k(rel, K)})
    return rows


# --------------------------------------------------------------------------------------------- (c) manual check
def top5(index, catalogue, embed, text):
    plain = search(index, embed(text), 5)
    hybrid, parsed = hybrid_search(index, text, embed, catalogue, 5)
    return plain, hybrid, parsed


def save_grids(index, catalogue, embed):
    GRID_DIR.mkdir(parents=True, exist_ok=True)
    shown = {}
    for start in range(0, len(MANUAL_QUERIES), 4):
        chunk = MANUAL_QUERIES[start : start + 4]
        fig, axes = plt.subplots(2 * len(chunk), 5, figsize=(11, 2.3 * 2 * len(chunk)), squeeze=False)
        for qi, (qid, text, visual) in enumerate(chunk):
            plain, hybrid, parsed = top5(index, catalogue, embed, text)
            shown[qid] = {"plain": plain["path"].tolist(), "hybrid": hybrid["path"].tolist()}
            for row, (name, res) in enumerate((("plain", plain), ("hybrid", hybrid))):
                for col in range(5):
                    ax = axes[2 * qi + row][col]
                    ax.axis("off")
                    if col < len(res):
                        r = res.iloc[col]
                        with Image.open(DATA_DIR / r["path"]) as im:
                            ax.imshow(im.convert("RGB").resize((192, 192)))
                        ax.set_title(f"{name} #{col + 1}: {r['style']}, {r['label']}", fontsize=6)
                axes[2 * qi + row][0].text(-0.05, 0.5, f"{qid}\n{name}", transform=axes[2 * qi + row][0].transAxes, ha="right", fontsize=6)
            axes[2 * qi][2].set_title(f"{text!r}  [look for: {visual}]", fontsize=7)
        fig.tight_layout()
        fig.savefig(GRID_DIR / f"grid_{start // 4 + 1}.png", dpi=110)
        plt.close(fig)
    (GRID_DIR / "top5.json").write_text(json.dumps(shown, indent=1))
    print(f"grids saved in {GRID_DIR} (not versioned: they show paintings); judge them, then write {MANUAL_FILE}")


def manual_summary(index, catalogue, embed):
    if not MANUAL_FILE.exists():
        return None
    saved = json.loads(MANUAL_FILE.read_text())
    per_query, stale = [], []
    totals = {m: {"both": [], "catalogue": [], "visual": []} for m in ("plain", "hybrid")}
    for qid, text, _ in MANUAL_QUERIES:
        entry = saved["queries"].get(qid)
        if entry is None:
            continue
        plain, hybrid, parsed = top5(index, catalogue, embed, text)
        match = catalogue.mask(index.meta, parsed) if parsed.has_filters else np.ones(len(index), dtype=bool)
        by_path = dict(zip(index.meta["path"], match, strict=True))
        row = {"id": qid, "text": text, "catalogue_fields": parsed.describe(), "n_fit_catalogue": int(match.sum()) if parsed.has_filters else None}
        for name, res in (("plain", plain), ("hybrid", hybrid)):
            paths, visual = res["path"].tolist(), entry[name]["visual"]
            if paths != entry[name]["paths"]:
                stale.append(f"{qid}/{name}")
            cat = [int(by_path[p]) for p in paths]
            both = [c * v for c, v in zip(cat, visual, strict=False)]
            n = max(len(paths), 1)
            row[name] = {"catalogue_ok": sum(cat) / n, "visual_ok": sum(visual) / n, "both_ok": sum(both) / n, "n_shown": len(paths)}
            totals[name]["both"].append(sum(both) / n)
            totals[name]["catalogue"].append(sum(cat) / n)
            totals[name]["visual"].append(sum(visual) / n)
        per_query.append(row)
    wins = sum(r["hybrid"]["both_ok"] > r["plain"]["both_ok"] for r in per_query)
    losses = sum(r["hybrid"]["both_ok"] < r["plain"]["both_ok"] for r in per_query)
    # the visual part alone: the catalogue filter cannot improve it by construction, so this is the informative count
    visual_wins = sum(r["hybrid"]["visual_ok"] > r["plain"]["visual_ok"] for r in per_query)
    visual_losses = sum(r["hybrid"]["visual_ok"] < r["plain"]["visual_ok"] for r in per_query)
    return {
        "reviewer": saved.get("reviewer"), "protocol": saved.get("protocol"), "n_queries": len(per_query),
        "mean_top5": {m: {k: float(np.mean(v)) for k, v in t.items()} for m, t in totals.items()},
        "hybrid_better_on": wins, "hybrid_worse_on": losses,
        "hybrid_visual_part_better_on": visual_wins, "hybrid_visual_part_worse_on": visual_losses, "per_query": per_query,
        "stale_judgments": stale,
    }


# --------------------------------------------------------------------------------------------- report
def plot(retrieval, manual, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    names = ["genre", "style_genre", "artist"]
    x = np.arange(len(names))
    for j, (label, key, color) in enumerate((("plain CLIP", "plain_en", "tab:gray"), ("catalogue filter, then CLIP", "hybrid_en", "tab:blue"))):
        axes[0].bar(x + (j - 0.5) * 0.35, [retrieval["summary"][key][n][f"precision@{K}"] for n in names], 0.35, label=label, color=color,
                    hatch="//" if j else None)
    axes[0].set_xticks(x, names)
    axes[0].set_ylim(0, 1.3)
    axes[0].set_title(f"precision@{K} on metadata-defined questions\n(by construction: an upper bound, not evidence)", fontsize=9)
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(axis="y", alpha=0.3)
    if manual:
        ids = [r["id"] for r in manual["per_query"]]
        y = np.arange(len(ids))
        axes[1].barh(y - 0.2, [r["plain"]["both_ok"] for r in manual["per_query"]], 0.4, label="plain CLIP", color="tab:gray")
        axes[1].barh(y + 0.2, [r["hybrid"]["both_ok"] for r in manual["per_query"]], 0.4, label="catalogue filter, then CLIP", color="tab:blue")
        axes[1].set_yticks(y, ids, fontsize=7)
        axes[1].invert_yaxis()
        axes[1].set_xlim(0, 1.05)
        axes[1].set_title("manual check: share of the top 5 that fit BOTH parts\n(AI reviewer, small set)", fontsize=9)
        axes[1].legend(fontsize=8, loc="center right")
        visual = manual["mean_top5"]
        axes[1].text(0.99, 0.02, f"visual part alone: plain {visual['plain']['visual']:.2f}, catalogue filter {visual['hybrid']['visual']:.2f}",
                     transform=axes[1].transAxes, ha="right", va="bottom", fontsize=7, bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8})
    else:
        axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grids", action="store_true", help="only save the image grids of the manual check (data/hybrid_manual)")
    args = p.parse_args()

    df = load_metadata()
    index = load_or_build_index(df, DATA_DIR, DATA_DIR / "search_index.npz", cache_file=DATA_DIR / "clip_embeddings.npz")
    assert index.meta["id"].tolist() == df["id"].astype(str).tolist(), "index and metadata are out of sync: use --rebuild"
    catalogue = Catalogue.from_metadata(index.meta)
    queries_en = build_query_set(df)
    queries_fr = french_query_set(queries_en)
    model, processor = load_clip(index.model)

    texts = [q["text"] for q in queries_en + queries_fr] + [m[1] for m in MANUAL_QUERIES]
    texts += [catalogue.parse(t).visual_text for t in texts]
    embed = make_embedder(model, processor, texts)
    if args.grids:
        return save_grids(index, catalogue, embed)

    parser = {"english": parser_report(catalogue, queries_en), "french": parser_report(catalogue, queries_fr),
              "questions_without_catalogue_fields": neutral_report(catalogue)}
    chance = [random_baseline(len(q["relevant"]), len(index), K) for q in queries_en]
    conditions = {
        "plain_en": evaluate(index, catalogue, embed, queries_en, use_catalogue=False),
        "plain_fr": evaluate(index, catalogue, embed, queries_fr, use_catalogue=False),
        "hybrid_en": evaluate(index, catalogue, embed, queries_en),
        "hybrid_fr": evaluate(index, catalogue, embed, queries_fr),
        "boost_en": evaluate(index, catalogue, embed, queries_en, boost=BOOST),
        "boost_fr": evaluate(index, catalogue, embed, queries_fr, boost=BOOST),
    }
    retrieval = {
        "warning": "relevance is defined by the same metadata the filter reads: these numbers are an upper bound that shows the "
                   "filter works as intended, not evidence of better retrieval",
        "boost": BOOST, "summary": {name: summarise(rows, chance) for name, rows in conditions.items()},
        "questions_with_a_filter": {"english": sum(catalogue.parse(q["text"]).has_filters for q in queries_en),
                                    "french": sum(catalogue.parse(q["text"]).has_filters for q in queries_fr)},
    }
    manual = manual_summary(index, catalogue, embed)
    results = {"model": index.model, "k": K, "collection_size": len(index), "catalogue_entries": len(catalogue.lookup),
               "ambiguous_names_dropped": [" ".join(w) for w in catalogue.ambiguous],
               "parser": parser, "retrieval_by_construction": retrieval, "manual_check": manual}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "search_hybrid.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    plot(retrieval, manual, REPORTS / "figures" / "search_hybrid.png")

    for lang in ("english", "french"):
        r = parser[lang]
        print(f"\nparser, {lang}: exact match {r['exact_match_rate']:.3f}  " + "  ".join(
            f"{f}: P {v['precision'] if v['precision'] is None else round(v['precision'], 3)} R {v['recall'] if v['recall'] is None else round(v['recall'], 3)}"
            for f, v in r["per_field"].items()))
    n = parser["questions_without_catalogue_fields"]
    print(f"questions without catalogue fields: {n['questions_with_a_filter_found']}/{n['n_questions']} got a filter: {n['examples']}")
    print("\nby construction (upper bound):")
    for name in conditions:
        s = retrieval["summary"][name]
        print(f"  {name:10s} P@{K} all {s['all'][f'precision@{K}']:.3f}  AP {s['all']['average_precision']:.3f}  "
              + "  ".join(f"{t} {s[t][f'precision@{K}']:.2f}" for t in ("genre", "style_genre", "artist")))
    if manual:
        print(f"\nmanual check ({manual['n_queries']} questions): {json.dumps(manual['mean_top5'])}, hybrid better on {manual['hybrid_better_on']}, "
              f"worse on {manual['hybrid_worse_on']}; visual part alone: better on {manual['hybrid_visual_part_better_on']}, "
              f"worse on {manual['hybrid_visual_part_worse_on']}; stale: {manual['stale_judgments']}")


if __name__ == "__main__":
    main()
