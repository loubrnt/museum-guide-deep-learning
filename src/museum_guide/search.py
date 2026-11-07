"""Natural-language search over the collection: an index of CLIP image vectors, ranking, and metrics.

The index is the 1,500 normalised image vectors plus the metadata needed to show a result (path, genre,
style, artist). It is built once (`build_index`, about 30 s on CPU) and saved, so a question costs one text
encoding and a dot product. The metric helpers score a ranking against a binary relevance vector.

The query set (`build_query_set`) is generated from the WikiArt metadata, so "relevant" is objective and
reproducible. It measures what the metadata knows (genre, style, artist), not fine visual content.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from museum_guide.clip import MODEL, cached_embeddings, embed_images, load_clip, normalise

if TYPE_CHECKING:  # the catalogue module imports this one: only needed for the type hints
    from museum_guide.catalogue import Catalogue, ParsedQuery

META_COLUMNS = ("id", "path", "label", "style", "artist")


@dataclass
class Index:
    """Normalised image vectors (one row per painting) and their metadata, in the same order."""

    vectors: np.ndarray
    meta: pd.DataFrame
    model: str = MODEL

    def __len__(self) -> int:
        return len(self.vectors)


def make_index(vectors: np.ndarray, df: pd.DataFrame, model: str = MODEL) -> Index:
    """An index from vectors and a metadata table with the `META_COLUMNS` (extra columns are dropped)."""
    if len(vectors) != len(df):
        raise ValueError(f"{len(vectors)} vectors for {len(df)} metadata rows")
    meta = df[list(META_COLUMNS)].reset_index(drop=True).astype(str)
    return Index(normalise(vectors), meta, model)


def build_index(df: pd.DataFrame, data_dir: Path, cache_file: Path | None = None, model_name: str = MODEL) -> Index:
    """Embed every image of `df` with CLIP (or reuse the cached vectors) and build the index."""
    paths = [Path(data_dir) / p for p in df["path"]]

    def compute() -> np.ndarray:
        model, processor = load_clip(model_name)
        return embed_images(model, processor, paths)

    vectors = cached_embeddings(cache_file, model_name, paths, compute) if cache_file else compute()
    return make_index(vectors, df, model_name)


def load_or_build_index(
    df: pd.DataFrame, data_dir: Path, index_file: Path, rebuild: bool = False, cache_file: Path | None = None
) -> Index:
    """The saved index if there is one (and `rebuild` is off), otherwise build it and save it."""
    index_file = Path(index_file)
    if index_file.exists() and not rebuild:
        return load_index(index_file)
    index = build_index(df, data_dir, cache_file)
    save_index(index, index_file)
    return index


def save_index(index: Index, file: Path | str) -> None:
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(file, vectors=index.vectors, model=np.array(index.model), **{c: index.meta[c].to_numpy(dtype=str) for c in META_COLUMNS})


def load_index(file: Path | str) -> Index:
    with np.load(file) as f:
        meta = pd.DataFrame({c: f[c].astype(str) for c in META_COLUMNS})
        return Index(f["vectors"], meta, str(f["model"]))


def scores(index: Index, text_embedding: np.ndarray) -> np.ndarray:
    """Cosine similarity between one question vector (or the first row of a matrix) and every painting."""
    return index.vectors @ np.asarray(text_embedding, dtype=np.float32).reshape(-1)[: index.vectors.shape[1]]


def rank_all(index: Index, text_embedding: np.ndarray) -> np.ndarray:
    """Indices of all paintings, best match first (ties keep the collection order)."""
    return np.argsort(-scores(index, text_embedding), kind="stable")


def search(index: Index, text_embedding: np.ndarray, k: int = 5) -> pd.DataFrame:
    """The `k` best paintings for a question: rank, score and metadata, best first."""
    s = scores(index, text_embedding)
    top = np.argsort(-s, kind="stable")[: min(k, len(s))]
    out = index.meta.iloc[top].copy()
    out.insert(0, "score", s[top])
    out.insert(0, "rank", np.arange(1, len(top) + 1))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# Hybrid search: the catalogue fields named in the question, then CLIP for the rest.
# ---------------------------------------------------------------------------------------------


def rank_with_filter(index: Index, text_embedding: np.ndarray, match: np.ndarray, boost: float | None = None) -> np.ndarray:
    """Indices of all paintings, best first, when the question names catalogue fields (`match` is True where they fit).

    Hard filter (`boost=None`): the paintings that fit come first, ranked by CLIP, then the others, ranked by CLIP.
    Soft boost: CLIP's cosine plus `boost` for each painting that fits (a cosine gap between good and bad matches
    is a few hundredths, so 0.1 or more behaves almost like a filter, 0.02 like a nudge).
    """
    s = scores(index, text_embedding)
    match = np.asarray(match, dtype=bool)
    if boost is None:
        return np.lexsort((-s, ~match))  # primary key: fits first; secondary: CLIP score
    return np.argsort(-(s + boost * match), kind="stable")


def hybrid_search(
    index: Index,
    query: str,
    embed: Callable[[str], np.ndarray],
    catalogue: Catalogue,
    k: int = 5,
    boost: float | None = None,
) -> tuple[pd.DataFrame, ParsedQuery]:
    """Search with the catalogue fields found in `query`, and CLIP for what the catalogue cannot say.

    `embed` turns a sentence into a normalised text vector. Returns the results and the parsed question.
    Nothing found, or nothing in the collection fits all the fields: plain CLIP search on the whole question
    (`catalogue_match` is then False for every row). With a hard filter only the paintings that fit are returned
    (fewer than `k` if the catalogue has fewer); with `boost` the best `k` overall are returned.
    The text given to CLIP is what remains of the question once the fields are removed ("a woman with a hat"),
    or the whole question when nothing is left.
    """
    parsed = catalogue.parse(query)
    if not parsed.has_filters:
        results = search(index, embed(query), k)
        results["catalogue_match"] = False
        return results, parsed
    match = catalogue.mask(index.meta, parsed)
    if not match.any():  # e.g. 'a Baroque landscape' in a collection that has none: do not return an empty page
        results = search(index, embed(query), k)
        results["catalogue_match"] = False
        return results, parsed
    emb = embed(parsed.visual_text)
    order = rank_with_filter(index, emb, match, boost)
    if boost is None:
        order = order[match[order]]
    top = order[: min(k, len(order))]
    s = scores(index, emb)
    out = index.meta.iloc[top].copy()
    out.insert(0, "score", s[top])
    out.insert(0, "rank", np.arange(1, len(top) + 1))
    out["catalogue_match"] = match[top]
    return out.reset_index(drop=True), parsed


# ---------------------------------------------------------------------------------------------
# Metrics. `relevant` is a binary vector listed in ranking order (best match first).
# ---------------------------------------------------------------------------------------------


def precision_at_k(relevant: Sequence[int] | np.ndarray, k: int = 10) -> float:
    r = np.asarray(relevant)[:k]
    return float(r.sum() / k)


def average_precision(relevant: Sequence[int] | np.ndarray) -> float:
    """Mean of the precision at each relevant item, over the whole ranking (0 if nothing is relevant)."""
    r = np.asarray(relevant, dtype=float)
    if r.sum() == 0:
        return 0.0
    precision = np.cumsum(r) / np.arange(1, len(r) + 1)
    return float((precision * r).sum() / r.sum())


def ndcg_at_k(relevant: Sequence[int] | np.ndarray, k: int = 10) -> float:
    """Binary-gain nDCG: 1.0 when all the best possible items are at the top."""
    r = np.asarray(relevant, dtype=float)
    discount = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((r[:k] * discount[: len(r[:k])]).sum())
    ideal = float(discount[: int(min(k, float(r.sum())))].sum())
    return dcg / ideal if ideal > 0 else 0.0


def random_baseline(n_relevant: int, n_items: int, k: int = 10, n_draws: int = 300, seed: int = 0) -> dict[str, float]:
    """Expected metrics of a random ranking, by simulation: what 'no knowledge' scores on this query."""
    rng = np.random.default_rng(seed)
    base = np.zeros(n_items, dtype=int)
    base[:n_relevant] = 1
    draws = [rng.permutation(base) for _ in range(n_draws)]
    return {
        f"precision@{k}": float(np.mean([precision_at_k(d, k) for d in draws])),
        "average_precision": float(np.mean([average_precision(d) for d in draws])),
        f"ndcg@{k}": float(np.mean([ndcg_at_k(d, k) for d in draws])),
    }


# ---------------------------------------------------------------------------------------------
# Query set generated from the metadata.
# ---------------------------------------------------------------------------------------------

# Same five sentences as GENRE_QUERIES in scripts/search_clip.py, kept for continuity.
GENRE_QUERIES = {
    "cityscape": "I would like to see views of a city with streets and buildings",
    "landscape": "a painting of nature, fields, trees and mountains",
    "portrait": "I would like to see portraits of people",
    "religious_painting": "paintings showing saints, angels or scenes from the Bible",
    "still_life": "a still life with fruit, flowers and objects on a table",
}
GENRE_NOUN = {
    "cityscape": "city view",
    "landscape": "landscape",
    "portrait": "portrait",
    "religious_painting": "religious painting",
    "still_life": "still life",
}
# WikiArt `style` value -> adjective used in the question. The three Cubism labels are one movement for a visitor.
STYLE_ADJECTIVE = {
    "Impressionism": "Impressionist",
    "Post_Impressionism": "Post-Impressionist",
    "Realism": "Realist",
    "Expressionism": "Expressionist",
    "Romanticism": "Romantic",
    "Baroque": "Baroque",
    "Northern_Renaissance": "Northern Renaissance",
    "Cubism": "Cubist",
    "Synthetic_Cubism": "Cubist",
    "Analytical_Cubism": "Cubist",
    "Early_Renaissance": "Early Renaissance",
    "Rococo": "Rococo",
    "Symbolism": "Symbolist",
    "High_Renaissance": "High Renaissance",
    "Art_Nouveau": "Art Nouveau",
    "Naive_Art_Primitivism": "naive",
    "Mannerism_Late_Renaissance": "Mannerist",
    "Fauvism": "Fauvist",
    "Pop_Art": "Pop Art",
    "Ukiyo_e": "Ukiyo-e",
    "Pointillism": "Pointillist",
    "Contemporary_Realism": "Contemporary Realist",
    "New_Realism": "New Realist",
}


def _article(word: str) -> str:
    return "an" if word[0].lower() in "aeio" else "a"  # 'a Ukiyo-e', 'an Impressionist'


def artist_name(slug: str) -> str:
    """'claude-monet' -> 'Claude Monet'."""
    return " ".join(part.capitalize() for part in slug.split("-"))


def build_query_set(df: pd.DataFrame, min_relevant: int = 10) -> list[dict]:
    """Questions with objective relevance, from the metadata. Row order of `df` must match the index.

    Three types, each with at least `min_relevant` relevant paintings so that precision@10 can reach 1.0:
    `genre` (the five genre sentences), `style_genre` ("a Cubist still life": the style group and the genre
    must both match) and `artist` ("paintings by Claude Monet": the artist field matches).
    Each query is a dict with `id`, `type`, `text`, `relevance` (the metadata rule) and `relevant` (row indices).
    """
    queries = []
    for genre, text in GENRE_QUERIES.items():
        rel = np.flatnonzero(df["label"] == genre)
        queries.append({"id": f"genre:{genre}", "type": "genre", "text": text, "relevance": {"genre": genre}, "relevant": rel.tolist()})

    groups: dict[str, list[str]] = {}
    for style, adjective in STYLE_ADJECTIVE.items():
        groups.setdefault(adjective, []).append(style)
    for adjective, styles in groups.items():
        for genre, noun in GENRE_NOUN.items():
            rel = np.flatnonzero(df["style"].isin(styles) & (df["label"] == genre))
            if len(rel) >= min_relevant:
                text = f"{_article(adjective)} {adjective} {noun}"
                queries.append(
                    {"id": f"style_genre:{adjective}:{genre}", "type": "style_genre", "text": text,
                     "relevance": {"styles": styles, "genre": genre}, "relevant": rel.tolist()}
                )

    counts = df.loc[df["artist"] != "Unknown Artist", "artist"].value_counts()
    for artist in sorted(counts[counts >= min_relevant].index):
        rel = np.flatnonzero(df["artist"] == artist)
        queries.append(
            {"id": f"artist:{artist}", "type": "artist", "text": f"paintings by {artist_name(artist)}",
             "relevance": {"artist": artist}, "relevant": rel.tolist()}
        )
    return queries


def relevance_in_rank_order(order: np.ndarray, relevant: Sequence[int], n_items: int) -> np.ndarray:
    """Binary vector, in ranking order, marking which ranked items are relevant."""
    mask = np.zeros(n_items, dtype=int)
    mask[list(relevant)] = 1
    return mask[order]
