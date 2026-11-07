"""Is this a painting at all? Scores that separate paintings from other photos (a wall, a face, an object).

The classifiers only know five genres: shown a wall or a dog they still answer, often with confidence. These
functions give a "painting-ness" score per photo (higher = more like a painting) from quantities the guide
already computes: the classifier's probabilities, or CLIP's image vector compared with sentences and with the
collection. A photo is accepted when its score reaches a threshold chosen on validation data.

All functions are pure (numpy in, numpy out); loading CLIP and the images is done by the callers.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_RULE = Path("reports/non_painting_rule.json")
SAME_PAINTING = 0.9999  # cosine similarity above which a collection vector is the photo itself

# Sentence pairs for the zero-shot test: (what a painting is, what it is not). They were written before looking at
# any result on the non-painting photos.
PROMPT_PAIRS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "painting_vs_photograph": (("a painting",), ("a photograph",)),
    "museum_vs_everyday": (("a painting in a museum",), ("a photo of an object, a person or a wall",)),
    "oil_painting_vs_snapshot": (("an oil painting", "a painting"), ("a photograph", "a photo of an everyday scene")),
    "artwork_vs_scene": (("a work of art on canvas",), ("a photo taken with a phone camera",)),
}


# ------------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------------


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random painting scores higher than a random non-painting (ties count one half)."""
    pos, neg = np.asarray(pos, dtype=float), np.asarray(neg, dtype=float)
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    sorted_scores = scores[order]
    # Average ranks over ties.
    start = 0
    for i in range(1, len(scores) + 1):
        if i == len(scores) or sorted_scores[i] != sorted_scores[start]:
            ranks[order[start:i]] = (start + i + 1) / 2
            start = i
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def threshold_for_tpr(pos: np.ndarray, tpr: float = 0.95) -> float:
    """Highest threshold that still accepts at least `tpr` of the paintings (accept = score >= threshold)."""
    return float(np.quantile(np.asarray(pos, dtype=float), 1 - tpr, method="lower"))


def accepted_share(scores: np.ndarray, threshold: float) -> float:
    """Share of photos accepted as paintings."""
    return float((np.asarray(scores, dtype=float) >= threshold).mean())


def fpr_at_tpr(pos: np.ndarray, neg: np.ndarray, tpr: float = 0.95) -> tuple[float, float]:
    """(share of non-paintings accepted, threshold) when the threshold accepts `tpr` of the paintings."""
    t = threshold_for_tpr(pos, tpr)
    return accepted_share(neg, t), t


# ------------------------------------------------------------------------------------------------
# Scores
# ------------------------------------------------------------------------------------------------


def max_probability(probs: np.ndarray) -> np.ndarray:
    """The classifier's top probability: the usual 'how sure is it' score, used as the baseline."""
    return np.asarray(probs, dtype=float).max(axis=1)


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def side_embedding(text_vectors: np.ndarray) -> np.ndarray:
    """One unit vector for a set of sentences: the normalised mean of their (normalised) CLIP vectors."""
    return _normalise(_normalise(text_vectors).mean(axis=0))


def zero_shot_margin(image_vectors: np.ndarray, painting_text: np.ndarray, other_text: np.ndarray) -> np.ndarray:
    """cos(photo, painting sentences) - cos(photo, other sentences): positive when closer to 'a painting'."""
    v = _normalise(image_vectors)
    return v @ side_embedding(painting_text) - v @ side_embedding(other_text)


def zero_shot_probability(margin: np.ndarray, scale: float = 100.0) -> np.ndarray:
    """The margin as a probability of 'painting' (CLIP's usual logit scale of 100): for display, same ranking."""
    return 1.0 / (1.0 + np.exp(-scale * np.asarray(margin, dtype=float)))


def collection_similarity(
    vectors: np.ndarray,
    reference: np.ndarray,
    top_k: int = 1,
    ignore_above: float | None = None,
    exclude_index: np.ndarray | None = None,
) -> np.ndarray:
    """Mean cosine similarity between each photo and its `top_k` closest paintings of the collection.

    `ignore_above` drops reference vectors that are (almost) the photo itself, so a painting of the collection
    is judged like a new one (leave-one-out); the guide uses `SAME_PAINTING`. `exclude_index` (one reference row
    per photo, -1 for none) drops a known original, for a degraded copy of a collection painting.
    """
    sims = _normalise(vectors) @ _normalise(reference).T
    if ignore_above is not None:
        sims = np.where(sims >= ignore_above, -np.inf, sims)
    if exclude_index is not None:
        rows = np.where(np.asarray(exclude_index) >= 0)[0]
        sims[rows, np.asarray(exclude_index)[rows]] = -np.inf
    k = min(top_k, sims.shape[1])
    top = -np.sort(-sims, axis=1)[:, :k]
    return top.mean(axis=1)


# ------------------------------------------------------------------------------------------------
# A rule: how to score a photo and where to cut
# ------------------------------------------------------------------------------------------------


@dataclass
class NonPaintingRule:
    """How the guide decides 'this does not look like a painting', saved as a small JSON file.

    score = w_margin * zero-shot margin (painting sentences against other sentences)
          + w_similarity * mean similarity to the `top_k` closest collection paintings + bias;
    a photo is accepted as a painting when score >= threshold. `w_similarity = 0` needs no collection at all.
    """

    painting_prompts: tuple[str, ...]
    other_prompts: tuple[str, ...]
    threshold: float
    w_margin: float = 1.0
    w_similarity: float = 0.0
    bias: float = 0.0
    top_k: int = 5
    target_tpr: float = 0.95  # share of validation paintings accepted when the threshold was chosen
    info: dict = field(default_factory=dict)

    @property
    def needs_collection(self) -> bool:
        return self.w_similarity != 0.0

    def score(
        self, image_vectors: np.ndarray, painting_text: np.ndarray, other_text: np.ndarray, reference: np.ndarray | None = None
    ) -> np.ndarray:
        s = self.w_margin * zero_shot_margin(image_vectors, painting_text, other_text) + self.bias
        if self.needs_collection:
            if reference is None:
                raise ValueError("this rule compares the photo with the collection: pass the collection vectors")
            s = s + self.w_similarity * collection_similarity(image_vectors, reference, self.top_k, SAME_PAINTING)
        return s

    def accepts(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores, dtype=float) >= self.threshold


def painting_scores(rule: NonPaintingRule, encoder, image_vectors: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Score photos with a rule: `encoder.texts` turns the rule's sentences into vectors (a `museum_guide.clip.ClipEncoder`).

    `reference` are the collection's vectors (the search index), needed only when the rule compares with the collection.
    """
    return rule.score(image_vectors, encoder.texts(list(rule.painting_prompts)), encoder.texts(list(rule.other_prompts)), reference)


def save_rule(rule: NonPaintingRule, path: Path | str = DEFAULT_RULE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(rule)
    data["painting_prompts"], data["other_prompts"] = list(rule.painting_prompts), list(rule.other_prompts)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_rule(path: Path | str = DEFAULT_RULE) -> NonPaintingRule:
    """Read a rule saved by `save_rule`; says what to run if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: fit it first with `python scripts/non_painting.py`.")
    d = json.loads(path.read_text())
    d["painting_prompts"], d["other_prompts"] = tuple(d["painting_prompts"]), tuple(d["other_prompts"])
    return NonPaintingRule(**d)


def fit_linear_score(features_pos: np.ndarray, features_neg: np.ndarray) -> tuple[np.ndarray, float]:
    """Weights and bias of a logistic regression 'painting vs not' on a few raw features (no standardisation needed here).

    Used to combine, for example, the zero-shot margin and the collection similarity. Returns (weights, bias) so that
    `features @ weights + bias` is the score (a logit); fit it on a validation draw, never on the test photos.
    """
    from sklearn.linear_model import LogisticRegression

    X = np.vstack([features_pos, features_neg])
    y = np.concatenate([np.ones(len(features_pos)), np.zeros(len(features_neg))])
    scale = X.std(axis=0) + 1e-12
    clf = LogisticRegression(C=10.0, max_iter=5000, class_weight="balanced").fit(X / scale, y)
    w = clf.coef_[0] / scale
    return w, float(clf.intercept_[0])
