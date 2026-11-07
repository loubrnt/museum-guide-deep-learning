"""The guide in one call: a photo goes in; the genre, its story, similar paintings and answers to a question come out.

One CLIP model does all of it. The photo's vector gives the genre (a linear head), the nearest paintings of the
collection (a dot product with the vectors computed once, offline) and, for a question, the best matches (the
question's vector against the same collection vectors). Nothing here loads a model: it receives an encoder, a
head and an index, so it can be tested without downloading anything.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from museum_guide.notices import story_entry
from museum_guide.predict import ClipProbe, classify_vectors
from museum_guide.search import Index, search

SAME_PAINTING = 0.9999  # cosine similarity above which a collection painting is the photo itself


def similar_paintings(
    index: Index,
    vector,
    k: int = 5,
    photo_path: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """The `k` paintings of the collection closest to a photo's vector, never the photo itself.

    A painting is skipped when its vector is (almost) identical to the photo's, or when `photo_path` is that very
    file of the collection (`data_dir` / its path), so asking about a painting of the collection shows other paintings.
    """
    ranked = search(index, vector, len(index))
    same = ranked["score"] >= SAME_PAINTING
    if photo_path is not None and data_dir is not None:
        target = Path(photo_path).resolve()
        for i in ranked.index[:k + 5]:
            if (Path(data_dir) / ranked.at[i, "path"]).resolve() == target:
                same[i] = True
    out = ranked[~same].head(k).copy()
    out["rank"] = range(1, len(out) + 1)
    return out.reset_index(drop=True)


def _records(frame: pd.DataFrame) -> list[dict]:
    return [
        {"rank": int(r["rank"]), "score": round(float(r["score"]), 4), "path": r["path"], "label": r["label"],
         "style": r["style"], "artist": r["artist"]}
        for _, r in frame.iterrows()
    ]


def build_answer(
    photo: Image.Image,
    encoder,
    probe: ClipProbe,
    index: Index,
    *,
    lang: str = "fr",
    k: int = 5,
    ask: str | None = None,
    min_confidence: float | None = None,
    temperature: float | None = None,
    photo_path: Path | str | None = None,
    data_dir: Path | str | None = None,
) -> dict:
    """Everything the guide says about one photo, as a JSON-ready dict.

    - `genre`: the top 3 classes with probabilities (calibrated with the head's temperature unless `temperature` is
      given), and `unsure` when the best probability is below `min_confidence`;
    - `story`: the notice of the predicted genre (None when unsure: no story for a guess the guide doubts);
    - `similar`: the `k` closest paintings of the collection;
    - `answers`: with `ask`, the `k` paintings that best match the question.
    """
    vector = encoder.images([photo.convert("RGB")])
    top = classify_vectors(probe, vector, top_k=3, temperature=temperature)[0]
    unsure = min_confidence is not None and top[0][1] < min_confidence
    answer = {
        "genre": {
            "unsure": unsure,
            "min_confidence": min_confidence,
            "predictions": [{"class": c, "probability": round(p, 4)} for c, p in top],
        },
        "story": None if unsure else story_entry(top[0][0], lang),
        "similar": _records(similar_paintings(index, vector[0], k, photo_path, data_dir)),
    }
    if ask:
        answer["question"] = ask
        answer["answers"] = _records(search(index, encoder.texts([ask])[0], k))
    return answer


def format_text(answer: dict, photo_name: str = "photo") -> str:
    """A readable version of `build_answer`'s result."""
    lines = [photo_name]
    preds = answer["genre"]["predictions"]
    best = preds[0]
    if answer["genre"]["unsure"]:
        lines.append(f"Genre: unsure (best guess: {best['class']} {best['probability']:.0%})")
    else:
        rest = ", ".join(f"{p['class']} {p['probability']:.0%}" for p in preds[1:])
        lines.append(f"Genre: {best['class']} {best['probability']:.0%}" + (f" (then {rest})" if rest else ""))
    if answer["story"]:
        lines.append(f"  {answer['story']['text']}")

    def block(title, rows):
        lines.append(title)
        for r in rows:
            lines.append(f"  {r['rank']:2d}  {r['score']:.3f}  {r['path']}  [{r['label']}, {r['style']}, {r['artist']}]")

    block("Similar paintings:", answer["similar"])
    if "answers" in answer:
        block(f"Paintings for \"{answer['question']}\":", answer["answers"])
    return "\n".join(lines)
