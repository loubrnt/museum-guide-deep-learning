"""What the web demo shows, without any web library: a photo and a question go in, a structured result comes out.

The demo (`scripts/demo.py`) is a thin Gradio layer over this module. Everything that decides what is said lives
here, so it is tested without Gradio and without downloading CLIP: a `DemoGuide` owns ONE CLIP encoder, the
calibrated head, the collection index and (optionally) the non-painting rule, loaded once at start-up and
shared by every request. Answers come from `museum_guide.guide.build_answer`, so the demo says exactly what
`scripts/guide.py` says.

Statuses of a result:

- `painting`: the genre, its story and similar paintings;
- `unsure`: the best guess is below the confidence saved with the head: the genre is shown as a guess, no story;
- `not_a_painting`: the photo does not look like a painting (wall, face, object): no genre, no story;
- `no_image`: no photo was given (a question alone is still answered).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from museum_guide.guide import build_answer
from museum_guide.notices import LANGS, story_entry, title_for
from museum_guide.search import Index, search

STATUSES = ("painting", "unsure", "not_a_painting", "no_image")

UI_TEXT = {
    "fr": {
        "title": "Guide du musée",
        "intro": "Photographiez un tableau : le guide en donne le genre, raconte sa tradition et montre des tableaux proches. "
                 "Vous pouvez aussi poser une question.",
        "photo": "Photo du tableau",
        "question": "Une question (facultatif)",
        "question_placeholder": "Quel tableau montre une mer agitée ?",
        "language": "Langue",
        "submit": "Interroger le guide",
        "verdict": "Verdict",
        "genres": "Genre le plus probable",
        "story": "L'histoire de cette tradition",
        "similar": "Tableaux proches",
        "answers": "Tableaux pour votre question",
        "painting": "C'est un tableau : {genre} ({p:.0%}).",
        "unsure": "Je ne suis pas sûr. Ma meilleure hypothèse : {genre} ({p:.0%}).",
        "not_a_painting": "Cela ne ressemble pas à un tableau.",
        "no_image": "Ajoutez une photo ou posez une question.",
        "unknown_artist": "Artiste inconnu",
        "draft": "Texte provisoire, rédigé par une IA : à relire avant d'être montré aux visiteurs.",
    },
    "en": {
        "title": "Museum guide",
        "intro": "Take a photo of a painting: the guide tells its genre, the story of that tradition and shows similar paintings. "
                 "You can also ask a question.",
        "photo": "Photo of the painting",
        "question": "A question (optional)",
        "question_placeholder": "Which painting shows a stormy sea?",
        "language": "Language",
        "submit": "Ask the guide",
        "verdict": "Verdict",
        "genres": "Most likely genre",
        "story": "The story of this tradition",
        "similar": "Similar paintings",
        "answers": "Paintings for your question",
        "painting": "This is a painting: {genre} ({p:.0%}).",
        "unsure": "I am not sure. My best guess: {genre} ({p:.0%}).",
        "not_a_painting": "This does not look like a painting.",
        "no_image": "Add a photo or ask a question.",
        "unknown_artist": "Unknown artist",
        "draft": "Draft text written by an AI: to be reviewed before it is shown to visitors.",
    },
}


def text(lang: str, key: str) -> str:
    """A label of the interface in `lang` (French when the language is unknown)."""
    return UI_TEXT.get(lang, UI_TEXT["fr"])[key]


def genre_name(label: str, lang: str = "fr") -> str:
    """A genre or style in words ('religious_painting' -> 'La peinture religieuse'); the raw label if there is no title."""
    return title_for(label, lang) or label.replace("_", " ")


def _repair_encoding(name: str) -> str:
    """Undo the one encoding accident of the WikiArt metadata: 'joaquã\xadn' is 'joaquín' read as latin-1 and lower-cased.

    Only applied to a name containing 'ã' and only kept when the result is valid UTF-8, so correct names are untouched.
    """
    if "ã" not in name:
        return name
    try:
        return name.replace("ã", "Ã").encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return name


def artist_name(artist: str, lang: str = "fr") -> str:
    """'claude-monet' -> 'Claude Monet'; the catch-all 'Unknown Artist' in the language of the page."""
    if artist.strip().lower() in ("unknown artist", "unknown"):
        return text(lang, "unknown_artist")
    words = _repair_encoding(artist).replace("-", " ").replace("_", " ").split()
    return " ".join(w[:1].upper() + w[1:] for w in words)


def caption(record: dict, lang: str = "fr") -> str:
    """'Le paysage · L'impressionnisme · Claude Monet' for a painting of the collection."""
    return " · ".join([genre_name(record["label"], lang), genre_name(record["style"], lang), artist_name(record["artist"], lang)])


@dataclass
class DemoResult:
    """What one request produced. Paths of paintings are relative to the data directory."""

    status: str
    lang: str = "fr"
    genre: list[tuple[str, float]] = field(default_factory=list)  # (class, probability), best first
    story: dict | None = None  # {label, lang, title, text}
    similar: list[dict] = field(default_factory=list)
    question: str | None = None
    answers: list[dict] = field(default_factory=list)
    painting_score: float | None = None

    @property
    def best(self) -> tuple[str, float] | None:
        return self.genre[0] if self.genre else None


def from_answer(answer: dict, lang: str = "fr") -> DemoResult:
    """Turn `build_answer`'s dict into a `DemoResult` with a status."""
    result = DemoResult("painting", lang, question=answer.get("question"), answers=list(answer.get("answers", [])))
    if answer.get("not_a_painting"):
        result.status, result.painting_score = "not_a_painting", answer.get("painting_score")
        return result
    result.genre = [(p["class"], p["probability"]) for p in answer["genre"]["predictions"]]
    result.story = answer["story"]
    result.similar = list(answer["similar"])
    if answer["genre"]["unsure"]:
        result.status = "unsure"
    return result


class DemoGuide:
    """The guide, loaded once: one CLIP encoder shared by every request, plus the head, the index and the rule."""

    def __init__(self, encoder, probe, index: Index, rule=None, data_dir: Path | str = "data/wikiart_genre5", k: int = 5):
        self.encoder, self.probe, self.index, self.rule = encoder, probe, index, rule
        self.data_dir, self.k = Path(data_dir), k

    @classmethod
    def load(cls, probe_path=None, index_path=None, rule_path=None, data_dir=None, encoder=None, k: int = 5) -> DemoGuide:
        """Load everything the demo needs; `encoder` replaces CLIP (tests). Errors say what to run first."""
        from museum_guide.clip import load_encoder
        from museum_guide.data import DATA_DIR
        from museum_guide.ood import DEFAULT_RULE, load_rule
        from museum_guide.predict import DEFAULT_PROBE, load_probe
        from museum_guide.search import load_or_build_index

        data_dir = Path(data_dir or DATA_DIR)
        probe = load_probe(probe_path or DEFAULT_PROBE)
        index_path = Path(index_path or data_dir / "search_index.npz")
        if not index_path.exists():
            if not (data_dir / "metadata.csv").exists():
                raise FileNotFoundError(
                    f"{index_path} not found and no data in {data_dir}: build the data first (README, section Data)."
                )
            from museum_guide.data import load_metadata

            index = load_or_build_index(load_metadata(), data_dir, index_path, cache_file=data_dir / "clip_embeddings.npz")
        else:
            index = load_or_build_index(None, data_dir, index_path)
        if index.model != probe.model:
            raise ValueError(f"the index was built with {index.model} but the head expects {probe.model}: rebuild the index.")
        rule = None
        try:
            rule = load_rule(rule_path or DEFAULT_RULE)
        except FileNotFoundError:
            pass  # the demo still works; it just cannot say "not a painting"
        return cls(encoder or load_encoder(probe.model), probe, index, rule, data_dir, k)

    @property
    def rejects_non_paintings(self) -> bool:
        return self.rule is not None

    def answer(self, photo: Image.Image | None, question: str | None = None, lang: str = "fr") -> DemoResult:
        """The result for a photo and/or a question. A blank question is no question."""
        lang = lang if lang in LANGS else "fr"
        question = (question or "").strip() or None
        if photo is None:
            result = DemoResult("no_image", lang)
            if question:
                result.question = question
                result.answers = _records(search(self.index, self.encoder.texts([question])[0], self.k))
            return result
        answer = build_answer(
            photo, self.encoder, self.probe, self.index, lang=lang, k=self.k, ask=question,
            min_confidence=self.probe.min_confidence, reject_rule=self.rule,
        )
        return from_answer(answer, lang)


def _records(frame) -> list[dict]:
    return [
        {"rank": int(r["rank"]), "score": round(float(r["score"]), 4), "path": r["path"], "label": r["label"],
         "style": r["style"], "artist": r["artist"]}
        for _, r in frame.iterrows()
    ]


@dataclass
class Display:
    """What the page shows for a result, in the page's language. `kind` picks the colour of the banner."""

    kind: str
    verdict: str
    genres: dict[str, float]
    story_title: str
    story_text: str
    story_note: str
    similar: list[tuple[Image.Image, str]]
    answers: list[tuple[Image.Image, str]]


def _gallery(records: list[dict], data_dir: Path, lang: str) -> list[tuple[Image.Image, str]]:
    """(picture, caption) pairs; pictures are read into memory, so no file path is exposed by the web layer."""
    items = []
    for r in records:
        try:
            with Image.open(data_dir / r["path"]) as img:
                items.append((img.convert("RGB"), f"{caption(r, lang)} ({r['score']:.2f})"))
        except OSError:
            continue  # a missing file must not break the page
    return items


def display(result: DemoResult, data_dir: Path | str, lang: str | None = None) -> Display:
    """The strings and pictures for a result; `lang` overrides the language the result was computed in."""
    lang = lang or result.lang
    data_dir = Path(data_dir)
    kind = result.status
    if kind == "painting" or kind == "unsure":
        name, p = result.best
        verdict = text(lang, kind).format(genre=genre_name(name, lang), p=p)
    else:
        verdict = text(lang, kind)
    genres = {genre_name(c, lang): p for c, p in result.genre}
    story_title = story_text = story_note = ""
    entry = story_entry(result.best[0], lang) if kind == "painting" else None  # in the page's language, even after a switch
    if entry:
        story_title, story_text, story_note = entry["title"], entry["text"], text(lang, "draft")
    return Display(
        kind, verdict, genres, story_title, story_text, story_note,
        _gallery(result.similar, data_dir, lang), _gallery(result.answers, data_dir, lang),
    )
