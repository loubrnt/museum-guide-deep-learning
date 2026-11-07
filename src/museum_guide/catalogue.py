"""Use the museum's catalogue metadata in a question: find the artist, style and genre a visitor typed.

A real museum catalogue has an artist, a style and a genre for every work, and a visitor who writes "paintings by
Claude Monet" or "un paysage impressionniste" names those fields. `Catalogue.parse` finds them in the question, in
English and in French, so that the search can filter or boost on them and rank by CLIP only what the catalogue
cannot say ("a woman with a hat").

The parser is deliberately conservative: a wrong filter hides good results, a missed filter only falls back to plain
CLIP search. So it matches whole words only, ignores accents, case, hyphens and plurals, takes the longest match,
and drops every name that two catalogue entries share. A bare surname that is also a common word ("gris", "chase")
counts only after a cue such as "by" or "de".

Names come from the metadata and from the tables already in the project: `museum_guide.french` (the French
spelling of the painters, the French style adjectives), `museum_guide.notices` (the titles of the traditions) and
`museum_guide.search` (the English style adjectives). The few synonyms and cues added here are listed below so that
they can be reviewed.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from museum_guide.french import ARTIST_FR, GENRE_FR, STYLE_FR
from museum_guide.notices import title_for
from museum_guide.search import GENRE_NOUN, STYLE_ADJECTIVE, artist_name

UNKNOWN_ARTIST = "Unknown Artist"

# Extra ways to name a genre (the nouns of `search.GENRE_NOUN` and `french.GENRE_FR` are already included).
GENRE_SYNONYMS = {
    "cityscape": ["cityscape", "urban landscape", "vue de ville"],
    "religious_painting": ["religious scene", "tableau religieux", "scène religieuse"],
}

# A surname that is also a common English or French word is accepted only after one of the CUES below
# ("paintings by Chase", "un tableau de Léger"). Surnames shorter than five letters are treated the same way.
AMBIGUOUS_SURNAMES = {"gris", "leger", "chase", "wright", "martin", "turner", "dore"}
CUES = {"by", "of", "from", "de", "d", "du", "par"}
MIN_BARE_SURNAME = 5
_AMBIGUOUS_STEMS = set()  # filled below, once `stem` exists

# Particles that belong to the surname: "van Gogh", "da Vinci", "de Toulouse-Lautrec".
_SURNAME_SUFFIX = {"the younger", "the elder"}  # 'hans-holbein-the-younger' -> 'holbein'
_ARTICLES = {"le", "la", "l", "les", "un", "une", "the", "a", "an", "de", "du", "des"}

# Words with no meaning for the visual part of a question (English and French).
STOPWORDS = {
    "a", "an", "the", "of", "with", "in", "on", "by", "and", "or", "to", "from", "for", "at", "i", "would", "like", "see",
    "show", "shows", "showing", "which", "what", "painting", "picture", "work", "artwork", "me", "us", "some", "any",
    "de", "du", "des", "la", "le", "les", "un", "une", "d", "l", "par", "avec", "et", "ou", "sur", "dans", "au", "aux",
    "tableau", "tableaux", "peinture", "oeuvre", "œuvre", "je", "voudrais", "voir", "montre", "montrant", "quel", "quelle",
    "que", "qui", "est", "ce", "cette", "ces", "en", "pour",
}


def fold(text: str) -> str:
    """Lower-case and remove accents character by character, so that character positions are unchanged."""
    out = []
    for ch in unicodedata.normalize("NFC", text):
        base = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
        out.append(base.lower() if len(base) == 1 else ch.lower())
    return "".join(out)


def stem(token: str) -> str:
    """Drop a plural 's' ('portraits', 'impressionnistes'); applied to the question and to the names alike."""
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


_AMBIGUOUS_STEMS.update(stem(w) for w in AMBIGUOUS_SURNAMES)
_TOKEN = re.compile(r"[^\W_]+")


def tokens(text: str) -> list[tuple[str, int, int]]:
    """(normalised word, start, end) for every word of `text`; hyphens and apostrophes split words."""
    return [(stem(m.group()), m.start(), m.end()) for m in _TOKEN.finditer(fold(text))]


def key(text: str) -> tuple[str, ...]:
    """The normalised word tuple that names are looked up by."""
    return tuple(t for t, _, _ in tokens(text))


def repair_slug(slug: str) -> str:
    """Repair a slug whose UTF-8 bytes were read as Latin-1 ('joaquã\xadn-sorolla' -> 'joaquín-sorolla').

    The slug was lower-cased after the damage, so the lead byte 'Ã' (C3) became 'ã' (E3): put it back first.
    """
    fixed = re.sub(
        r"[ãÃ]([\x80-\xbf])", lambda m: ("Ã" + m.group(1)).encode("latin-1").decode("utf-8", "replace"), slug
    )
    return fixed if "\ufffd" not in fixed else slug


def _strip_articles(words: tuple[str, ...]) -> tuple[str, ...]:
    while len(words) > 1 and words[0] in _ARTICLES:
        words = words[1:]
    return words


@dataclass(frozen=True)
class Detected:
    """One field found in the question."""

    field: str  # 'artist' | 'style' | 'genre'
    values: tuple[str, ...]  # artist slug / WikiArt style values / genre
    label: str  # readable name of what was found
    matched: str  # the words of the question that matched
    start: int
    end: int


@dataclass(frozen=True)
class ParsedQuery:
    """The catalogue fields found in a question, and the words left for the visual part."""

    query: str
    artists: tuple[str, ...] = ()
    styles: tuple[str, ...] = ()  # WikiArt style values (a 'Cubist' question gives three)
    genres: tuple[str, ...] = ()
    detections: tuple[Detected, ...] = ()
    remainder: str = ""  # '' when nothing but stop words is left

    @property
    def has_filters(self) -> bool:
        return bool(self.artists or self.styles or self.genres)

    @property
    def visual_text(self) -> str:
        """What to give CLIP: the remainder if it says something, else the whole question."""
        return self.remainder or self.query

    def describe(self) -> str:
        if not self.detections:
            return "no catalogue field found"
        return ", ".join(f"{d.field}: {d.label}" for d in self.detections)


@dataclass
class Catalogue:
    """Names a visitor may type for the artists, styles and genres of a collection."""

    lookup: dict[tuple[str, ...], tuple[str, tuple[str, ...], str, bool]] = field(default_factory=dict)
    ambiguous: list[tuple[str, ...]] = field(default_factory=list)
    max_len: int = 1

    # ------------------------------------------------------------------ building
    @classmethod
    def from_metadata(cls, meta: pd.DataFrame) -> Catalogue:
        """Build the vocabulary from a metadata table with the columns `artist`, `style` and `label` (the genre)."""
        entries: dict[tuple[str, ...], list[tuple[str, tuple[str, ...], str, bool]]] = {}

        def add(names: Iterable[str], field_: str, values: tuple[str, ...], label: str, needs_cue: bool = False) -> None:
            for name in names:
                words = _strip_articles(key(name))
                if words:
                    entries.setdefault(words, []).append((field_, values, label, needs_cue))

        for slug in sorted(set(meta["artist"].astype(str)) - {UNKNOWN_ARTIST}):
            clean = repair_slug(slug)
            english = artist_name(clean)
            variants = {english, ARTIST_FR.get(slug, english)}
            for name in variants:
                add([name], "artist", (slug,), english)
                words = key(name)
                for suffix in _SURNAME_SUFFIX:
                    if " ".join(words[-2:]) == suffix:
                        words = words[:-2]
                        add([" ".join(words)], "artist", (slug,), english)  # 'Hans Holbein the Younger' -> 'Hans Holbein'
                if len(words) >= 3:
                    add([f"{words[0]} {words[-1]}"], "artist", (slug,), english)  # 'Vincent Gogh' style short forms
                surname_options = {words[-1:]} | ({words[-2:]} if len(words) >= 3 else set())
                for surname in surname_options:
                    bare = " ".join(surname)
                    weak = len(surname) == 1 and (len(bare) < MIN_BARE_SURNAME or bare in _AMBIGUOUS_STEMS)
                    add([bare], "artist", (slug,), english, needs_cue=weak)

        present = sorted(set(meta["style"].astype(str)))
        for adjective in dict.fromkeys(STYLE_ADJECTIVE.values()):
            values = tuple(s for s, adj in STYLE_ADJECTIVE.items() if adj == adjective and s in present)
            if not values:
                continue
            names = [adjective, *(STYLE_FR.get(adjective, ()))]
            names = [re.sub(r"^(de la|du|de) ", "", n) for n in names]
            main = values[0]  # the main label of the movement: its noun names the whole group
            names += [title_for(main, lang) or "" for lang in ("en", "fr")]
            names.append(main.replace("_", " "))
            add(names, "style", values, adjective)
            for value in values[1:]:  # a sub-label named on its own ('Synthetic Cubism') keeps to itself
                sub = [value.replace("_", " "), *(t for t in (title_for(value, lang) for lang in ("en", "fr")) if t)]
                add(sub, "style", (value,), value.replace("_", " "))

        for genre in sorted(set(meta["label"].astype(str)) & set(GENRE_NOUN)):
            names = [GENRE_NOUN[genre], GENRE_FR[genre][0], *GENRE_SYNONYMS.get(genre, ())]
            add(names, "genre", (genre,), GENRE_NOUN[genre])

        cat = cls()
        for words, candidates in entries.items():
            distinct = {(f, v) for f, v, _, _ in candidates}
            if len(distinct) > 1:  # two entries share this name: no match is better than a wrong one
                cat.ambiguous.append(words)
                continue
            f, v, label, _ = candidates[0]
            cat.lookup[words] = (f, v, label, all(c[3] for c in candidates))
        cat.max_len = max((len(w) for w in cat.lookup), default=1)
        return cat

    # ------------------------------------------------------------------ parsing
    def parse(self, query: str) -> ParsedQuery:
        """The artist, style and genre named in `query`, and the words left over for the visual part."""
        toks = tokens(query)
        found: list[Detected] = []
        i = 0
        while i < len(toks):
            hit = None
            for n in range(min(self.max_len, len(toks) - i), 0, -1):
                words = tuple(t for t, _, _ in toks[i : i + n])
                entry = self.lookup.get(words)
                if entry is None:
                    continue
                field_, values, label, needs_cue = entry
                if needs_cue and not (i > 0 and toks[i - 1][0] in CUES):
                    continue
                hit = (n, field_, values, label)
                break
            if hit is None:
                i += 1
                continue
            n, field_, values, label = hit
            start, end = toks[i][1], toks[i + n - 1][2]
            found.append(Detected(field_, values, label, query[start:end], start, end))
            i += n

        def collect(name: str) -> tuple[str, ...]:
            return tuple(dict.fromkeys(v for d in found if d.field == name for v in d.values))

        return ParsedQuery(
            query, collect("artist"), collect("style"), collect("genre"), tuple(found),
            _remainder(query, toks, found) if found else "",  # nothing found: CLIP gets the question untouched
        )

    # ------------------------------------------------------------------ using the result
    @staticmethod
    def mask(meta: pd.DataFrame, parsed: ParsedQuery) -> np.ndarray:
        """True for the works that satisfy every field found in the question (all True if none was found)."""
        keep = np.ones(len(meta), dtype=bool)
        if parsed.artists:
            keep &= meta["artist"].isin(parsed.artists).to_numpy()
        if parsed.styles:
            keep &= meta["style"].isin(parsed.styles).to_numpy()
        if parsed.genres:
            keep &= meta["label"].isin(parsed.genres).to_numpy()
        return keep


def _remainder(query: str, toks: list[tuple[str, int, int]], found: list[Detected]) -> str:
    """The question without the fields found and without the cue words next to a painter's name."""
    drop = [False] * len(toks)
    for d in found:
        for j, (_, s, e) in enumerate(toks):
            if s >= d.start and e <= d.end:
                drop[j] = True
        if d.field == "artist":  # 'by', 'de', "d'" in front of the name mean nothing to CLIP
            first = next((j for j, (_, s, _) in enumerate(toks) if s >= d.start), None)
            if first is not None and first > 0 and toks[first - 1][0] in CUES:
                drop[first - 1] = True
    kept = [(t, s, e) for (t, s, e), gone in zip(toks, drop, strict=True) if not gone]
    while kept and kept[0][0] in STOPWORDS:
        kept.pop(0)
    while kept and kept[-1][0] in STOPWORDS:
        kept.pop()
    if not any(t not in STOPWORDS for t, _, _ in kept):
        return ""
    # keep the words of the question as written (with their own stop words), in order
    return " ".join(query[s:e] for _, s, e in kept)
