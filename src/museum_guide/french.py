"""French versions of the search questions.

The museum is French, so visitors will ask their questions in French. `translate_query` turns each question of the
metadata-labelled query set (`museum_guide.search.build_query_set`) into French WITHOUT touching its relevance rule:
the French query set has the same ids, types and relevant paintings as the English one, so scores can be compared.

The translations are written from explicit tables (genre, style, artist) with the grammatical agreement French needs
("un paysage impressionniste", "une nature morte cubiste", "des tableaux d'Alfred Sisley"). They were written by an AI
assistant and have to be read by a French speaker before anyone relies on them.
"""
from __future__ import annotations

import unicodedata

from museum_guide.search import artist_name

# The five genre sentences of `search.GENRE_QUERIES`, in French.
GENRE_SENTENCE_FR = {
    "cityscape": "Je voudrais voir des vues de ville avec des rues et des immeubles",
    "landscape": "un tableau de nature, avec des champs, des arbres et des montagnes",
    "portrait": "Je voudrais voir des portraits de personnes",
    "religious_painting": "des tableaux représentant des saints, des anges ou des scènes de la Bible",
    "still_life": "une nature morte avec des fruits, des fleurs et des objets sur une table",
}

# Genre -> (noun, gender). The noun carries the gender that the style adjective must agree with.
GENRE_FR = {
    "cityscape": ("paysage urbain", "m"),
    "landscape": ("paysage", "m"),
    "portrait": ("portrait", "m"),
    "religious_painting": ("peinture religieuse", "f"),
    "still_life": ("nature morte", "f"),
}

# English style adjective (the values of `search.STYLE_ADJECTIVE`) -> (masculine, feminine) French wording.
# Several movements are written as "de la ..." because French does not make them an adjective.
STYLE_FR = {
    "Impressionist": ("impressionniste", "impressionniste"),
    "Post-Impressionist": ("post-impressionniste", "post-impressionniste"),
    "Realist": ("réaliste", "réaliste"),
    "Expressionist": ("expressionniste", "expressionniste"),
    "Romantic": ("romantique", "romantique"),
    "Baroque": ("baroque", "baroque"),
    "Northern Renaissance": ("de la Renaissance nordique", "de la Renaissance nordique"),
    "Cubist": ("cubiste", "cubiste"),
    "Early Renaissance": ("de la première Renaissance", "de la première Renaissance"),
    "Rococo": ("rococo", "rococo"),
    "Symbolist": ("symboliste", "symboliste"),
    "High Renaissance": ("de la Haute Renaissance", "de la Haute Renaissance"),
    "Art Nouveau": ("Art nouveau", "Art nouveau"),
    "naive": ("naïf", "naïve"),
    "Mannerist": ("maniériste", "maniériste"),
    "Fauvist": ("fauve", "fauve"),
    "Pop Art": ("pop art", "pop art"),
    "Ukiyo-e": ("ukiyo-e", "ukiyo-e"),
    "Pointillist": ("pointilliste", "pointilliste"),
    "Contemporary Realist": ("réaliste contemporain", "réaliste contemporaine"),
    "New Realist": ("du nouveau réalisme", "du nouveau réalisme"),
}

# Metadata slug -> the name as a French reader writes it (accents, hyphens, usual transliteration of Russian names).
ARTIST_FR = {
    "albrecht-durer": "Albrecht Dürer",
    "alfred-sisley": "Alfred Sisley",
    "amedeo-modigliani": "Amedeo Modigliani",
    "antoine-blanchard": "Antoine Blanchard",
    "bartolome-esteban-murillo": "Bartolomé Esteban Murillo",
    "camille-corot": "Camille Corot",
    "camille-pissarro": "Camille Pissarro",
    "claude-monet": "Claude Monet",
    "edouard-cortes": "Édouard Cortès",
    "georges-braque": "Georges Braque",
    "gustave-loiseau": "Gustave Loiseau",
    "henri-fantin-latour": "Henri Fantin-Latour",
    "henri-matisse": "Henri Matisse",
    "ilya-mashkov": "Ilia Machkov",
    "isaac-levitan": "Isaac Levitan",
    "juan-gris": "Juan Gris",
    "konstantin-korovin": "Constantin Korovine",
    "mstislav-dobuzhinsky": "Mstislav Doboujinski",
    "nicholas-roerich": "Nicolas Roerich",
    "paul-gauguin": "Paul Gauguin",
    "vasily-polenov": "Vassili Polenov",
    "zinaida-serebriakova": "Zinaïda Serebriakova",
}

# The five free questions of `scripts/search_clip.py`, judged by eye (there is no metadata rule for "a stormy sea").
FREE_QUERIES = [
    ("Which painting shows a stormy sea?", "Quel tableau montre une mer agitée ?"),
    ("portraits of children", "des portraits d'enfants"),
    ("a snowy winter village", "un village enneigé en hiver"),
    ("the crucifixion of Christ", "la crucifixion du Christ"),
    ("a vase of flowers", "un vase de fleurs"),
]


def _starts_with_vowel_sound(name: str) -> bool:
    """True if 'de' elides before this name ('d'Alfred', 'd'Édouard', 'd'Henri': the h is silent)."""
    first = unicodedata.normalize("NFD", name)[0].lower()
    return first in "aeiouyh"


def de_artist(name: str) -> str:
    """'de Claude Monet' / "d'Alfred Sisley"."""
    return f"d'{name}" if _starts_with_vowel_sound(name) else f"de {name}"


def strip_accents(text: str) -> str:
    """'Édouard Cortès, réaliste' -> 'Edouard Cortes, realiste' (accents and cedillas removed, nothing else changed)."""
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if not unicodedata.combining(c)))


def artist_french(slug: str) -> str:
    """The French spelling of a painter (the plain 'Claude Monet' form if the slug has no entry)."""
    return ARTIST_FR.get(slug, artist_name(slug))


def translate_query(query: dict, original_names: bool = False) -> str:
    """The French text of one query of the English query set (`id`, `type`, `relevance`).

    With `original_names`, artists keep the spelling of the metadata ('Albrecht Durer') instead of the French one
    ('Albrecht Dürer'), to separate the effect of the language from the effect of the spelling.
    """
    kind = query["type"]
    if kind == "genre":
        return GENRE_SENTENCE_FR[query["relevance"]["genre"]]
    if kind == "style_genre":
        _, adjective, genre = query["id"].split(":", 2)
        noun, gender = GENRE_FR[genre]
        masculine, feminine = STYLE_FR[adjective]
        article = "un" if gender == "m" else "une"
        return f"{article} {noun} {masculine if gender == 'm' else feminine}"
    if kind == "artist":
        slug = query["relevance"]["artist"]
        name = artist_name(slug) if original_names else artist_french(slug)
        return f"des tableaux {de_artist(name)}"
    raise ValueError(f"unknown query type {kind!r}")


def french_query_set(queries: list[dict], original_names: bool = False) -> list[dict]:
    """Copy of the query set with French texts. Ids, types, relevance rules and relevant rows are unchanged."""
    return [{**q, "text": translate_query(q, original_names)} for q in queries]
