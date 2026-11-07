import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from museum_guide.french import (
    ARTIST_FR,
    FREE_QUERIES,
    GENRE_FR,
    GENRE_SENTENCE_FR,
    STYLE_FR,
    de_artist,
    french_query_set,
    strip_accents,
    translate_query,
)
from museum_guide.multilingual import embed_texts_multilingual, load_multilingual
from museum_guide.search import GENRE_NOUN, GENRE_QUERIES, STYLE_ADJECTIVE, build_query_set

QUERIES_JSON = Path(__file__).resolve().parents[1] / "reports" / "search_queries.json"


@pytest.fixture
def metadata() -> pd.DataFrame:
    """440 paintings: 4 styles x 5 genres x 22 each, painters assigned in turn (20 paintings each)."""
    styles = ["Impressionism", "Cubism", "Baroque", "Romanticism"]
    genres = list(GENRE_NOUN)
    slugs = sorted(ARTIST_FR)
    rows = []
    for style in styles:
        for genre in genres:
            for _ in range(22):
                rows.append({"style": style, "label": genre, "artist": slugs[len(rows) % len(slugs)]})
    return pd.DataFrame(rows)


def test_tables_cover_every_genre_and_style_of_the_english_query_definitions():
    assert set(GENRE_SENTENCE_FR) == set(GENRE_QUERIES)
    assert set(GENRE_FR) == set(GENRE_NOUN)
    assert {g for _, g in GENRE_FR.values()} <= {"m", "f"}
    assert set(STYLE_FR) == set(STYLE_ADJECTIVE.values())


def test_every_artist_of_the_saved_query_set_has_a_french_entry():
    queries = json.loads(QUERIES_JSON.read_text())
    slugs = {q["relevance"]["artist"] for q in queries if q["type"] == "artist"}
    assert len(slugs) == 22
    assert slugs <= set(ARTIST_FR)


def test_the_saved_query_set_translates_completely():
    queries = json.loads(QUERIES_JSON.read_text())
    assert [q["type"] for q in queries].count("genre") == 5
    french = french_query_set(queries)
    assert len(french) == len(queries) == 66
    assert all(q["text"].strip() for q in french)
    assert len({q["text"] for q in french}) == len(french)  # no two questions collapse into one


def test_relevance_is_identical_between_english_and_french(metadata):
    english = build_query_set(metadata)
    french = french_query_set(english)
    assert len(english) == len(french) > 40
    for en, fr in zip(english, french):
        assert fr["id"] == en["id"]
        assert fr["type"] == en["type"]
        assert fr["relevance"] == en["relevance"]
        assert fr["relevant"] == en["relevant"]
        assert fr["text"] != en["text"]
    assert english[0]["text"] == build_query_set(metadata)[0]["text"]  # the input was not modified


def test_style_adjectives_agree_with_the_gender_of_the_genre():
    def q(adjective, genre):
        return {"id": f"style_genre:{adjective}:{genre}", "type": "style_genre", "relevance": {}}

    assert translate_query(q("Impressionist", "landscape")) == "un paysage impressionniste"
    assert translate_query(q("Cubist", "still_life")) == "une nature morte cubiste"
    assert translate_query(q("Baroque", "religious_painting")) == "une peinture religieuse baroque"
    assert translate_query(q("naive", "portrait")) == "un portrait naïf"
    assert translate_query(q("naive", "still_life")) == "une nature morte naïve"
    assert translate_query(q("Contemporary Realist", "still_life")) == "une nature morte réaliste contemporaine"
    assert translate_query(q("Northern Renaissance", "portrait")) == "un portrait de la Renaissance nordique"
    assert translate_query(q("Fauvist", "cityscape")) == "un paysage urbain fauve"


def test_every_style_and_genre_pair_translates_with_the_right_article():
    for adjective in STYLE_FR:
        for genre, (_, gender) in GENRE_FR.items():
            text = translate_query({"id": f"style_genre:{adjective}:{genre}", "type": "style_genre", "relevance": {}})
            assert text.startswith("un ") if gender == "m" else text.startswith("une ")


def test_artist_questions_elide_before_a_vowel_or_a_silent_h():
    assert de_artist("Claude Monet") == "de Claude Monet"
    assert de_artist("Alfred Sisley") == "d'Alfred Sisley"
    assert de_artist("Édouard Cortès") == "d'Édouard Cortès"
    assert de_artist("Henri Matisse") == "d'Henri Matisse"
    q = {"id": "artist:albrecht-durer", "type": "artist", "relevance": {"artist": "albrecht-durer"}}
    assert translate_query(q) == "des tableaux d'Albrecht Dürer"
    assert translate_query(q, original_names=True) == "des tableaux d'Albrecht Durer"
    q = {"id": "artist:claude-monet", "type": "artist", "relevance": {"artist": "claude-monet"}}
    assert translate_query(q) == "des tableaux de Claude Monet"


def test_an_artist_without_a_table_entry_falls_back_to_the_plain_name():
    q = {"id": "artist:pablo-picasso", "type": "artist", "relevance": {"artist": "pablo-picasso"}}
    assert translate_query(q) == "des tableaux de Pablo Picasso"


def test_unknown_query_type_is_rejected():
    with pytest.raises(ValueError):
        translate_query({"id": "x", "type": "colour", "relevance": {}})


def test_free_questions_come_in_pairs():
    assert len(FREE_QUERIES) == 5
    assert all(en and fr and en != fr for en, fr in FREE_QUERIES)


def test_multilingual_encoder_is_optional_and_explains_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes `import sentence_transformers` fail
    with pytest.raises(ImportError, match=r"\.\[multilingual\]"):
        load_multilingual()


def test_multilingual_embeddings_are_normalised():
    import numpy as np

    class FakeEncoder:
        def encode(self, texts, **kwargs):
            return np.array([[3.0, 4.0]] * len(texts))

    out = embed_texts_multilingual(FakeEncoder(), ["un paysage", "une nature morte"])
    assert out.shape == (2, 2)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_strip_accents_removes_diacritics_and_nothing_else():
    assert strip_accents("Édouard Cortès, réaliste") == "Edouard Cortes, realiste"
    assert strip_accents("une peinture naïve de Zinaïda") == "une peinture naive de Zinaida"
    assert strip_accents("Claude Monet") == "Claude Monet"
    assert strip_accents("Albrecht Dürer") == "Albrecht Durer"
