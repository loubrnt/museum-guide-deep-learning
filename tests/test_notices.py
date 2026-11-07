import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from museum_guide.notices import (
    GENRE_LABELS,
    LANGS,
    STYLE_LABELS,
    covered_labels,
    normalise_label,
    notice_for,
    story_entry,
    title_for,
)
from museum_guide.predict import CLASSES

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict.py"

# Styles that appear at least 15 times in data/wikiart_genre5/metadata.csv (hard-coded: no data is needed to test).
STYLES_IN_THE_DATA = [
    "Impressionism",
    "Post_Impressionism",
    "Realism",
    "Expressionism",
    "Romanticism",
    "Baroque",
    "Northern_Renaissance",
    "Cubism",
    "Early_Renaissance",
    "Rococo",
    "Symbolism",
    "High_Renaissance",
    "Art_Nouveau",
    "Naive_Art_Primitivism",
    "Mannerism_Late_Renaissance",
    "Synthetic_Cubism",
    "Fauvism",
    "Pop_Art",
    "Ukiyo_e",
]
FRENCH_WORDS = {"le", "la", "les", "de", "des", "du", "et", "un", "une", "en", "à", "au"}


def test_the_covered_labels_are_the_project_genres_and_styles():
    assert sorted(GENRE_LABELS) == sorted(CLASSES)
    assert sorted(STYLE_LABELS) == sorted(STYLES_IN_THE_DATA)
    assert covered_labels() == GENRE_LABELS + STYLE_LABELS
    assert len(set(covered_labels())) == len(covered_labels())  # no label is both a genre and a style


@pytest.mark.parametrize("label", covered_labels())
@pytest.mark.parametrize("lang", LANGS)
def test_every_label_has_a_titled_notice_in_both_languages(label, lang):
    text, title = notice_for(label, lang), title_for(label, lang)
    assert text and title
    assert 15 <= len(text.split()) <= 80
    assert text.endswith(".")
    assert "  " not in text
    for placeholder in ("TODO", "XXX", "TBD", "...", "lorem", "{", "}"):
        assert placeholder not in text and placeholder not in title


@pytest.mark.parametrize("label", covered_labels())
def test_french_and_english_notices_differ_and_french_looks_french(label):
    fr, en = notice_for(label, "fr"), notice_for(label, "en")
    assert fr != en
    words = set(re.findall(r"[\wàâçéèêëîïôûùüœ']+", fr.lower()))
    assert len(words & FRENCH_WORDS) >= 3, fr


def test_label_normalisation_accepts_wikiart_and_human_spellings():
    cases = {
        "Post_Impressionism": "Post_Impressionism",
        "post-impressionism": "Post_Impressionism",
        "Religious painting": "religious_painting",
        "religious_painting": "religious_painting",
        "STILL LIFE": "still_life",
        "Ukiyo-e": "Ukiyo_e",
        "Naive Art (Primitivism)": "Naive_Art_Primitivism",
        "Mannerism (Late Renaissance)": "Mannerism_Late_Renaissance",
        "  northern renaissance ": "Northern_Renaissance",
    }
    for spelling, canonical in cases.items():
        assert normalise_label(spelling) == canonical, spelling
    assert notice_for("Post-Impressionism", "en") == notice_for("Post_Impressionism", "en")


def test_unknown_labels_give_none_or_a_generic_fallback():
    assert normalise_label("Dadaism") is None
    assert notice_for("Dadaism", "fr") is None
    assert title_for("Dadaism", "en") is None
    assert story_entry("Dadaism", "fr") is None
    fallback_fr, fallback_en = notice_for("Dadaism", "fr", fallback=True), notice_for("Dadaism", "en", fallback=True)
    assert fallback_fr and fallback_en and fallback_fr != fallback_en
    assert title_for("Dadaism", "fr", fallback=True)


def test_an_unknown_language_is_an_error_not_a_silent_default():
    with pytest.raises(ValueError, match="language"):
        notice_for("portrait", "de")


def test_each_text_starts_with_the_name_of_its_tradition_and_story_entry_is_json_ready():
    # The command line prints the text alone, so it must name the tradition itself.
    assert notice_for("portrait", "fr").startswith("Le portrait")
    assert notice_for("portrait", "en").startswith("A portrait")
    entry = story_entry("Post-Impressionism", "en")
    assert entry == {
        "label": "Post_Impressionism",
        "lang": "en",
        "title": "Post-Impressionism",
        "text": notice_for("Post_Impressionism", "en"),
    }
    json.dumps(entry)  # JSON-ready


# --- the command line: random weights, as in tests/test_predict.py -----------------------------------------


@pytest.fixture
def weights(tmp_path):
    torch.manual_seed(0)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    path = tmp_path / "resnet18.pt"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 90, 3), dtype=np.uint8)).save(path)
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def _top_class(weights, photo):
    return json.loads(_run(photo, "--weights", weights, "--json").stdout)[str(photo)][0]["class"]


def test_cli_without_story_is_unchanged(weights, photo):
    plain = _run(photo, "--weights", weights)
    assert plain.returncode == 0, plain.stderr
    assert len(plain.stdout.strip().splitlines()) == 1
    shape = json.loads(_run(photo, "--weights", weights, "--json").stdout)[str(photo)]
    assert isinstance(shape, list)  # still a bare list of predictions


def test_cli_story_adds_the_french_notice_by_default(weights, photo):
    top = _top_class(weights, photo)
    out = _run(photo, "--weights", weights, "--story")
    assert out.returncode == 0, out.stderr
    first, second = out.stdout.strip().splitlines()
    assert first == _run(photo, "--weights", weights).stdout.strip()  # the usual line is untouched
    assert second == f"  {notice_for(top, 'fr')}"


def test_cli_story_in_english(weights, photo):
    top = _top_class(weights, photo)
    out = _run(photo, "--weights", weights, "--story", "--lang", "en")
    assert out.stdout.strip().splitlines()[1] == f"  {notice_for(top, 'en')}"


def test_cli_json_story(weights, photo):
    top = _top_class(weights, photo)
    data = json.loads(_run(photo, "--weights", weights, "--json", "--story", "--lang", "en").stdout)[str(photo)]
    assert set(data) == {"predictions", "story"}
    assert data["predictions"][0]["class"] == top
    assert data["story"] == story_entry(top, "en")
    sure = json.loads(_run(photo, "--weights", weights, "--json", "--story", "--min-confidence", "0").stdout)[str(photo)]
    assert set(sure) == {"unsure", "predictions", "story"}
    assert sure["unsure"] is False and sure["story"]["lang"] == "fr"


def test_cli_tells_no_story_for_a_guess_it_is_not_sure_about(weights, photo):
    out = _run(photo, "--weights", weights, "--story", "--min-confidence", "1.01")  # nothing can reach 1.01
    assert out.returncode == 0, out.stderr
    assert len(out.stdout.strip().splitlines()) == 1 and "unsure" in out.stdout
    data = json.loads(_run(photo, "--weights", weights, "--json", "--story", "--min-confidence", "1.01").stdout)[str(photo)]
    assert data["unsure"] is True and data["story"] is None


def test_cli_lang_alone_is_an_error(weights, photo):
    out = _run(photo, "--weights", weights, "--lang", "en")
    assert out.returncode == 2
    assert "--story" in out.stderr
