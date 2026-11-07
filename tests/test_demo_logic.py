"""The web demo's logic: every status, both languages, captions, the loader. Nothing here needs Gradio or CLIP.

A fake encoder says "painting" for bright-red pictures and "wall" for the others, like tests/test_reject_non_paintings.py.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from museum_guide.clip import normalise
from museum_guide.demo_logic import (
    STATUSES,
    UI_TEXT,
    DemoGuide,
    DemoResult,
    artist_name,
    caption,
    display,
    from_answer,
    genre_name,
    text,
)
from museum_guide.ood import NonPaintingRule, save_rule
from museum_guide.predict import CLASSES, ClipProbe, save_probe
from museum_guide.search import make_index, save_index

ROOT = Path(__file__).resolve().parents[1]
PAINTING_DIRECTION = normalise(np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
WALL_DIRECTION = normalise(np.array([0.0, 0.0, 0.0, 1.0, 1.0]))


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEncoder:
    def images(self, images):
        return np.stack([PAINTING_DIRECTION if img.convert("RGB").getpixel((0, 0))[0] >= 100 else WALL_DIRECTION for img in images])

    def texts(self, texts):
        return np.stack([PAINTING_DIRECTION if "painting" in t else WALL_DIRECTION for t in texts])


def painting_image():
    return Image.new("RGB", (8, 8), (200, 0, 0))


def wall_image():
    return Image.new("RGB", (8, 8), (10, 10, 10))


def make_probe(min_confidence=0.1):
    return ClipProbe(coef=10 * np.eye(5), intercept=np.zeros(5), classes=CLASSES, C=1.0, temperature=1.0, min_confidence=min_confidence)


@pytest.fixture
def data_dir(tmp_path):
    root = tmp_path / "data"
    (root / "portrait").mkdir(parents=True)
    for j in range(3):
        Image.new("RGB", (20, 12), (200, 10 * j, 0)).save(root / "portrait" / f"i{j}.jpg")
    return root


@pytest.fixture
def index(data_dir):
    rows = [
        {"id": f"i{j}", "path": f"portrait/i{j}.jpg", "label": "portrait", "style": "Post_Impressionism",
         "artist": "claude-monet" if j else "Unknown Artist"}
        for j in range(3)
    ]
    vectors = np.stack([normalise(PAINTING_DIRECTION + 0.05 * j * WALL_DIRECTION) for j in range(3)])
    return make_index(vectors, pd.DataFrame(rows))


@pytest.fixture
def rule():
    return NonPaintingRule(("a painting",), ("a photo of a wall",), threshold=0.0)


@pytest.fixture
def guide(data_dir, index, rule):
    return DemoGuide(FakeEncoder(), make_probe(), index, rule, data_dir, k=2)


# ---- statuses -----------------------------------------------------------------------------------


def test_a_painting_gets_a_genre_a_story_and_similar_paintings(guide):
    result = guide.answer(painting_image())
    assert result.status == "painting"
    assert [c for c, _ in result.genre][0] == CLASSES[0]
    assert len(result.genre) == 3
    assert result.story and result.story["text"]
    assert len(result.similar) == 2
    assert result.answers == [] and result.question is None


def test_an_unsure_guess_has_no_story(data_dir, index, rule):
    guide = DemoGuide(FakeEncoder(), make_probe(min_confidence=0.99), index, rule, data_dir)
    result = guide.answer(painting_image())
    assert result.status == "unsure"
    assert result.best is not None and result.story is None
    assert display(result, data_dir).story_text == ""


def test_a_wall_is_not_a_painting(guide):
    result = guide.answer(wall_image())
    assert result.status == "not_a_painting"
    assert result.genre == [] and result.story is None and result.similar == []
    assert result.painting_score is not None


def test_without_a_rule_a_wall_still_gets_a_guess(data_dir, index):
    result = DemoGuide(FakeEncoder(), make_probe(), index, None, data_dir).answer(wall_image())
    assert result.status in ("painting", "unsure")
    assert result.genre


def test_no_image_and_no_question(guide):
    result = guide.answer(None)
    assert result.status == "no_image"
    assert result.genre == [] and result.answers == []
    assert guide.answer(None, "   ").answers == []  # a blank question is no question


def test_a_question_alone_is_answered(guide):
    result = guide.answer(None, "a painting of a portrait")
    assert result.status == "no_image"
    assert result.question == "a painting of a portrait" and len(result.answers) == 2


def test_a_question_with_a_photo(guide):
    result = guide.answer(painting_image(), "a painting")
    assert result.status == "painting" and len(result.answers) == 2


def test_a_question_is_answered_even_for_a_wall(guide):
    result = guide.answer(wall_image(), "a painting")
    assert result.status == "not_a_painting" and len(result.answers) == 2


def test_every_status_is_reachable():
    assert set(STATUSES) == {"painting", "unsure", "not_a_painting", "no_image"}


def test_an_unknown_language_falls_back_to_french(guide):
    assert guide.answer(painting_image(), lang="de").lang == "fr"


def test_from_answer_maps_the_guide_dict():
    wall = from_answer({"not_a_painting": True, "painting_score": -1.0, "threshold": 0.0})
    assert wall.status == "not_a_painting" and wall.painting_score == -1.0
    answer = {"genre": {"unsure": True, "predictions": [{"class": "portrait", "probability": 0.4}]}, "story": None, "similar": []}
    assert from_answer(answer).status == "unsure"


# ---- what the page shows ------------------------------------------------------------------------


def test_both_languages_have_the_same_labels():
    assert set(UI_TEXT["fr"]) == set(UI_TEXT["en"])
    assert all(UI_TEXT["fr"][k] and UI_TEXT["en"][k] for k in UI_TEXT["fr"])
    assert text("xx", "title") == UI_TEXT["fr"]["title"]


def test_names_and_captions_follow_the_language():
    assert genre_name("religious_painting", "fr") == "La peinture religieuse"
    assert genre_name("religious_painting", "en") == "Religious painting"
    assert genre_name("Mystery_Style") == "Mystery Style"  # no title: the label, readable
    assert artist_name("claude-monet") == "Claude Monet"
    assert artist_name("joaquã\xadn-sorolla") == "Joaquín Sorolla"  # mojibake found in the WikiArt metadata
    assert artist_name("jean-françois-millet") == "Jean François Millet"  # correct accents stay
    assert artist_name("joão-silva") == "João Silva"  # a real 'ã' that cannot be repaired stays as it is
    assert artist_name("Unknown Artist", "fr") == "Artiste inconnu"
    assert artist_name("Unknown Artist", "en") == "Unknown artist"
    record = {"label": "landscape", "style": "Impressionism", "artist": "claude-monet"}
    assert caption(record, "en").endswith("Claude Monet") and " · " in caption(record, "en")


def test_display_of_a_painting_in_each_language(guide, data_dir):
    result = guide.answer(painting_image(), lang="fr")
    fr, en = display(result, data_dir), display(result, data_dir, "en")
    assert fr.kind == "painting" and fr.verdict.startswith("C'est un tableau")
    assert en.verdict.startswith("This is a painting")
    assert fr.story_title != en.story_title  # the story follows the page's language, even after a switch
    assert fr.story_text and en.story_text and fr.story_note != en.story_note
    assert len(fr.genres) == 3 and abs(sum(fr.genres.values()) - 1) < 0.2
    assert len(fr.similar) == 2 and all(isinstance(img, Image.Image) and cap for img, cap in fr.similar)


def test_display_of_each_status(guide, data_dir):
    wall = display(guide.answer(wall_image()), data_dir, "en")
    assert wall.kind == "not_a_painting" and "does not look like a painting" in wall.verdict
    assert wall.genres == {} and wall.story_text == "" and wall.similar == []
    nothing = display(guide.answer(None), data_dir, "en")
    assert nothing.kind == "no_image" and nothing.verdict == UI_TEXT["en"]["no_image"]


def test_display_names_the_guess_when_unsure(data_dir, index, rule):
    guide = DemoGuide(FakeEncoder(), make_probe(min_confidence=0.99), index, rule, data_dir)
    shown = display(guide.answer(painting_image()), data_dir, "en")
    assert shown.kind == "unsure" and "not sure" in shown.verdict and "%" in shown.verdict


def test_a_missing_picture_does_not_break_the_page(guide, data_dir):
    result = guide.answer(painting_image())
    for path in (data_dir / "portrait").glob("*.jpg"):
        path.unlink()
    assert display(result, data_dir).similar == []


def test_demo_result_best():
    assert DemoResult("no_image").best is None
    assert DemoResult("painting", genre=[("portrait", 0.9)]).best == ("portrait", 0.9)


# ---- loading ------------------------------------------------------------------------------------


@pytest.fixture
def saved(tmp_path, data_dir, index, rule):
    save_probe(make_probe(), tmp_path / "probe.npz")
    save_index(index, tmp_path / "index.npz")
    save_rule(rule, tmp_path / "rule.json")
    return {"probe_path": tmp_path / "probe.npz", "index_path": tmp_path / "index.npz", "rule_path": tmp_path / "rule.json",
            "data_dir": data_dir}


def test_load_with_an_injected_encoder(saved):
    guide = DemoGuide.load(**saved, encoder=FakeEncoder(), k=3)
    assert guide.rejects_non_paintings and guide.k == 3
    assert guide.answer(wall_image()).status == "not_a_painting"


def test_load_without_a_rule_still_works(saved, tmp_path):
    guide = DemoGuide.load(**{**saved, "rule_path": tmp_path / "none.json"}, encoder=FakeEncoder())
    assert not guide.rejects_non_paintings


def test_load_says_what_is_missing(saved, tmp_path):
    with pytest.raises(FileNotFoundError):
        DemoGuide.load(**{**saved, "probe_path": tmp_path / "none.npz"}, encoder=FakeEncoder())
    with pytest.raises(FileNotFoundError, match="no data"):
        DemoGuide.load(**{**saved, "index_path": tmp_path / "none.npz"}, encoder=FakeEncoder())


def test_load_refuses_an_index_from_another_model(saved, index):
    other = make_index(index.vectors, index.meta, model="another/model")
    save_index(other, saved["index_path"])
    with pytest.raises(ValueError, match="rebuild"):
        DemoGuide.load(**saved, encoder=FakeEncoder())


# ---- the script, without Gradio -----------------------------------------------------------------


def test_banner_escapes_html():
    demo = load_script("demo")
    html = demo.banner("painting", "<b>x</b> & y")
    assert "&lt;b&gt;" in html and "<b>" not in html


def test_the_script_sets_telemetry_off():
    import os

    load_script("demo")
    assert os.environ["GRADIO_ANALYTICS_ENABLED"] == "False"


def test_the_script_says_how_to_install_gradio(monkeypatch):
    demo = load_script("demo")
    monkeypatch.setitem(sys.modules, "gradio", None)  # makes `import gradio` fail, whether or not it is installed
    with pytest.raises(SystemExit, match=r"\.\[demo\]"):
        demo.main([])


# ---- the Gradio page (skipped when Gradio is not installed: the CI does not need it) -------------


def test_the_page_is_built_from_a_guide(guide):
    gr = pytest.importorskip("gradio")
    demo = load_script("demo").build_demo(guide, "en")
    assert isinstance(demo, gr.Blocks)


def test_the_page_answers_through_the_gradio_client(guide, tmp_path):
    pytest.importorskip("gradio")
    gradio_client = pytest.importorskip("gradio_client")
    demo = load_script("demo").build_demo(guide, "fr")
    _, url, _ = demo.launch(server_name="127.0.0.1", prevent_thread_lock=True, quiet=True, ssr_mode=False)
    try:
        client = gradio_client.Client(url, verbose=False)
        painting, wall = tmp_path / "painting.png", tmp_path / "wall.png"
        painting_image().save(painting)
        wall_image().save(wall)
        verdict, genres, story, similar, answers = client.predict(gradio_client.handle_file(str(painting)), "", "fr", api_name="/respond")
        assert "C&#x27;est un tableau" in verdict and genres["confidences"] and story
        assert similar["visible"] and len(similar["value"]) == 2 and not answers["visible"]  # an empty gallery is hidden
        verdict, genres, story, similar, answers = client.predict(gradio_client.handle_file(str(wall)), "a painting", "en", api_name="/respond")
        assert "does not look like a painting" in verdict and story == ""
        assert not similar["visible"] and answers["visible"] and len(answers["value"]) == 2
        assert set(client.view_api(return_format="dict")["named_endpoints"]) == {"/respond", "/switch_language"}
    finally:
        demo.close()


def test_offline_flag_sets_the_hub_to_offline(monkeypatch):
    demo = load_script("demo")
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    monkeypatch.setitem(sys.modules, "gradio", None)  # stop right after the flag is read: no server, no model
    with pytest.raises(SystemExit):
        demo.main(["--offline"])
    import os

    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
