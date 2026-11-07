"""`--reject-non-paintings` in predict.py and guide.py, and the guide's rejected answer.

Nothing here loads CLIP: a fake encoder says "painting" for bright-red pictures and "wall" for the others, and the
rule's sentences are told apart by the word "painting".
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from museum_guide.clip import normalise
from museum_guide.guide import build_answer, format_text
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
    """A bright-red picture looks like a painting, any other like a wall."""

    def images(self, images):
        return np.stack([PAINTING_DIRECTION if img.convert("RGB").getpixel((0, 0))[0] >= 100 else WALL_DIRECTION for img in images])

    def texts(self, texts):
        return np.stack([PAINTING_DIRECTION if "painting" in t else WALL_DIRECTION for t in texts])


def painting_image():
    return Image.new("RGB", (8, 8), (200, 0, 0))


def wall_image():
    return Image.new("RGB", (8, 8), (10, 10, 10))


@pytest.fixture
def files(tmp_path):
    paths = {}
    for name, img in (("painting", painting_image()), ("wall", wall_image())):
        paths[name] = str(tmp_path / f"{name}.png")
        img.save(paths[name])
    probe = ClipProbe(coef=10 * np.eye(5), intercept=np.zeros(5), classes=CLASSES, C=1.0, temperature=1.0, min_confidence=0.1)
    save_probe(probe, tmp_path / "probe.npz")
    paths["probe"] = str(tmp_path / "probe.npz")
    save_rule(NonPaintingRule(("a painting",), ("a photo of a wall",), threshold=0.0), tmp_path / "rule.json")
    paths["rule"] = str(tmp_path / "rule.json")
    # a rule that also compares with the collection
    save_rule(NonPaintingRule(("a painting",), ("a photo of a wall",), threshold=0.5, w_margin=0.0, w_similarity=1.0, top_k=1), tmp_path / "rule_sim.json")
    paths["rule_sim"] = str(tmp_path / "rule_sim.json")
    rows = [{"id": f"i{j}", "path": f"portrait/i{j}.jpg", "label": "portrait", "style": "Realism", "artist": "a"} for j in range(2)]
    index = make_index(np.stack([PAINTING_DIRECTION, normalise(PAINTING_DIRECTION + 0.01)]), pd.DataFrame(rows))
    save_index(index, tmp_path / "index.npz")
    paths["index"] = str(tmp_path / "index.npz")
    return paths


# ---- predict.py ---------------------------------------------------------------------------------


def test_predict_is_unchanged_without_the_flag(files, capsys):
    predict = load_script("predict")
    predict.main([files["painting"], files["wall"], "--model", "clip", "--probe", files["probe"]], encoder=FakeEncoder())
    lines = capsys.readouterr().out.splitlines()
    assert all("does not look like a painting" not in line for line in lines)
    assert lines[0].startswith(f"{files['painting']}: ") and lines[1].startswith(f"{files['wall']}: ")


def test_predict_clip_rejects_a_wall_and_keeps_a_painting(files, capsys):
    predict = load_script("predict")
    args = ["--model", "clip", "--probe", files["probe"], "--reject-non-paintings", "--rule", files["rule"]]
    assert predict.main([files["painting"], files["wall"], *args], encoder=FakeEncoder()) == 0
    lines = capsys.readouterr().out.splitlines()
    assert "does not look like a painting" not in lines[0] and "%" in lines[0]
    assert lines[1].startswith(f"{files['wall']}: this does not look like a painting (score -1.00, threshold 0.00)")


def test_predict_json_adds_the_painting_check(files, capsys):
    predict = load_script("predict")
    args = ["--model", "clip", "--probe", files["probe"], "--reject-non-paintings", "--rule", files["rule"], "--json", "--top-k", "2"]
    predict.main([files["painting"], files["wall"], *args], encoder=FakeEncoder())
    data = json.loads(capsys.readouterr().out)
    ok, rejected = data[files["painting"]], data[files["wall"]]
    assert ok["not_a_painting"] is False and ok["painting_score"] == pytest.approx(1.0) and len(ok["predictions"]) == 2
    assert rejected["not_a_painting"] is True and rejected["painting_score"] == pytest.approx(-1.0)


def test_predict_json_with_story_and_confidence_keeps_their_fields(files, capsys):
    predict = load_script("predict")
    args = ["--model", "clip", "--probe", files["probe"], "--reject-non-paintings", "--rule", files["rule"], "--json",
            "--story", "--lang", "en", "--min-confidence", "0.1"]
    predict.main([files["painting"], files["wall"], *args], encoder=FakeEncoder())
    data = json.loads(capsys.readouterr().out)
    assert data[files["painting"]]["story"] is not None and data[files["painting"]]["unsure"] is False
    assert data[files["wall"]]["not_a_painting"] is True and data[files["wall"]]["story"] is None  # no story for a wall


def test_predict_resnet_model_can_use_the_flag(files, tmp_path, capsys):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    torch.save(model.state_dict(), tmp_path / "resnet18.pt")
    predict = load_script("predict")
    code = predict.main([files["wall"], "--weights", str(tmp_path / "resnet18.pt"), "--reject-non-paintings", "--rule", files["rule"]],
                        encoder=FakeEncoder())
    assert code == 0 and "does not look like a painting" in capsys.readouterr().out


def test_predict_rule_with_collection_needs_the_index(files, tmp_path, capsys):
    predict = load_script("predict")
    base = ["--model", "clip", "--probe", files["probe"], "--reject-non-paintings", "--rule", files["rule_sim"]]
    assert predict.main([files["painting"], *base, "--index", str(tmp_path / "missing.npz")], encoder=FakeEncoder()) == 1
    err = capsys.readouterr().err
    assert "search.py" in err and "Traceback" not in err
    assert predict.main([files["painting"], files["wall"], *base, "--index", files["index"]], encoder=FakeEncoder()) == 0
    out = capsys.readouterr().out.splitlines()
    assert "does not look like a painting" not in out[0]  # close to the collection
    assert "does not look like a painting" in out[1]  # far from it


def test_predict_missing_rule_says_what_to_run(files, tmp_path, capsys):
    predict = load_script("predict")
    code = predict.main([files["painting"], "--model", "clip", "--probe", files["probe"], "--reject-non-paintings",
                         "--rule", str(tmp_path / "none.json")], encoder=FakeEncoder())
    assert code == 1
    err = capsys.readouterr().err
    assert "non_painting.py" in err and "Traceback" not in err


# ---- the guide ----------------------------------------------------------------------------------


@pytest.fixture
def probe():
    return ClipProbe(coef=10 * np.eye(5), intercept=np.zeros(5), classes=CLASSES, C=1.0, temperature=1.0, min_confidence=0.1)


@pytest.fixture
def index(files):
    from museum_guide.search import load_index

    return load_index(files["index"])


@pytest.fixture
def rule():
    return NonPaintingRule(("a painting",), ("a photo of a wall",), threshold=0.0)


def test_build_answer_is_unchanged_without_a_rule(probe, index):
    assert set(build_answer(wall_image(), FakeEncoder(), probe, index)) == {"genre", "story", "similar"}


def test_build_answer_for_a_painting_with_a_rule_keeps_the_full_answer(probe, index, rule):
    answer = build_answer(painting_image(), FakeEncoder(), probe, index, reject_rule=rule)
    assert set(answer) == {"genre", "story", "similar"}


def test_build_answer_for_a_wall_has_no_genre(probe, index, rule):
    answer = build_answer(wall_image(), FakeEncoder(), probe, index, reject_rule=rule)
    assert answer["not_a_painting"] is True and set(answer) == {"not_a_painting", "painting_score", "threshold"}
    assert answer["painting_score"] == pytest.approx(-1.0)


def test_build_answer_for_a_wall_still_answers_a_question(probe, index, rule):
    answer = build_answer(wall_image(), FakeEncoder(), probe, index, reject_rule=rule, ask="a painting", k=2)
    assert answer["not_a_painting"] and len(answer["answers"]) == 2 and answer["question"] == "a painting"


def test_format_text_for_a_wall(probe, index, rule):
    text = format_text(build_answer(wall_image(), FakeEncoder(), probe, index, reject_rule=rule, ask="a painting", k=1), "wall.jpg")
    lines = text.splitlines()
    assert lines[0] == "wall.jpg" and lines[1].startswith("This does not look like a painting (score -1.00")
    assert 'Paintings for "a painting":' in text and "Genre" not in text


def test_guide_cli_flag(files, capsys):
    guide = load_script("guide")
    base = ["--probe", files["probe"], "--index", files["index"], "--reject-non-paintings", "--rule", files["rule"], "--json", "--k", "1"]
    assert guide.main([files["wall"], *base], encoder=FakeEncoder()) == 0
    assert json.loads(capsys.readouterr().out)[files["wall"]]["not_a_painting"] is True
    assert guide.main([files["painting"], *base], encoder=FakeEncoder()) == 0
    assert "genre" in json.loads(capsys.readouterr().out)[files["painting"]]


def test_guide_cli_missing_rule(files, tmp_path, capsys):
    guide = load_script("guide")
    code = guide.main([files["wall"], "--probe", files["probe"], "--index", files["index"], "--reject-non-paintings",
                       "--rule", str(tmp_path / "none.json")], encoder=FakeEncoder())
    assert code == 1 and "non_painting.py" in capsys.readouterr().err
