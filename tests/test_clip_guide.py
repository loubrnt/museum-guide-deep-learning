"""The CLIP head in the real path: saved probe, classify_clip, `predict.py --model clip`, and the guide.

Nothing here loads CLIP: a fake encoder turns a picture's red channel into a class, and a tiny synthetic
index stands in for the collection.
"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from museum_guide.clip import ClipEncoder, embed_images, embed_pil_images, normalise
from museum_guide.guide import build_answer, format_text, similar_paintings
from museum_guide.predict import (
    CLASSES,
    ClipProbe,
    classify_clip,
    classify_vectors,
    load_probe,
    save_probe,
)
from museum_guide.search import make_index, save_index

ROOT = Path(__file__).resolve().parents[1]
DIM = len(CLASSES)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_vector(c):
    return normalise(np.eye(DIM)[c] + 0.2)


class FakeEncoder:
    """Image class = red channel // 50; a question is about the first genre keyword it contains."""

    KEYWORDS = {"city": 0, "landscape": 1, "portrait": 2, "saint": 3, "flower": 4}

    def images(self, images):
        return np.stack([class_vector(min(img.convert("RGB").getpixel((0, 0))[0] // 50, DIM - 1)) for img in images])

    def texts(self, texts):
        out = []
        for t in texts:
            out.append(class_vector(next((c for w, c in self.KEYWORDS.items() if w in t), 0)))
        return np.stack(out)


@pytest.fixture
def probe():
    return ClipProbe(coef=10 * np.eye(DIM), intercept=np.zeros(DIM), classes=CLASSES, C=1.0, temperature=1.0, min_confidence=0.5)


@pytest.fixture
def index():
    rows, vectors = [], []
    for c, name in enumerate(CLASSES):
        for j in range(2):
            vec = class_vector(c) if j == 0 else normalise(class_vector(c) + 0.05 * np.eye(DIM)[(c + 1) % DIM])
            vectors.append(vec)
            rows.append({"id": f"img{c}{j}", "path": f"{name}/img{c}{j}.jpg", "label": name, "style": "Realism", "artist": f"artist{c}"})
    return make_index(np.stack(vectors), pd.DataFrame(rows))


def photo_of(class_index):
    return Image.new("RGB", (8, 8), (class_index * 50 + 10, 0, 0))


# ---- the saved head ------------------------------------------------------------------------------


def test_probe_round_trip(tmp_path, probe):
    probe.temperature, probe.min_confidence, probe.split_seed = 0.44, 0.83, 3
    save_probe(probe, tmp_path / "probe.npz")
    loaded = load_probe(tmp_path / "probe.npz")
    assert loaded.classes == CLASSES
    assert np.allclose(loaded.coef, probe.coef) and np.allclose(loaded.intercept, probe.intercept)
    assert (loaded.C, loaded.model, loaded.split_seed) == (1.0, probe.model, 3)
    assert loaded.temperature == pytest.approx(0.44)
    assert loaded.min_confidence == pytest.approx(0.83)


def test_probe_without_threshold_round_trips_as_none(tmp_path, probe):
    probe.min_confidence = None
    save_probe(probe, tmp_path / "p.npz")
    assert load_probe(tmp_path / "p.npz").min_confidence is None


def test_probe_file_is_tiny(tmp_path):
    big = ClipProbe(coef=np.random.default_rng(0).normal(size=(5, 512)), intercept=np.zeros(5), classes=CLASSES, C=1.0)
    save_probe(big, tmp_path / "p.npz")
    assert (tmp_path / "p.npz").stat().st_size < 30_000


def test_missing_probe_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_clip_probe"):
        load_probe(tmp_path / "nope.npz")


# ---- classification ------------------------------------------------------------------------------


def test_classify_vectors_orders_classes_and_caps_top_k(probe):
    vectors = np.stack([class_vector(2), class_vector(4)])
    tops = classify_vectors(probe, vectors, top_k=3)
    assert [t[0][0] for t in tops] == ["portrait", "still_life"]
    for top in tops:
        names, probs = zip(*top)
        assert len(top) == 3 and set(names) <= set(CLASSES)
        assert list(probs) == sorted(probs, reverse=True)
    assert len(classify_vectors(probe, vectors, top_k=99)[0]) == DIM
    assert sum(p for _, p in classify_vectors(probe, vectors, top_k=99)[0]) == pytest.approx(1.0)


def test_temperature_changes_confidence_but_not_the_ranking(probe):
    v = class_vector(1)[None]
    sharp = classify_vectors(probe, v, top_k=5, temperature=0.5)[0]
    raw = classify_vectors(probe, v, top_k=5, temperature=1.0)[0]
    flat = classify_vectors(probe, v, top_k=5, temperature=5.0)[0]
    assert sharp[0][1] > raw[0][1] > flat[0][1]
    assert [c for c, _ in sharp] == [c for c, _ in raw] == [c for c, _ in flat]


def test_default_temperature_is_the_one_saved_with_the_head(probe):
    v = class_vector(0)[None]
    probe.temperature = 3.0
    assert classify_vectors(probe, v)[0][0][1] == pytest.approx(classify_vectors(probe, v, temperature=3.0)[0][0][1])
    assert classify_vectors(probe, v)[0][0][1] < classify_vectors(probe, v, temperature=1.0)[0][0][1]


def test_probe_probabilities_match_scikit_learn():
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(0)
    y = np.arange(150) % DIM
    X = normalise(np.eye(DIM)[y] * 2 + rng.normal(scale=0.5, size=(150, DIM)))
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(X, y)
    probe = ClipProbe(coef=clf.coef_, intercept=clf.intercept_, classes=CLASSES, C=1.0)
    tops = classify_vectors(probe, X[:20], top_k=DIM, temperature=1.0)
    for top, expected in zip(tops, clf.predict_proba(X[:20])):
        assert np.allclose([dict(top)[c] for c in CLASSES], expected, atol=1e-4)


def test_classify_clip_uses_the_encoder_on_pil_images(probe):
    tops = classify_clip(FakeEncoder(), probe, [photo_of(1), photo_of(3)], top_k=2)
    assert [t[0][0] for t in tops] == ["landscape", "religious_painting"]
    assert all(len(t) == 2 for t in tops)


# ---- sharing the embedding code ------------------------------------------------------------------


class FakeClipModel:
    """get_image_features returns the mean colour of each picture (3 numbers), like a tiny image encoder."""

    def __init__(self, wrap=False):
        self.wrap = wrap

    def get_image_features(self, pixel_values):
        out = pixel_values.mean(dim=(2, 3))
        return SimpleNamespace(pooler_output=out) if self.wrap else out


def fake_processor(images, return_tensors):
    arrays = [torch.tensor(np.asarray(img), dtype=torch.float32).permute(2, 0, 1) for img in images]
    return {"pixel_values": torch.stack(arrays)}


@pytest.mark.parametrize("wrap", [False, True])
def test_embed_pil_images_and_embed_images_agree(tmp_path, wrap):
    paths = []
    for i, colour in enumerate([(200, 10, 10), (10, 200, 10), (10, 10, 200)]):
        path = tmp_path / f"{i}.png"
        Image.new("RGB", (6, 6), colour).save(path)
        paths.append(path)
    model = FakeClipModel(wrap)
    from_paths = embed_images(model, fake_processor, paths, batch_size=2)
    from_pil = embed_pil_images(model, fake_processor, [Image.open(p) for p in paths], batch_size=1)
    assert from_paths.shape == (3, 3)
    assert np.allclose(from_paths, from_pil, atol=1e-6)
    assert np.allclose(np.linalg.norm(from_pil, axis=1), 1.0, atol=1e-5)
    assert from_pil[0].argmax() == 0 and from_pil[1].argmax() == 1 and from_pil[2].argmax() == 2


def test_clip_encoder_images_method(tmp_path):
    encoder = ClipEncoder(FakeClipModel(), fake_processor)
    vecs = encoder.images([Image.new("L", (6, 6), 120)])  # grey-scale input is converted to RGB
    assert vecs.shape == (1, 3)


# ---- predict.py --model clip ---------------------------------------------------------------------


@pytest.fixture
def probe_file(tmp_path, probe):
    save_probe(probe, tmp_path / "probe.npz")
    return tmp_path / "probe.npz"


@pytest.fixture
def photos(tmp_path):
    paths = []
    for c in (0, 2):
        path = tmp_path / f"class{c}.png"
        photo_of(c).save(path)
        paths.append(str(path))
    return paths


def test_predict_cli_clip_text_output(probe_file, photos, capsys):
    predict = load_script("predict")
    assert predict.main([photos[0], photos[1], "--model", "clip", "--probe", str(probe_file)], encoder=FakeEncoder()) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"{photos[0]}: cityscape ") and "%" in lines[0]
    assert lines[1].startswith(f"{photos[1]}: portrait ")
    assert len(lines[0].split(", ")) == 3  # top 3 by default


def test_predict_cli_clip_does_not_need_resnet_weights(probe_file, photos, tmp_path, capsys):
    predict = load_script("predict")
    code = predict.main([photos[0], "--model", "clip", "--probe", str(probe_file), "--weights", str(tmp_path / "none.pt")], encoder=FakeEncoder())
    assert code == 0


def test_predict_cli_clip_unsure_json_and_story(probe_file, photos, capsys):
    predict = load_script("predict")
    args = ["--model", "clip", "--probe", str(probe_file), photos[1]]
    predict.main([*args, "--min-confidence", "0.999", "--story", "--lang", "en"], encoder=FakeEncoder())
    assert "unsure (best guess: portrait" in capsys.readouterr().out
    predict.main([*args, "--min-confidence", "0.1", "--story", "--lang", "en", "--top-k", "2"], encoder=FakeEncoder())
    out = capsys.readouterr().out
    assert "portrait" in out and "A portrait shows" in out
    predict.main([*args, "--min-confidence", "0.1", "--json", "--top-k", "2"], encoder=FakeEncoder())
    data = json.loads(capsys.readouterr().out)[photos[1]]
    assert data["unsure"] is False and [p["class"] for p in data["predictions"]][0] == "portrait" and len(data["predictions"]) == 2


def test_predict_cli_clip_temperature_override(probe_file, photos, capsys):
    predict = load_script("predict")

    def top_probability(*extra):
        predict.main([photos[0], "--model", "clip", "--probe", str(probe_file), "--top-k", "1", "--json", *extra], encoder=FakeEncoder())
        return json.loads(capsys.readouterr().out)[photos[0]][0]["probability"]

    assert top_probability("--temperature", "5") < top_probability() < top_probability("--temperature", "0.5")


def test_predict_cli_clip_missing_probe(tmp_path, photos, capsys):
    predict = load_script("predict")
    assert predict.main([photos[0], "--model", "clip", "--probe", str(tmp_path / "no.npz")], encoder=FakeEncoder()) == 1
    err = capsys.readouterr().err
    assert "train_clip_probe" in err and "Traceback" not in err


# ---- the guide -----------------------------------------------------------------------------------


def test_similar_paintings_skips_the_photo_itself(index):
    own = index.vectors[0]  # the photo is painting 0 of the collection
    near = similar_paintings(index, own, k=3)
    assert "img00" not in set(near["id"])
    assert near["rank"].tolist() == [1, 2, 3]
    assert near["score"].is_monotonic_decreasing
    assert near.iloc[0]["id"] == "img01"  # the other painting of the same genre comes first


def test_similar_paintings_skips_the_same_file_even_with_a_different_vector(index, tmp_path):
    (tmp_path / "cityscape").mkdir()
    photo = tmp_path / "cityscape" / "img00.jpg"
    photo.write_bytes(b"x")
    vector = normalise(index.vectors[0] + 0.3 * np.eye(DIM)[3])  # different vector, same file
    near = similar_paintings(index, vector, k=len(index), photo_path=photo, data_dir=tmp_path)
    assert "img00" not in set(near["id"])
    assert len(near) == len(index) - 1


def test_build_answer_structure(probe, index):
    answer = build_answer(photo_of(2), FakeEncoder(), probe, index, lang="en", k=3, min_confidence=0.5)
    assert [p["class"] for p in answer["genre"]["predictions"]][0] == "portrait"
    assert len(answer["genre"]["predictions"]) == 3
    assert answer["genre"]["unsure"] is False
    assert answer["story"]["lang"] == "en" and answer["story"]["label"] == "portrait"
    assert len(answer["similar"]) == 3 and answer["similar"][0]["label"] == "portrait"
    assert set(answer["similar"][0]) == {"rank", "score", "path", "label", "style", "artist"}
    assert "answers" not in answer
    json.dumps(answer)  # JSON-ready


def test_build_answer_with_a_question(probe, index):
    answer = build_answer(photo_of(0), FakeEncoder(), probe, index, k=2, ask="a landscape with trees")
    assert answer["question"] == "a landscape with trees"
    assert [r["label"] for r in answer["answers"]] == ["landscape", "landscape"]
    assert [r["rank"] for r in answer["answers"]] == [1, 2]


def test_build_answer_is_unsure_below_the_threshold_and_tells_no_story(probe, index):
    answer = build_answer(photo_of(1), FakeEncoder(), probe, index, min_confidence=0.9999)
    assert answer["genre"]["unsure"] is True
    assert answer["story"] is None
    assert len(answer["similar"]) == 5  # similar paintings are still shown


def test_build_answer_temperature_override_changes_probabilities(probe, index):
    sharp = build_answer(photo_of(1), FakeEncoder(), probe, index, temperature=0.5)["genre"]["predictions"][0]["probability"]
    flat = build_answer(photo_of(1), FakeEncoder(), probe, index, temperature=5.0)["genre"]["predictions"][0]["probability"]
    assert sharp > flat


def test_format_text(probe, index):
    answer = build_answer(photo_of(2), FakeEncoder(), probe, index, lang="en", k=2, ask="a portrait", min_confidence=0.5)
    text = format_text(answer, "visit.jpg")
    lines = text.splitlines()
    assert lines[0] == "visit.jpg"
    assert lines[1].startswith("Genre: portrait ") and "(then " in lines[1]
    assert "A portrait shows" in lines[2]
    assert "Similar paintings:" in lines and 'Paintings for "a portrait":' in lines
    unsure = format_text(build_answer(photo_of(2), FakeEncoder(), probe, index, min_confidence=0.9999))
    assert "Genre: unsure (best guess: portrait " in unsure and "A portrait shows" not in unsure


@pytest.fixture
def guide_files(tmp_path, probe, index):
    save_probe(probe, tmp_path / "probe.npz")
    save_index(index, tmp_path / "index.npz")
    photo = tmp_path / "visit.png"
    photo_of(3).save(photo)
    return {"probe": str(tmp_path / "probe.npz"), "index": str(tmp_path / "index.npz"), "photo": str(photo)}


def test_guide_cli_text(guide_files, capsys):
    guide = load_script("guide")
    code = guide.main([guide_files["photo"], "--probe", guide_files["probe"], "--index", guide_files["index"],
                       "--ask", "a flower", "--k", "2", "--lang", "en"], encoder=FakeEncoder())
    out = capsys.readouterr().out
    assert code == 0
    assert out.splitlines()[0] == guide_files["photo"]
    assert "Genre: religious_painting" in out and "Similar paintings:" in out and 'Paintings for "a flower":' in out


def test_guide_cli_json(guide_files, capsys):
    guide = load_script("guide")
    guide.main([guide_files["photo"], "--probe", guide_files["probe"], "--index", guide_files["index"], "--json", "--k", "2"],
               encoder=FakeEncoder())
    data = json.loads(capsys.readouterr().out)[guide_files["photo"]]
    assert set(data) == {"genre", "story", "similar"}
    assert data["genre"]["min_confidence"] == pytest.approx(0.5)  # the threshold saved with the head
    assert len(data["similar"]) == 2


def test_guide_cli_min_confidence_override(guide_files, capsys):
    guide = load_script("guide")
    base = [guide_files["photo"], "--probe", guide_files["probe"], "--index", guide_files["index"], "--json"]
    guide.main([*base, "--min-confidence", "0.9999"], encoder=FakeEncoder())
    assert json.loads(capsys.readouterr().out)[guide_files["photo"]]["genre"]["unsure"] is True
    guide.main([*base, "--min-confidence", "0"], encoder=FakeEncoder())
    assert json.loads(capsys.readouterr().out)[guide_files["photo"]]["genre"]["unsure"] is False


def test_guide_cli_fails_cleanly(guide_files, tmp_path, capsys):
    guide = load_script("guide")
    base = [guide_files["photo"], "--probe", guide_files["probe"], "--index", guide_files["index"]]
    assert guide.main([guide_files["photo"], "--probe", str(tmp_path / "no.npz"), "--index", guide_files["index"]], encoder=FakeEncoder()) == 1
    assert "train_clip_probe" in capsys.readouterr().err
    assert guide.main([str(tmp_path / "missing.png"), *base[1:]], encoder=FakeEncoder()) == 1
    assert "missing.png" in capsys.readouterr().err
    bad = tmp_path / "bad.png"
    bad.write_text("not an image")
    assert guide.main([str(bad), *base[1:]], encoder=FakeEncoder()) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_guide_cli_refuses_an_index_built_with_another_model(guide_files, tmp_path, index, capsys):
    guide = load_script("guide")
    index.model = "some/other-clip"
    save_index(index, tmp_path / "other.npz")
    code = guide.main([guide_files["photo"], "--probe", guide_files["probe"], "--index", str(tmp_path / "other.npz")], encoder=FakeEncoder())
    assert code == 1
    assert "some/other-clip" in capsys.readouterr().err


def test_guide_cli_without_index_nor_data_says_what_to_do(guide_files, tmp_path, monkeypatch, capsys):
    guide = load_script("guide")
    monkeypatch.setattr(guide, "DATA_DIR", tmp_path / "no_data")
    code = guide.main([guide_files["photo"], "--probe", guide_files["probe"], "--index", str(tmp_path / "none.npz")], encoder=FakeEncoder())
    assert code == 1
    err = capsys.readouterr().err
    assert "search.py" in err and "Data" in err


def test_the_temperature_saved_with_the_head_is_the_default(tmp_path, probe, index, photos, capsys):
    probe.temperature = 0.5
    save_probe(probe, tmp_path / "t.npz")
    predict = load_script("predict")

    def predict_top(*extra):
        predict.main([photos[0], "--model", "clip", "--probe", str(tmp_path / "t.npz"), "--top-k", "1", "--json", *extra], encoder=FakeEncoder())
        return json.loads(capsys.readouterr().out)[photos[0]][0]["probability"]

    assert predict_top() == pytest.approx(predict_top("--temperature", "0.5"))
    assert predict_top() > predict_top("--temperature", "1")

    save_index(index, tmp_path / "i.npz")
    guide = load_script("guide")

    def guide_top(*extra):
        guide.main([photos[0], "--probe", str(tmp_path / "t.npz"), "--index", str(tmp_path / "i.npz"), "--json", *extra], encoder=FakeEncoder())
        return json.loads(capsys.readouterr().out)[photos[0]]["genre"]["predictions"][0]["probability"]

    assert guide_top() == pytest.approx(guide_top("--temperature", "0.5"))
    assert guide_top() > guide_top("--temperature", "1")
