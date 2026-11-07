"""The real-photo evaluation harness: folder scanning, the guide's decisions, the summary, the script.

No real photo and no CLIP here. A fake encoder reads the colour of the picture: red looks like a portrait, green like a
still life, blue like a landscape, dark like a wall.
"""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from museum_guide.ood import NonPaintingRule, save_rule
from museum_guide.photo_eval import (
    ANSWERED,
    NOT_A_PAINTING,
    REJECTED,
    UNSURE,
    Photo,
    build_demo_set,
    decide,
    evaluate,
    format_summary,
    normalise_label,
    open_photo,
    painting_key,
    plot_confusion,
    run_photos,
    scan_photo_dir,
    write_records,
)
from museum_guide.predict import CLASSES, ClipProbe, save_probe

ROOT = Path(__file__).resolve().parents[1]
RED, GREEN, BLUE, DARK = (200, 0, 0), (0, 200, 0), (0, 0, 200), (10, 10, 10)
WALL_AXIS = 5  # the sixth dimension of the fake vectors


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"script_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def vector_of(img):
    """A fake CLIP vector: one-hot genre for a bright picture (red portrait, green still life, blue landscape), wall axis for a dark one."""
    r, g, b = img.convert("RGB").getpixel((img.size[0] // 2, img.size[1] // 2))
    v = np.zeros(6)
    if max(r, g, b) < 50:
        v[WALL_AXIS] = 1.0
    else:
        v[[2, 4, 1][int(np.argmax([r, g, b]))]] = 1.0
    return v


class FakeEncoder:
    def images(self, images):
        return np.stack([vector_of(i) for i in images])

    def texts(self, texts):
        out = []
        for t in texts:
            v = np.zeros(6)
            if "wall" in t:
                v[WALL_AXIS] = 1.0
            else:
                v[:5] = 1.0
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def save(path, colour=RED, size=(32, 24), **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, **kwargs)
    return path


@pytest.fixture
def photo_dir(tmp_path):
    root = tmp_path / "photos"
    save(root / "portrait" / "red__a.png", RED)
    save(root / "portrait" / "red__b.png", RED)  # same painting as red__a
    save(root / "still_life" / "green__a.png", GREEN)
    save(root / "cityscape" / "redcity__a.png", RED)  # looks like a portrait: a wrong answer
    save(root / "not_a_painting" / "wall.png", DARK)
    return root


@pytest.fixture
def probe_file(tmp_path):
    coef = np.hstack([10 * np.eye(5), np.zeros((5, 1))])  # logit 10 for the genre of a one-hot vector, all zero for the wall axis
    save_probe(ClipProbe(coef=coef, intercept=np.zeros(5), classes=CLASSES, C=1.0, temperature=1.0, min_confidence=None), tmp_path / "probe.npz")
    return str(tmp_path / "probe.npz")


@pytest.fixture
def rule_file(tmp_path):
    save_rule(NonPaintingRule(("a painting",), ("a photo of a wall",), threshold=0.0), tmp_path / "rule.json")
    return str(tmp_path / "rule.json")


# ---- finding photos -----------------------------------------------------------------------------


def test_normalise_label():
    assert normalise_label(" Religious painting ") == "religious_painting"
    assert normalise_label("still-life") == "still_life"


def test_painting_key_follows_the_naming_convention(tmp_path):
    assert painting_key(tmp_path / "mona__close_up.jpg", "portrait") == "portrait/mona"
    assert painting_key(tmp_path / "mona__angle.jpg", "portrait") == "portrait/mona"
    assert painting_key(tmp_path / "mona__angle.jpg", "landscape") == "landscape/mona"  # the same id in another genre is another painting
    assert painting_key(tmp_path / "IMG_0042.jpg", "portrait", tmp_path) == "IMG_0042.jpg"  # no convention: its own cluster, relative path


def test_scan_finds_labelled_photos_and_notes_what_it_ignores(tmp_path):
    root = tmp_path / "p"
    save(root / "Religious painting" / "a__1.jpg", RED)  # folder names are normalised
    save(root / "portrait" / "sub" / "b__1.png", RED)  # nested folders are read
    save(root / "portrait" / ".hidden.jpg", RED)
    (root / "portrait" / "notes.txt").write_text("not a photo")
    save(root / "holidays" / "c.jpg", RED)  # not a label
    (root / "landscape").mkdir()  # empty
    photos, notes = scan_photo_dir(root)
    assert sorted((p.label, p.path.name) for p in photos) == [("portrait", "b__1.png"), ("religious_painting", "a__1.jpg")]
    assert any("holidays" in n and "ignored" in n for n in notes)
    assert any("landscape" in n and "no photo" in n for n in notes)


def test_scan_errors_are_explicit(tmp_path):
    with pytest.raises(FileNotFoundError, match="one sub-folder per label"):
        scan_photo_dir(tmp_path / "missing")
    (tmp_path / "empty" / "portrait").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="no photo found"):
        scan_photo_dir(tmp_path / "empty")


# ---- opening photos -----------------------------------------------------------------------------


def test_open_photo_applies_the_exif_rotation(tmp_path):
    img = Image.new("RGB", (40, 20), RED)
    exif = img.getexif()
    exif[0x0112] = 6  # stored sideways: to be shown rotated by 90 degrees
    img.save(tmp_path / "side.jpg", exif=exif)
    assert open_photo(tmp_path / "side.jpg", max_side=0).size == (20, 40)


def test_open_photo_brings_big_photos_down_and_keeps_small_ones(tmp_path):
    save(tmp_path / "big.jpg", RED, size=(3000, 2000), quality=80)
    save(tmp_path / "small.png", RED, size=(64, 48))
    big = open_photo(tmp_path / "big.jpg", max_side=1024)
    assert max(big.size) <= 1024 and big.size[0] > big.size[1]
    assert open_photo(tmp_path / "big.jpg", max_side=0).size == (3000, 2000)
    assert open_photo(tmp_path / "small.png", max_side=1024).size == (64, 48)  # never enlarged, PNG has no draft mode


def test_open_photo_raises_oserror_on_a_file_that_is_not_an_image(tmp_path):
    (tmp_path / "broken.jpg").write_text("not an image")
    with pytest.raises(OSError):
        open_photo(tmp_path / "broken.jpg")


# ---- decisions ----------------------------------------------------------------------------------


def test_decide_precedence():
    assert decide(0.99, 0.5, rejected=True) == REJECTED  # a refused photo is refused whatever its probability
    assert decide(0.1, 0.5, rejected=True) == REJECTED  # ... including one the guide would otherwise call 'unsure'
    assert decide(0.4, 0.5) == UNSURE
    assert decide(0.5, 0.5) == ANSWERED  # the threshold itself is answered, as in the guide
    assert decide(0.1, None) == ANSWERED  # no threshold: always answer


class FakeRule:
    def accepts(self, score):
        return np.asarray(score) >= 0.0


def test_run_photos_records_every_photo_and_skips_unreadable_ones(photo_dir):
    (photo_dir / "portrait" / "broken__x.jpg").write_text("not an image")
    photos, _ = scan_photo_dir(photo_dir)
    seen_batches = []

    def predict_batch(images):
        seen_batches.append(len(images))
        return [[("portrait", 0.9)] for _ in images], [1.0] * len(images)

    records, skipped = run_photos(photos, predict_batch, rule=FakeRule(), batch_size=2, root=photo_dir)
    assert len(records) == 5 and len(skipped) == 1
    assert skipped[0]["path"] == "portrait/broken__x.jpg" and "reason" in skipped[0]
    assert max(seen_batches) <= 2
    paths = {r["path"] for r in records}
    assert "portrait/red__a.png" in paths and not any(str(photo_dir) in p for p in paths)  # relative paths only
    assert {r["painting_id"] for r in records if r["true_label"] == "portrait"} == {"portrait/red"}


def test_run_photos_applies_the_rules(photo_dir):
    photos, _ = scan_photo_dir(photo_dir)
    scores = {"wall.png": -1.0}

    def predict_batch(images):
        tops = [[("portrait", 0.4 if i.size == (32, 24) and i.getpixel((0, 0)) == GREEN else 0.9)] for i in images]
        return tops, [scores.get("wall.png", 1.0) if i.getpixel((0, 0)) == DARK else 1.0 for i in images]

    records, _ = run_photos(photos, predict_batch, min_confidence=0.5, rule=FakeRule())
    by_name = {Path(r["path"]).name: r["status"] for r in records}
    assert by_name["wall.png"] == REJECTED
    assert by_name["green__a.png"] == UNSURE
    assert by_name["red__a.png"] == ANSWERED
    # without a rule the scores are not asked for and nothing is refused
    records, _ = run_photos(photos, lambda images: ([[("portrait", 0.9)] for _ in images], None))
    assert {r["status"] for r in records} == {ANSWERED}


# ---- summary ------------------------------------------------------------------------------------


def record(true, predicted, probability=0.9, status=ANSWERED, painting_id=None, path="p.jpg"):
    return {"path": path, "true_label": true, "painting_id": painting_id or f"{true}/{path}", "predicted": predicted,
            "probability": probability, "status": status, "painting_score": None}


def test_evaluate_counts_and_metrics():
    records = [
        record("portrait", "portrait", path="a"),
        record("portrait", "landscape", path="b"),
        record("landscape", "landscape", path="c"),
        record("still_life", "still_life", 0.4, UNSURE, path="d"),
        record("cityscape", "cityscape", 0.9, REJECTED, path="e"),  # a painting refused by mistake
        record(NOT_A_PAINTING, "portrait", 0.9, ANSWERED, path="f"),
        record(NOT_A_PAINTING, "portrait", 0.9, REJECTED, path="g"),
        record(NOT_A_PAINTING, "portrait", 0.3, UNSURE, path="h"),
    ]
    s = evaluate(records, rejection_enabled=True, min_confidence=0.5, n_boot=200)
    g = s["genre"]
    assert (s["n_photos"], s["n_genre_photos"], s["n_non_painting_photos"]) == (8, 5, 3)
    assert g["coverage"] == pytest.approx(3 / 5)
    assert g["unsure_rate"] == pytest.approx(1 / 5) and g["wrongly_rejected_rate"] == pytest.approx(1 / 5)
    assert g["accuracy_on_answered"]["accuracy"] == pytest.approx(2 / 3)  # a, b, c answered: a and c right
    assert g["accuracy_forced_top1"]["accuracy"] == pytest.approx(4 / 5)  # every genre photo, ignoring unsure/refused
    low, high = g["accuracy_forced_top1"]["low"], g["accuracy_forced_top1"]["high"]
    assert low <= 0.8 <= high
    labels, counts = g["confusion_forced_top1"]["labels"], np.array(g["confusion_forced_top1"]["counts"])
    assert counts[labels.index("portrait"), labels.index("landscape")] == 1 and counts.sum() == 5
    assert g["per_class_forced_top1"]["per_class"]["landscape"]["recall"] == 1.0
    o = s["non_paintings"]
    assert (o["answered_rate"], o["rejected_rate"], o["unsure_rate"]) == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert o["false_accept_rate"] == pytest.approx(2 / 3)  # answered or unsure: not refused
    json.dumps(s)  # the summary is written as JSON


def test_false_accept_rate_needs_rejection_to_be_on():
    s = evaluate([record("portrait", "portrait"), record(NOT_A_PAINTING, "portrait")], rejection_enabled=False, n_boot=50)
    assert s["non_paintings"]["false_accept_rate"] is None and s["non_paintings"]["answered_rate"] == 1.0


def test_summary_without_answered_photos_or_without_non_paintings():
    s = evaluate([record("portrait", "portrait", 0.1, UNSURE)], n_boot=50)
    assert s["genre"]["accuracy_on_answered"] is None and s["genre"]["coverage"] == 0
    assert "non_paintings" not in s
    assert "n/a" in format_summary(s)


def test_intervals_resample_paintings_not_photos():
    # 20 shots of 4 paintings: a painting is right or wrong as a whole, so the interval must be wide.
    clustered = [record("portrait", "portrait" if k < 2 else "landscape", path=str(i), painting_id=f"portrait/p{k}") for k in range(4) for i in range(5 * k, 5 * k + 5)]
    independent = [{**r, "painting_id": r["path"]} for r in clustered]
    c = evaluate(clustered, n_boot=500)["genre"]["accuracy_forced_top1"]
    i = evaluate(independent, n_boot=500)["genre"]["accuracy_forced_top1"]
    assert c["n_paintings"] == 4 and i["n_paintings"] == 20
    assert c["accuracy"] == i["accuracy"] == pytest.approx(0.5)
    assert (c["high"] - c["low"]) > (i["high"] - i["low"])


def test_format_summary_is_readable():
    s = evaluate([record("portrait", "portrait"), record(NOT_A_PAINTING, "portrait", status=REJECTED)], rejection_enabled=True, min_confidence=0.8, n_boot=50)
    text = format_summary(s, skipped=[{"path": "x.jpg", "reason": "bad"}], notes=["ignored folder 'holidays'"])
    for fragment in ("unsure' below 0.80", "refuse non-paintings", "100.0%", "false-accept rate 0.0%", "skipped", "holidays"):
        assert fragment in text


def test_write_records_and_figure(tmp_path):
    records = [record("portrait", "portrait", path="a.jpg"), {**record(NOT_A_PAINTING, "portrait", path="b.jpg"), "painting_score": 1.5}]
    write_records(records, tmp_path / "out" / "predictions.csv")
    rows = list(csv.DictReader((tmp_path / "out" / "predictions.csv").open()))
    assert list(rows[0]) == ["path", "true_label", "painting_id", "predicted", "probability", "status", "painting_score"]
    assert rows[0]["painting_score"] == "" and rows[1]["painting_score"] == "1.5"
    plot_confusion(evaluate(records, n_boot=50), tmp_path / "out" / "cm.png")
    assert (tmp_path / "out" / "cm.png").stat().st_size > 0
    plot_confusion(evaluate([record(NOT_A_PAINTING, "portrait")], n_boot=50), tmp_path / "out" / "none.png")
    assert not (tmp_path / "out" / "none.png").exists()  # no genre photo: nothing to draw


# ---- the demo set -------------------------------------------------------------------------------


@pytest.fixture
def dataset(tmp_path):
    """A tiny stand-in for data/wikiart_genre5: 5 genres x 40 solid-colour images, one artist each."""
    root = tmp_path / "wikiart_genre5"
    rows = []
    for idx, label in enumerate(CLASSES):
        for k in range(40):
            rel = f"{label}/i{idx}_{k}.jpg"
            save(root / rel, (RED, BLUE, RED, BLUE, GREEN)[idx], size=(32, 32))
            rows.append({"path": rel, "label": label, "label_idx": idx, "id": f"i{idx}_{k}", "artist": f"artist{idx}_{k}"})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)
    return root


def test_build_demo_set(dataset, tmp_path):
    out = build_demo_set(tmp_path / "demo", dataset, per_class=2)
    for genre in CLASSES:
        files = sorted((out / genre).glob("*.jpg"))
        assert 1 <= len(files) <= 2
        assert all("__" in f.stem for f in files)  # <imageid>__<perturbation>: the painting-id convention
    assert len(list((out / NOT_A_PAINTING).glob("*.png"))) == 4
    photos, _ = scan_photo_dir(out)
    assert all(p.painting_id.startswith(f"{p.label}/") for p in photos)


# ---- the script ---------------------------------------------------------------------------------


def run_script(argv, encoder, capsys):
    script = load_script("evaluate_photos")
    code = script.main(argv, encoder=encoder)
    return code, capsys.readouterr()


def test_script_without_refusal_with_a_threshold(photo_dir, probe_file, tmp_path, capsys):
    out = tmp_path / "out"
    code, cap = run_script([str(photo_dir), "--probe", probe_file, "--out", str(out), "--min-confidence", "0.5"], FakeEncoder(), capsys)
    assert code == 0
    s = json.loads((out / "summary.json").read_text())
    assert (out / "predictions.csv").exists() and (out / "confusion_matrix.png").exists()
    assert s["demo"] is False and s["model"] == "clip" and s["rejection_enabled"] is False
    assert s["photo_folder_name"] == "photos" and str(photo_dir.parent) not in json.dumps(s)  # no absolute path in the summary
    assert s["genre"]["accuracy_forced_top1"]["accuracy"] == pytest.approx(3 / 4)  # the red 'cityscape' is guessed portrait
    assert s["genre"]["coverage"] == 1.0
    wall = s["non_paintings"]
    assert wall["unsure_rate"] == 1.0 and wall["false_accept_rate"] is None  # all-zero logits: 20% each, below the threshold
    assert str(photo_dir) not in (out / "predictions.csv").read_text()  # no absolute path leaks into the table
    assert "paintings:" in cap.out


def test_script_refuses_non_paintings(photo_dir, probe_file, rule_file, tmp_path, capsys):
    out = tmp_path / "out"
    code, _ = run_script([str(photo_dir), "--probe", probe_file, "--out", str(out), "--reject-non-paintings", "--rule", rule_file], FakeEncoder(), capsys)
    s = json.loads((out / "summary.json").read_text())
    assert code == 0 and s["rejection_enabled"] is True
    assert s["non_paintings"]["rejected_rate"] == 1.0 and s["non_paintings"]["false_accept_rate"] == 0.0
    assert s["genre"]["wrongly_rejected_rate"] == 0.0


def test_script_threshold_defaults_to_the_one_saved_with_the_head(photo_dir, tmp_path, capsys):
    coef = np.hstack([10 * np.eye(5), np.zeros((5, 1))])
    save_probe(ClipProbe(coef=coef, intercept=np.zeros(5), classes=CLASSES, C=1.0, temperature=1.0, min_confidence=0.9999), tmp_path / "strict.npz")
    out = tmp_path / "out"
    run_script([str(photo_dir), "--probe", str(tmp_path / "strict.npz"), "--out", str(out)], FakeEncoder(), capsys)
    assert json.loads((out / "summary.json").read_text())["min_confidence"] == 0.9999  # as scripts/guide.py
    run_script([str(photo_dir), "--probe", str(tmp_path / "strict.npz"), "--out", str(out), "--always-answer"], FakeEncoder(), capsys)
    assert json.loads((out / "summary.json").read_text())["min_confidence"] is None
    run_script([str(photo_dir), "--probe", str(tmp_path / "strict.npz"), "--out", str(out), "--min-confidence", "0.3"], FakeEncoder(), capsys)
    assert json.loads((out / "summary.json").read_text())["min_confidence"] == 0.3


def test_script_makes_the_same_decisions_as_predict(photo_dir, probe_file, rule_file, tmp_path, capsys):
    """Same photos, same flags: the harness and scripts/predict.py must give the same verdict on every photo."""
    out = tmp_path / "out"
    flags = ["--probe", probe_file, "--reject-non-paintings", "--rule", rule_file, "--min-confidence", "0.5"]
    run_script([str(photo_dir), *flags, "--out", str(out)], FakeEncoder(), capsys)
    harness = {r["path"]: r for r in csv.DictReader((out / "predictions.csv").open())}
    predict = load_script("predict")
    files = sorted(str(p) for p in photo_dir.rglob("*.png"))
    predict.main([*files, "--model", "clip", *flags, "--json"], encoder=FakeEncoder())
    verdicts = json.loads(capsys.readouterr().out)
    assert len(verdicts) == len(harness) == 5
    for path, v in verdicts.items():
        mine = harness[str(Path(path).relative_to(photo_dir))]
        status = REJECTED if v["not_a_painting"] else UNSURE if v["unsure"] else ANSWERED
        assert mine["status"] == status, path
        assert mine["predicted"] == v["predictions"][0]["class"], path


def test_script_demo(dataset, probe_file, tmp_path, capsys):
    out = tmp_path / "out"
    code, cap = run_script(["--demo", "--demo-data", str(dataset), "--probe", probe_file, "--out", str(out)], FakeEncoder(), capsys)
    assert code == 0
    s = json.loads((out / "demo" / "summary.json").read_text())
    assert s["demo"] is True and s["n_non_painting_photos"] == 4 and s["n_genre_photos"] >= 5
    assert s["photo_folder_name"] is None  # the temporary folder is not recorded
    assert "NOT results on real photos" in cap.out and "NOT real-photo results" in cap.out
    assert not (out / "summary.json").exists()  # demo output stays in its own sub-folder


def test_script_errors(photo_dir, probe_file, tmp_path, capsys):
    script = load_script("evaluate_photos")
    assert script.main([str(tmp_path / "nope"), "--probe", probe_file], encoder=FakeEncoder()) == 1
    assert "one sub-folder per label" in capsys.readouterr().err
    assert script.main([str(photo_dir), "--probe", str(tmp_path / "missing.npz")], encoder=FakeEncoder()) == 1
    assert "train_clip_probe" in capsys.readouterr().err
    assert script.main(["--demo", "--demo-data", str(tmp_path / "nodata"), "--probe", probe_file], encoder=FakeEncoder()) == 1
    assert "needs the dataset" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        script.main(["--probe", probe_file], encoder=FakeEncoder())  # neither a folder nor --demo
    (tmp_path / "all_broken" / "portrait").mkdir(parents=True)
    (tmp_path / "all_broken" / "portrait" / "x.jpg").write_text("nope")
    assert script.main([str(tmp_path / "all_broken"), "--probe", probe_file], encoder=FakeEncoder()) == 1
    assert "no photo could be read" in capsys.readouterr().err


def test_photo_dataclass_is_hashable():
    p = Photo(Path("a.jpg"), "portrait", "portrait/a")
    assert {p: 1}[p] == 1


def test_documented_options_exist_in_the_script(capsys):
    """Every --option shown in the protocol and the README for evaluate_photos.py is a real option."""
    import re

    script = load_script("evaluate_photos")
    with pytest.raises(SystemExit):
        script.main(["--help"])
    help_text = capsys.readouterr().out
    documented = set()
    for doc in (ROOT / "docs" / "REAL_PHOTO_PROTOCOL.md", ROOT / "README.md"):
        for line in doc.read_text().splitlines():
            if "scripts/evaluate_photos.py" in line and line.lstrip().startswith("python"):
                documented |= set(re.findall(r"(?<![\w-])--[a-z][a-z-]+", line))
    assert {"--demo", "--reject-non-paintings", "--model", "--always-answer"} <= documented
    assert not [flag for flag in documented if flag not in help_text]
