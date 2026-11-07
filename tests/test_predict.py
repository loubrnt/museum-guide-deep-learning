import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.models import resnet18

from museum_guide.predict import CLASSES, classify, load_classifier, open_image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "predict.py"


@pytest.fixture
def weights(tmp_path):
    """Random ResNet-18 weights with a 5-class head, saved like scripts/train_resnet.py does."""
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


def test_classes_match_the_dataset_label_order():
    assert list(CLASSES) == sorted(CLASSES)  # label_idx = position in the sorted genre names
    assert len(CLASSES) == 5


def test_load_classifier_reports_missing_weights(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_resnet"):
        load_classifier(tmp_path / "missing.pt")


def test_classify_returns_sorted_probabilities(weights, photo):
    model = load_classifier(weights)
    (top,) = classify(model, [open_image(photo)], top_k=3)
    assert len(top) == 3
    names, probs = zip(*top)
    assert set(names) <= set(CLASSES)
    assert list(probs) == sorted(probs, reverse=True)
    assert 0 < sum(probs) <= 1 + 1e-6


def test_classify_top_k_is_capped_and_covers_all_classes(weights, photo):
    (top,) = classify(load_classifier(weights), [open_image(photo)], top_k=99)
    assert len(top) == len(CLASSES)
    assert sum(p for _, p in top) == pytest.approx(1.0, abs=1e-5)


def test_classify_handles_a_batch_and_any_image_size(weights):
    model = load_classifier(weights)
    images = [Image.new("RGB", (30, 200)), Image.new("RGB", (500, 40)), Image.new("L", (64, 64))]
    results = classify(model, images)
    assert len(results) == 3


def test_open_image_applies_exif_rotation(tmp_path):
    path = tmp_path / "sideways.jpg"
    img = Image.new("RGB", (40, 20))
    exif = img.getexif()
    exif[0x0112] = 6  # orientation: rotate 90 degrees clockwise to display
    img.save(path, exif=exif)
    assert open_image(path).size == (20, 40)


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)


def test_cli_json_output(weights, photo):
    out = _run(photo, "--weights", weights, "--top-k", "2", "--json")
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert len(data[str(photo)]) == 2
    assert {"class", "probability"} == set(data[str(photo)][0])


def test_cli_text_output(weights, photo):
    out = _run(photo, "--weights", weights)
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith(f"{photo}: ")
    assert "%" in out.stdout


def test_cli_fails_cleanly_without_weights(tmp_path, photo):
    out = _run(photo, "--weights", tmp_path / "nope.pt")
    assert out.returncode == 1
    assert "train_resnet" in out.stderr
    assert "Traceback" not in out.stderr


def test_cli_skips_unreadable_files_but_reports_failure(weights, photo, tmp_path):
    broken = tmp_path / "broken.jpg"
    broken.write_text("not an image")
    out = _run(broken, photo, "--weights", weights)
    assert out.returncode == 1
    assert "cannot read image" in out.stderr
    assert str(photo) in out.stdout  # the good file was still classified
