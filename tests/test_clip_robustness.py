import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

from museum_guide.clip import normalise
from museum_guide.clip_robustness import (
    answer_stats,
    change_rate,
    embed_shifted,
    mean_over_seeds,
    paired_summary,
    shift_pixel_values,
)
from museum_guide.data import IMAGENET_MEAN, IMAGENET_STD
from museum_guide.perturb import photo_augment_pil, photo_train_transform

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "clip_photo_robustness.py"


# ---- the shift --------------------------------------------------------------------------------


def test_shift_is_a_circular_roll_down_and_right():
    x = torch.arange(16.0).reshape(1, 1, 4, 4)
    out = shift_pixel_values(x, 0.5)  # 2 pixels
    assert torch.equal(out, torch.roll(x, shifts=(2, 2), dims=(2, 3)))
    assert out[0, 0, 2, 2] == x[0, 0, 0, 0]  # the top-left pixel moved down and right
    assert out[0, 0, 0, 0] == x[0, 0, 2, 2]  # and the opposite corner wrapped around


def test_shift_follows_the_clip_input_size_convention():
    x = torch.zeros(2, 3, 224, 224)
    x[:, :, 0, 0] = 1.0
    for fraction, pixels in ((0.0625, 14), (0.125, 28)):
        out = shift_pixel_values(x, fraction)
        assert out[0, 0, pixels, pixels] == 1.0
        assert out[0, 0, 0, 0] == 0.0


def test_shift_is_at_least_one_pixel_and_keeps_values():
    x = torch.rand(1, 3, 8, 8)
    out = shift_pixel_values(x, 0.001)
    assert not torch.equal(out, x)
    assert torch.allclose(out.sum(), x.sum())  # a roll only moves pixels


class _FakeProcessor:
    """Turns PIL images into a (N, 3, 8, 8) tensor with no resizing."""

    def __call__(self, images, return_tensors):
        arrays = [torch.tensor(np.asarray(i), dtype=torch.float32).permute(2, 0, 1) for i in images]
        return {"pixel_values": torch.stack(arrays)}


class _FakeModel:
    """The 'vector' of an image is its top-left pixel (3 numbers): it changes when the picture is shifted."""

    def get_image_features(self, pixel_values):
        return pixel_values[:, :, 0, 0]


def _ramp_image():
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[..., 0] = np.arange(8)[:, None] * 30 + 10  # grows downwards
    arr[..., 1] = np.arange(8)[None, :] * 30 + 10  # grows to the right
    arr[..., 2] = 50
    return Image.fromarray(arr)


def test_embed_shifted_embeds_the_shifted_picture_and_normalises():
    img = _ramp_image()
    out = embed_shifted(_FakeModel(), _FakeProcessor(), [img], 0.25)  # 2 px
    expected = normalise(np.asarray(img, dtype=np.float32)[6, 6][None, :])  # new top-left = old (6, 6)
    assert out.shape == (1, 3)
    assert np.allclose(out, expected, atol=1e-6)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-6)


def test_embed_shifted_batches_give_the_same_result():
    images = [_ramp_image(), _ramp_image().rotate(90), _ramp_image().transpose(Image.FLIP_LEFT_RIGHT)]
    a = embed_shifted(_FakeModel(), _FakeProcessor(), images, 0.25, batch_size=1)
    b = embed_shifted(_FakeModel(), _FakeProcessor(), images, 0.25, batch_size=3)
    assert np.allclose(a, b)


# ---- what the kiosk sees ------------------------------------------------------------------------


def test_answer_stats():
    confidence = np.array([0.95, 0.90, 0.60, 0.40])
    correct = np.array([True, True, False, False])
    s = answer_stats(confidence, correct, threshold=0.8)
    assert s["answered_share"] == 0.5
    assert s["accuracy_when_answered"] == 1.0
    assert s["mean_confidence"] == pytest.approx(0.7125)
    assert s["mean_confidence_when_wrong"] == pytest.approx(0.5)


def test_answer_stats_without_threshold_answers_everything():
    s = answer_stats(np.array([0.5, 0.6]), np.array([True, False]), None)
    assert s["answered_share"] == 1.0
    assert s["accuracy_when_answered"] == 0.5


def test_answer_stats_when_nothing_is_answered_or_nothing_is_wrong():
    s = answer_stats(np.array([0.2, 0.3]), np.array([True, True]), threshold=0.9)
    assert s["answered_share"] == 0.0
    assert s["accuracy_when_answered"] is None
    assert s["mean_confidence_when_wrong"] is None


def test_change_rate():
    assert change_rate([0, 1, 2, 3], [0, 1, 0, 0]) == 0.5
    assert change_rate([1, 1], [1, 1]) == 0.0


def test_paired_summary():
    s = paired_summary([0.8, 0.7, 0.6], [0.7, 0.7, 0.8])
    assert s["mean_diff"] == pytest.approx(-0.0333333, abs=1e-6)
    assert (s["seeds_a_better"], s["seeds_b_better"], s["n_seeds"]) == (1, 1, 3)


def test_mean_over_seeds():
    out = mean_over_seeds([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 2.0}], ["a", "b"])
    assert out["a"]["mean"] == 2.0
    assert out["a"]["std"] == pytest.approx(1.4142135)
    assert out["b"]["std"] == 0.0


# ---- the augmentation refactor keeps the ResNet study's behaviour -------------------------------


def test_photo_train_transform_is_the_pil_augmentation_plus_tensor_conversion():
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8))
    finish = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    for seed in range(3):
        torch.manual_seed(seed)
        a = photo_train_transform(32)(img)
        torch.manual_seed(seed)
        b = finish(photo_augment_pil(32)(img))
        assert torch.equal(a, b)


def test_photo_augment_pil_returns_a_pil_image_of_the_asked_size():
    out = photo_augment_pil(40)(Image.new("RGB", (64, 48), (120, 80, 200)))
    assert isinstance(out, Image.Image)
    assert out.size == (40, 40)


# ---- the study itself, on synthetic vectors ------------------------------------------------------


@pytest.fixture(scope="module")
def script():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("clip_photo_robustness_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_collection(n=300, dim=16, n_classes=5, seed=0):
    import pandas as pd

    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    centres = rng.normal(size=(n_classes, dim)) * 3
    artists = [f"artist_{i // 6}" for i in range(n)]
    df = pd.DataFrame({"id": [f"id{i}" for i in range(n)], "artist": artists, "label_idx": y})
    df["group"] = df["artist"]
    return df, y, centres, rng


def test_run_seed_on_synthetic_vectors(script):
    df, y, centres, rng = _synthetic_collection()
    base = centres[y] + rng.normal(size=(len(y), 16))
    V = {"clean": normalise(base)}
    for cond in (*script.PERTURBATIONS, *script.SHIFTS):
        V[cond] = normalise(centres[y] + rng.normal(size=(len(y), 16)) * 2.5)  # noisier than clean
    for k in range(2):
        V[f"aug_{k}"] = normalise(centres[y] + rng.normal(size=(len(y), 16)) * 2.0)
    row = script.run_seed(0, df, V, y, aug_draws=2)

    assert row["seed"] == 0 and row["n_test"] > 20
    for variant in script.VARIANTS:
        head = row[variant]
        assert head["temperature"] > 0
        for cond in script.ALL_CONDITIONS:
            r = head[cond]
            assert 0.0 <= r["accuracy"] <= 1.0
            assert 0.0 <= r["answered_share"] <= 1.0
            assert 0.0 <= r["prediction_change_rate"] <= 1.0
        assert head["clean"]["prediction_change_rate"] == 0.0  # clean vs itself
        assert head["clean"]["accuracy"] > 0.8  # separable data
        assert head["clean"]["accuracy"] >= head["rotation_15deg"]["accuracy"] - 0.05  # noisier conditions do not help


def test_run_seed_is_deterministic(script):
    df, y, centres, rng = _synthetic_collection()
    V = {c: normalise(centres[y] + rng.normal(size=(len(y), 16))) for c in
         ("clean", *script.PERTURBATIONS, *script.SHIFTS, "aug_0")}
    a = script.run_seed(1, df, V, y, aug_draws=1)
    b = script.run_seed(1, df, V, y, aug_draws=1)
    assert a == b


def test_summarise_with_and_without_the_resnet_reference(script):
    df, y, centres, rng = _synthetic_collection()
    V = {c: normalise(centres[y] + rng.normal(size=(len(y), 16))) for c in
         ("clean", *script.PERTURBATIONS, *script.SHIFTS, "aug_0")}
    per_seed = [script.run_seed(s, df, V, y, aug_draws=1) for s in range(2)]
    plain = script.summarise(per_seed, None)
    assert set(plain["accuracy"]) == {"standard", "augmented"}
    assert "clip_standard_minus_resnet_standard" not in plain
    resnet = {"per_seed": [{"standard": {c: 0.5 for c in script.CONDITIONS}} for _ in per_seed]}
    full = script.summarise(per_seed, resnet)
    assert full["clip_standard_minus_resnet_standard"]["clean"]["n_seeds"] == 2
    assert "mean_over_perturbations" in full["augmented_minus_standard"]
    assert full["kiosk_behaviour"]["standard"]["clean"]["answered_share"] <= 1.0


def test_answer_stats_answers_a_confidence_equal_to_the_threshold():
    s = answer_stats(np.array([0.8, 0.79]), np.array([True, False]), threshold=0.8)
    assert s["answered_share"] == 0.5  # 'at least the threshold', like predict.py's unsure rule
    assert s["accuracy_when_answered"] == 1.0
