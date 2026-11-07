import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from museum_guide.perturb import PERTURBATIONS, loaders_with_train_transform, photo_train_transform


@pytest.fixture
def painting():
    """A 64x48 image with structure (gradient + a bright square), so every perturbation visibly changes it."""
    arr = np.zeros((48, 64, 3), dtype=np.uint8)
    arr[..., 0] = np.linspace(40, 220, 64, dtype=np.uint8)[None, :]
    arr[..., 1] = np.linspace(200, 30, 48, dtype=np.uint8)[:, None]
    arr[..., 2] = 120
    arr[10:25, 20:40] = (250, 250, 20)
    return Image.fromarray(arr)


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_perturbation_keeps_size_and_mode(name, painting):
    out = PERTURBATIONS[name](painting)
    assert out.size == painting.size
    assert out.mode == "RGB"


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_perturbation_is_deterministic(name, painting):
    a, b = PERTURBATIONS[name](painting), PERTURBATIONS[name](painting)
    assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("name", PERTURBATIONS)
def test_perturbation_changes_the_pixels(name, painting):
    out = np.asarray(PERTURBATIONS[name](painting)).astype(int)
    assert np.abs(out - np.asarray(painting).astype(int)).mean() > 1.0


def test_perturbations_do_not_modify_their_input(painting):
    before = np.asarray(painting).copy()
    for fn in PERTURBATIONS.values():
        fn(painting)
    assert np.array_equal(np.asarray(painting), before)


def test_brightness_perturbations_go_in_the_right_direction(painting):
    mean = lambda im: np.asarray(im, dtype=float).mean()  # noqa: E731
    assert mean(PERTURBATIONS["darker"](painting)) < mean(painting) < mean(PERTURBATIONS["brighter"](painting))


def test_grayscale_has_equal_channels(painting):
    arr = np.asarray(PERTURBATIONS["grayscale"](painting))
    assert np.array_equal(arr[..., 0], arr[..., 1]) and np.array_equal(arr[..., 1], arr[..., 2])


def test_blur_scales_with_image_size():
    # The same drawing at 2x resolution is blurred by 2x the radius: the result is about as smooth.
    rng = np.random.default_rng(0)
    small = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    large = small.resize((128, 128), Image.NEAREST)
    s = np.asarray(PERTURBATIONS["blur"](small), dtype=float)
    lg = np.asarray(PERTURBATIONS["blur"](large).resize((64, 64)), dtype=float)
    assert np.abs(s - lg).mean() < 6  # about 2.4 with scaling; 10.6 if the radius were a fixed 2 px


def test_photo_train_transform_output(painting):
    out = photo_train_transform(32)(painting)
    assert out.shape == (3, 32, 32)
    assert out.dtype == torch.float32


def test_photo_train_transform_is_random(painting):
    t = photo_train_transform(32)
    outs = [t(painting) for _ in range(6)]
    assert any(not torch.equal(outs[0], o) for o in outs[1:])


def test_loaders_augment_only_the_train_part(tmp_path):
    rows = []
    for i in range(4):
        (tmp_path / "a").mkdir(exist_ok=True)
        Image.fromarray(np.full((40, 40, 3), 30 * i, dtype=np.uint8)).save(tmp_path / "a" / f"{i}.jpg")
        rows.append({"path": f"a/{i}.jpg", "label_idx": i % 2})
    df = pd.DataFrame(rows)
    train_t = photo_train_transform(16)
    dl = loaders_with_train_transform({"train": df, "val": df, "test": df}, 16, train_t, batch_size=2, data_dir=tmp_path)
    assert dl["train"].dataset.transform is train_t
    assert dl["val"].dataset.transform is not train_t and dl["test"].dataset.transform is dl["val"].dataset.transform
    x, y = next(iter(dl["val"]))
    assert x.shape == (2, 3, 16, 16) and y.shape == (2,)
