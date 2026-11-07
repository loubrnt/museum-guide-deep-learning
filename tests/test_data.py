"""The artist-disjoint split is the methodological core of the project: test it first."""
import pandas as pd
import torch

from museum_guide.data import (
    IMAGENET_MEAN,
    WikiArt,
    class_names,
    load_metadata,
    loaders,
    make_transforms,
    split,
)


def _with_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["group"] = df["artist"].where(df["artist"] != "Unknown Artist", "unknown_" + df["id"])
    return df


def test_split_has_no_artist_in_two_parts(metadata):
    parts = split(_with_groups(metadata))
    named = {k: set(v["artist"]) - {"Unknown Artist"} for k, v in parts.items()}
    assert not named["train"] & named["val"]
    assert not named["train"] & named["test"]
    assert not named["val"] & named["test"]


def test_split_is_a_partition_of_the_rows(metadata):
    parts = split(_with_groups(metadata))
    ids = pd.concat([p["id"] for p in parts.values()])
    assert len(ids) == len(metadata)
    assert ids.is_unique
    assert set(ids) == set(metadata["id"])


def test_split_sizes_are_roughly_70_15_15(metadata):
    parts = split(_with_groups(metadata))
    total = len(metadata)
    assert 0.60 < len(parts["train"]) / total < 0.80
    assert 0.08 < len(parts["val"]) / total < 0.25
    assert 0.08 < len(parts["test"]) / total < 0.25


def test_split_is_reproducible_and_depends_on_seed(metadata):
    df = _with_groups(metadata)
    a, b, c = split(df, seed=0), split(df, seed=0), split(df, seed=1)
    assert a["test"]["id"].tolist() == b["test"]["id"].tolist()
    assert a["test"]["id"].tolist() != c["test"]["id"].tolist()


def test_unknown_artist_images_are_independent_groups(image_folder):
    df = load_metadata(image_folder)
    unknown = df[df["artist"] == "Unknown Artist"]
    assert len(unknown) == 6
    assert unknown["group"].is_unique  # not one catch-all group
    assert all(g.startswith("unknown_") for g in unknown["group"])
    named = df[df["artist"] != "Unknown Artist"]
    assert (named["group"] == named["artist"]).all()


def test_class_names_follow_label_idx(image_folder):
    assert class_names(load_metadata(image_folder)) == ["portrait", "landscape"]


def test_transforms_output_shape_and_normalisation(image_folder):
    from PIL import Image

    img = Image.open(image_folder / "portrait" / "img00.jpg").convert("RGB")  # all-black 40x30 image
    out = make_transforms(16, augment=False)(img)
    assert out.shape == (3, 16, 16)
    expected = torch.tensor([-m / s for m, s in zip(IMAGENET_MEAN, (0.229, 0.224, 0.225))])
    assert torch.allclose(out[:, 0, 0], expected, atol=1e-4)
    assert make_transforms(16, augment=True)(img).shape == (3, 16, 16)


def test_dataset_and_loaders(image_folder):
    df = load_metadata(image_folder)
    ds = WikiArt(df, make_transforms(16, augment=False), image_folder)
    x, y = ds[1]
    assert x.shape == (3, 16, 16)
    assert int(y) == 1
    assert len(ds) == 12


def test_augmentation_only_applies_to_the_train_part(image_folder):
    df = load_metadata(image_folder)  # datasets only join paths at construction: no file is read here
    dl = loaders({"train": df, "val": df}, size=16, augment=True, batch_size=4)
    train_steps = dl["train"].dataset.transform.transforms
    val_steps = dl["val"].dataset.transform.transforms
    assert any("RandomResizedCrop" in type(t).__name__ for t in train_steps)
    assert not any("Random" in type(t).__name__ for t in val_steps)


def test_data_dir_can_be_overridden_with_an_environment_variable(tmp_path):
    import os
    import subprocess
    import sys

    code = "from museum_guide.data import DATA_DIR; print(DATA_DIR)"
    env = {**os.environ, "MUSEUM_GUIDE_DATA": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == str(tmp_path)
    env.pop("MUSEUM_GUIDE_DATA")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == os.path.join("data", "wikiart_genre5")
