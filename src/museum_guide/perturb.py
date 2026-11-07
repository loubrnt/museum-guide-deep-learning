"""Camera-like perturbations, and a training augmentation that imitates them.

The museum guide will see phone photos, not the clean WikiArt scans it is trained on. We have no
labelled phone photos, so `PERTURBATIONS` is a proxy: deterministic, named, one effect at a time
(a tilted phone, a defocused shot, an oblique view, dim or bright rooms, a close-up...).
`photo_train_transform` is the matching *random* augmentation for training.
"""
from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF

from museum_guide.data import DATA_DIR, IMAGENET_MEAN, IMAGENET_STD, WikiArt, make_transforms


def rotate_15(img: Image.Image) -> Image.Image:
    """Phone held at a slant: 15 degrees counter-clockwise, black corners."""
    return TF.rotate(img, 15)


def perspective(img: Image.Image) -> Image.Image:
    """Oblique view: the four corners move by up to ~9% of the width/height (fixed offsets)."""
    w, h = img.size
    start = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    end = [[0.09 * w, 0.06 * h], [0.90 * w, 0.0], [w - 1, h - 1], [0.0, 0.92 * h]]
    return TF.perspective(img, start, end)


def darker(img: Image.Image) -> Image.Image:
    """Dim gallery: brightness x0.5."""
    return ImageEnhance.Brightness(img).enhance(0.5)


def brighter(img: Image.Image) -> Image.Image:
    """Spot light: brightness x1.6."""
    return ImageEnhance.Brightness(img).enhance(1.6)


def low_contrast(img: Image.Image) -> Image.Image:
    """Glare or haze: contrast x0.5."""
    return ImageEnhance.Contrast(img).enhance(0.5)


def blur(img: Image.Image) -> Image.Image:
    """Out-of-focus shot: Gaussian blur, radius 2 px at 128 px (scaled with the image width)."""
    return img.filter(ImageFilter.GaussianBlur(2 * img.size[0] / 128))


def jpeg(img: Image.Image) -> Image.Image:
    """Heavy compression: JPEG quality 15."""
    buffer = io.BytesIO()
    img.save(buffer, "JPEG", quality=15)
    return Image.open(buffer).convert("RGB")


def close_up(img: Image.Image) -> Image.Image:
    """Visitor stands close: the central 60% of the painting, scaled back up."""
    w, h = img.size
    return TF.resize(TF.center_crop(img, [round(0.6 * h), round(0.6 * w)]), [h, w])


def grayscale(img: Image.Image) -> Image.Image:
    """Black-and-white filter."""
    return TF.rgb_to_grayscale(img, 3)


PERTURBATIONS: dict[str, Callable[[Image.Image], Image.Image]] = {
    "rotation_15deg": rotate_15,
    "perspective": perspective,
    "blur": blur,
    "close_up_60pct": close_up,
    "darker": darker,
    "brighter": brighter,
    "low_contrast": low_contrast,
    "jpeg_q15": jpeg,
    "grayscale": grayscale,
}


def photo_augment_steps(size: int) -> list:
    """The random, PIL-level part of the photo augmentation: crop, flip, tilt, oblique view, light, blur."""
    return [
        transforms.RandomResizedCrop(size, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
    ]


def photo_augment_pil(size: int) -> transforms.Compose:
    """`photo_train_transform` without the tensor conversion: a random augmentation PIL image -> PIL image.

    Used to embed augmented images with a model that has its own preprocessing (CLIP).
    """
    return transforms.Compose(photo_augment_steps(size))


def photo_train_transform(size: int) -> transforms.Compose:
    """Random training augmentation imitating `PERTURBATIONS`: crop, tilt, oblique view, light, blur, flip."""
    return transforms.Compose(
        [*photo_augment_steps(size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    )


def loaders_with_train_transform(
    parts: dict[str, pd.DataFrame],
    size: int,
    train_transform,
    batch_size: int = 32,
    data_dir: Path = DATA_DIR,
) -> dict[str, DataLoader]:
    """Like `data.loaders`, but with any transform on the train part; val/test stay un-augmented."""
    plain = make_transforms(size, augment=False)
    return {
        name: DataLoader(
            WikiArt(df, train_transform if name == "train" else plain, data_dir),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, df in parts.items()
    }
