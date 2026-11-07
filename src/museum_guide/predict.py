"""Inference: classify a photo of a painting with the fine-tuned ResNet-18.

This is the "point the camera at a painting" step of the brief. It needs the weights written by
`scripts/train_resnet.py` (`reports/resnet18.pt`, not versioned) and runs on CPU.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18

from museum_guide.data import make_transforms

# Order of `label_idx` in the WikiArt subset (sorted genre names), see `data.class_names`.
CLASSES = ("cityscape", "landscape", "portrait", "religious_painting", "still_life")
DEFAULT_WEIGHTS = Path("reports/resnet18.pt")
SIZE = 128  # the resolution the network was trained at


def load_classifier(weights: Path = DEFAULT_WEIGHTS, n_classes: int = len(CLASSES)) -> nn.Module:
    """ResNet-18 with the trained weights, in eval mode on CPU. No ImageNet download is needed."""
    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"{weights} not found: train the model first with `python scripts/train_resnet.py`.")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    return model.eval()


def open_image(path: Path | str) -> Image.Image:
    """Open a photo as RGB, applying its EXIF rotation (phone photos are often stored sideways)."""
    with Image.open(path) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


@torch.no_grad()
def classify(
    model: nn.Module,
    images: Sequence[Image.Image],
    classes: Sequence[str] = CLASSES,
    top_k: int = 3,
    temperature: float = 1.0,
) -> list[list[tuple[str, float]]]:
    """For each image, its `top_k` most likely classes as (name, probability), best first.

    `temperature` divides the logits before the softmax (see `museum_guide.calibration`): it makes the
    probabilities honest without changing the ranking. The default 1.0 is the raw network output.
    """
    transform = make_transforms(SIZE, augment=False)  # same resize + centre crop as at evaluation time
    batch = torch.stack([transform(img.convert("RGB")) for img in images])
    top = (model.eval()(batch) / temperature).softmax(dim=1).topk(min(top_k, len(classes)), dim=1)
    return [
        [(classes[int(i)], float(p)) for p, i in zip(probs, idx)]
        for probs, idx in zip(top.values, top.indices)
    ]
