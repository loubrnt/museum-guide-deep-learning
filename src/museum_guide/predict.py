"""Inference: classify a photo of a painting, with the fine-tuned ResNet-18 or with CLIP and a linear head.

This is the "point the camera at a painting" step of the brief. The ResNet needs the weights written by
`scripts/train_resnet.py` (`reports/resnet18.pt`, not versioned); the CLIP head is a small file written by
`scripts/train_clip_probe.py` (`reports/clip_probe.npz`, versioned). Both run on CPU.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18

from museum_guide.calibration import softmax
from museum_guide.clip import MODEL as CLIP_MODEL
from museum_guide.data import make_transforms

# Order of `label_idx` in the WikiArt subset (sorted genre names), see `data.class_names`.
CLASSES = ("cityscape", "landscape", "portrait", "religious_painting", "still_life")
DEFAULT_WEIGHTS = Path("reports/resnet18.pt")
DEFAULT_PROBE = Path("reports/clip_probe.npz")
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


# ---------------------------------------------------------------------------------------------
# CLIP image encoder + linear head ("probe"): the head is a few KB, the encoder is CLIP itself.
# ---------------------------------------------------------------------------------------------


@dataclass
class ClipProbe:
    """A logistic-regression head on normalised CLIP image vectors, with its calibration.

    `temperature` divides the logits (fitted on the validation set, see `museum_guide.calibration`);
    `min_confidence` is the answer threshold chosen on the validation set (None: not set).
    """

    coef: np.ndarray  # (n_classes, dim)
    intercept: np.ndarray  # (n_classes,)
    classes: tuple[str, ...]
    C: float
    model: str = CLIP_MODEL
    split_seed: int = 0
    temperature: float = 1.0
    min_confidence: float | None = None

    def logits(self, vectors: np.ndarray) -> np.ndarray:
        """Raw scores of normalised CLIP vectors (one row per image), before any temperature."""
        return np.asarray(vectors, dtype=np.float64) @ self.coef.T.astype(np.float64) + self.intercept.astype(np.float64)


def save_probe(probe: ClipProbe, path: Path | str = DEFAULT_PROBE) -> None:
    """Write the head as a small .npz file (float32 weights: about 10 KB for 5 genres)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coef=probe.coef.astype(np.float32),
        intercept=probe.intercept.astype(np.float32),
        classes=np.array(probe.classes),
        C=np.array(probe.C),
        model=np.array(probe.model),
        split_seed=np.array(probe.split_seed),
        temperature=np.array(probe.temperature),
        min_confidence=np.array(np.nan if probe.min_confidence is None else probe.min_confidence),
    )


def load_probe(path: Path | str = DEFAULT_PROBE) -> ClipProbe:
    """Read a head saved by `save_probe`; says what to run if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found: fit it first with `python scripts/train_clip_probe.py`.")
    with np.load(path) as f:
        threshold = float(f["min_confidence"])
        return ClipProbe(
            coef=f["coef"],
            intercept=f["intercept"],
            classes=tuple(str(c) for c in f["classes"]),
            C=float(f["C"]),
            model=str(f["model"]),
            split_seed=int(f["split_seed"]),
            temperature=float(f["temperature"]),
            min_confidence=None if np.isnan(threshold) else threshold,
        )


def classify_vectors(
    probe: ClipProbe, vectors: np.ndarray, top_k: int = 3, temperature: float | None = None
) -> list[list[tuple[str, float]]]:
    """For each CLIP vector, its `top_k` most likely classes as (name, probability), best first.

    Same structure as `classify`. `temperature=None` uses the one saved with the head; pass 1.0 for raw probabilities.
    """
    t = probe.temperature if temperature is None else temperature
    probs = softmax(probe.logits(vectors), t)
    k = min(top_k, len(probe.classes))
    order = np.argsort(-probs, axis=1, kind="stable")[:, :k]
    return [[(probe.classes[i], float(row[i])) for i in idx] for row, idx in zip(probs, order)]


def classify_clip(
    encoder, probe: ClipProbe, images: Sequence[Image.Image], top_k: int = 3, temperature: float | None = None
) -> list[list[tuple[str, float]]]:
    """Classify PIL images: `encoder.images(images)` gives the CLIP vectors (a `museum_guide.clip.ClipEncoder`)."""
    return classify_vectors(probe, encoder.images([img.convert("RGB") for img in images]), top_k, temperature)
