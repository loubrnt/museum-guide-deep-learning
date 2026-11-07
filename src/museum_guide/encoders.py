"""Frozen image encoders to compare: one registry, one embedding convention each, pure helpers.

The question is which pretrained encoder gives the best linear probe for the price. Every encoder is used FROZEN:
images go in, one vector per image comes out, and a logistic regression is trained on top (see `clip.fit_linear_probe`).

Conventions, fixed in advance (never tuned on the test sets):

- CLIP and SigLIP: the model's own projected image embedding (`get_image_features`).
- DINOv2: the CLS token concatenated with the mean of the patch tokens (the protocol of the DINOv2 paper's linear evaluation).
- ResNets: the global-average-pooled features before the classifier (512 values for ResNet-18, 2048 for ResNet-50).
- Every vector is L2-normalised, so a dot product is a cosine whatever the encoder.
- The images of the project are 128 px: they are UPSAMPLED to each encoder's own input size (224 px, or 128 px for the
  ResNet-18 reference used as in the rest of the project). No encoder sees more detail than 128 px contain.

Heavy imports (torch, transformers, torchvision) are done inside the functions that need them, so the pure helpers
can be imported and tested without them.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EncoderSpec:
    name: str  # short name used in reports
    family: str  # "clip" | "siglip" | "dinov2" | "resnet"
    hf_id: str  # Hugging Face id, or the torchvision weights name for the ResNets
    input_size: int  # side of the square input, in pixels
    has_text: bool  # a text tower exists, so zero-shot classification is possible
    convention: str  # how the vector is built (written in the report)


REGISTRY: dict[str, EncoderSpec] = {
    "clip_b32": EncoderSpec("clip_b32", "clip", "openai/clip-vit-base-patch32", 224, True, "projected image embedding"),
    "clip_b16": EncoderSpec("clip_b16", "clip", "openai/clip-vit-base-patch16", 224, True, "projected image embedding"),
    "clip_l14": EncoderSpec("clip_l14", "clip", "openai/clip-vit-large-patch14", 224, True, "projected image embedding"),
    "siglip_b16": EncoderSpec("siglip_b16", "siglip", "google/siglip-base-patch16-224", 224, True, "projected image embedding"),
    "dinov2_s": EncoderSpec("dinov2_s", "dinov2", "facebook/dinov2-small", 224, False, "CLS token + mean of patch tokens"),
    "dinov2_b": EncoderSpec("dinov2_b", "dinov2", "facebook/dinov2-base", 224, False, "CLS token + mean of patch tokens"),
    "resnet50_frozen": EncoderSpec("resnet50_frozen", "resnet", "resnet50", 224, False, "global-average-pooled features, ImageNet weights"),
    "resnet18_frozen": EncoderSpec("resnet18_frozen", "resnet", "resnet18", 128, False, "global-average-pooled features, ImageNet weights"),
}


# ---------------------------------------------------------------- pure helpers


def l2_normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise the rows (zero rows stay zero)."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def pool_cls_and_mean(tokens):
    """DINOv2 convention: the CLS token (first) concatenated with the mean of the patch tokens (all the others).

    `tokens` has shape (batch, 1 + n_patches [+ registers], dim); the result (batch, 2 * dim). Works on a torch
    tensor or a numpy array.
    """
    cls, mean = tokens[:, 0], tokens[:, 1:].mean(1)
    if isinstance(tokens, np.ndarray):
        return np.concatenate([cls, mean], axis=1)
    import torch

    return torch.cat([cls, mean], dim=1)


def paired_wins(a, b) -> dict[str, int]:
    """Over matched seeds: in how many does `a` beat `b`, tie, lose."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return {"wins": int((a > b).sum()), "ties": int((a == b).sum()), "losses": int((a < b).sum())}


def feature_size_mb(n_params: int, bytes_per_param: int = 4) -> float:
    """Size of `n_params` parameters in MB (1 MB = 10^6 bytes), a cross-check for a measured weight file."""
    return n_params * bytes_per_param / 1e6


# ---------------------------------------------------------------- embedding loop


def embed_paths(
    preprocess: Callable,  # PIL image -> (3,H,W) tensor, or a batch -> pixel_values tensor, see `batched`
    forward: Callable,  # pixel_values (batch tensor on `device`) -> (batch, dim) tensor
    paths: Sequence[Path | str],
    batch_size: int = 32,
    device="cpu",
) -> np.ndarray:
    """L2-normalised vectors, one row per path. `preprocess` is called on a list of PIL images (one batch)."""
    import torch
    from PIL import Image

    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            images = []
            for p in paths[i : i + batch_size]:
                with Image.open(p) as img:
                    images.append(img.convert("RGB"))
            pixel_values = preprocess(images).to(device)
            out.append(forward(pixel_values).float().cpu().numpy())
    return l2_normalise(np.concatenate(out))


@dataclass
class Encoder:
    """A loaded encoder: embeds image files, and sentences when it has a text tower."""

    spec: EncoderSpec
    embed_images: Callable[[Sequence[Path | str]], np.ndarray]
    embed_texts: Callable[[Sequence[str]], np.ndarray] | None = None


def _tensor(out):
    """transformers 5.x may return an output object instead of a tensor: take its pooled vector."""
    return out if hasattr(out, "shape") else out.pooler_output


def load_encoder(name: str, device="cpu", batch_size: int = 32) -> Encoder:
    """Load a pretrained encoder (downloads its weights on first use) with its own preprocessing."""
    import torch

    spec = REGISTRY[name]
    if spec.family in ("clip", "siglip"):
        from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained(spec.hf_id).eval().to(device)
        processor = AutoImageProcessor.from_pretrained(spec.hf_id)
        tokenizer = AutoTokenizer.from_pretrained(spec.hf_id)

        def pre(images):
            return processor(images=images, return_tensors="pt")["pixel_values"]

        def fwd(pixel_values):
            return _tensor(model.get_image_features(pixel_values=pixel_values))

        def texts(sentences):
            kwargs = {"padding": "max_length", "max_length": 64} if spec.family == "siglip" else {"padding": True}
            tokens = tokenizer(list(sentences), return_tensors="pt", **kwargs)
            keep = ("input_ids",) if spec.family == "siglip" else ("input_ids", "attention_mask")
            with torch.no_grad():
                feats = _tensor(model.get_text_features(**{k: v.to(device) for k, v in tokens.items() if k in keep}))
            return l2_normalise(feats.float().cpu().numpy())

        return Encoder(spec, lambda paths: embed_paths(pre, fwd, paths, batch_size, device), texts)

    if spec.family == "dinov2":
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained(spec.hf_id).eval().to(device)
        processor = AutoImageProcessor.from_pretrained(spec.hf_id)

        def pre(images):
            return processor(images=images, return_tensors="pt")["pixel_values"]

        def fwd(pixel_values):
            return pool_cls_and_mean(model(pixel_values=pixel_values).last_hidden_state)

        return Encoder(spec, lambda paths: embed_paths(pre, fwd, paths, batch_size, device))

    if spec.family == "resnet":
        from torchvision import models, transforms

        weights = {"resnet50": models.ResNet50_Weights.IMAGENET1K_V2, "resnet18": models.ResNet18_Weights.IMAGENET1K_V1}[spec.hf_id]
        model = getattr(models, spec.hf_id)(weights=weights)
        model.fc = torch.nn.Identity()
        model = model.eval().to(device)
        tf = transforms.Compose([
            transforms.Resize(spec.input_size), transforms.CenterCrop(spec.input_size), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        def pre(images):
            return torch.stack([tf(im) for im in images])

        return Encoder(spec, lambda paths: embed_paths(pre, model, paths, batch_size, device))

    raise ValueError(f"unknown encoder family {spec.family!r}")


# ---------------------------------------------------------------- image towers with random weights (for timing)


def image_tower(name: str):
    """The image tower of an encoder with RANDOM weights: same shapes, size, memory and speed as the trained one.

    Only configurations are read from the hub (a few KB); no checkpoint is loaded. Returns (module in eval mode
    mapping pixel values to one vector per image, input size in pixels).
    """
    import torch
    from torch import nn

    spec = REGISTRY[name]
    if spec.family == "clip":
        from museum_guide.benchmark import clip_image_tower

        return clip_image_tower(spec.hf_id), spec.input_size
    if spec.family == "siglip":
        from transformers import SiglipConfig, SiglipVisionConfig, SiglipVisionModel

        vision_config = SiglipConfig.from_pretrained(spec.hf_id).vision_config
        if not isinstance(vision_config, SiglipVisionConfig):
            raise TypeError(f"{spec.hf_id} has no SigLIP vision configuration (got {type(vision_config).__name__})")
        vision = SiglipVisionModel(vision_config).eval()

        class SiglipTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.vision = vision

            def forward(self, pixel_values):
                return self.vision(pixel_values=pixel_values).pooler_output

        return SiglipTower().eval(), spec.input_size
    if spec.family == "dinov2":
        from transformers import AutoConfig, Dinov2Model

        backbone = Dinov2Model(AutoConfig.from_pretrained(spec.hf_id)).eval()

        class DinoTower(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone

            def forward(self, pixel_values):
                return pool_cls_and_mean(self.backbone(pixel_values=pixel_values).last_hidden_state)

        return DinoTower().eval(), spec.input_size
    if spec.family == "resnet":
        from torchvision import models

        model = getattr(models, spec.hf_id)(weights=None)
        model.fc = torch.nn.Identity()
        return model.eval(), spec.input_size
    raise ValueError(f"unknown encoder family {spec.family!r}")
