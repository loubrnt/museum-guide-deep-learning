"""CLIP helpers: image and text embeddings, zero-shot classification, linear probe, embedding cache.

A CLIP dual encoder maps images and sentences into one vector space. The same image vectors serve
recognition (compare with the class names, or train a linear head on them) and natural-language search
(compare with a question). Heavy imports (torch, transformers, PIL) are done inside the functions that
need them, so the pure functions can be imported and tested quickly.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

MODEL = "openai/clip-vit-base-patch32"
# Two ways to turn a class name into a sentence. Neither is tuned on the test set.
PROMPT_TEMPLATES = {"simple": "a {name}", "painting": "a painting of a {name}"}


def class_prompts(classes: Sequence[str], template: str) -> list[str]:
    """One sentence per class: 'religious_painting' -> 'a painting of a religious painting'."""
    return [template.format(name=c.replace("_", " ")) for c in classes]


def normalise(x: np.ndarray) -> np.ndarray:
    """L2-normalise the rows, so a dot product is a cosine similarity."""
    x = np.asarray(x, dtype=np.float32)
    return x / np.clip(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12, None)


def _as_tensor(out):
    """transformers 5.x may return an output object instead of a tensor: take its pooled vector."""
    return out if hasattr(out, "shape") else out.pooler_output


def load_clip(model_name: str = MODEL):
    """The CLIP model (eval mode, CPU) and its processor. Downloads the weights on first use."""
    from transformers import CLIPModel, CLIPProcessor

    return CLIPModel.from_pretrained(model_name).eval(), CLIPProcessor.from_pretrained(model_name)


def _embed_batch(model, processor, images) -> np.ndarray:
    """Raw (not yet normalised) CLIP vectors for a list of RGB PIL images."""
    import torch

    with torch.no_grad():
        out = _as_tensor(model.get_image_features(**processor(images=list(images), return_tensors="pt")))
    return out.cpu().numpy()


def embed_pil_images(model, processor, images: Sequence, batch_size: int = 64) -> np.ndarray:
    """Normalised image vectors for PIL images already in memory (a photo from the camera), one row per image."""
    images = [img.convert("RGB") for img in images]
    vecs = [_embed_batch(model, processor, images[i : i + batch_size]) for i in range(0, len(images), batch_size)]
    return normalise(np.concatenate(vecs))


def embed_images(model, processor, paths: Sequence[Path | str], batch_size: int = 64) -> np.ndarray:
    """Normalised image vectors, one row per path (files are opened one batch at a time)."""
    from PIL import Image

    vecs = []
    for i in range(0, len(paths), batch_size):
        images = []
        for p in paths[i : i + batch_size]:
            with Image.open(p) as img:
                images.append(img.convert("RGB"))
        vecs.append(_embed_batch(model, processor, images))
    return normalise(np.concatenate(vecs))


def embed_texts(model, processor, texts: Sequence[str]) -> np.ndarray:
    """Normalised text vectors, one row per sentence."""
    import torch

    with torch.no_grad():
        out = _as_tensor(model.get_text_features(**processor(text=list(texts), return_tensors="pt", padding=True)))
    return normalise(out.cpu().numpy())


def zero_shot_predict(image_emb: np.ndarray, text_emb: np.ndarray) -> np.ndarray:
    """Class of the closest sentence for each image: no training, only the class names."""
    return (np.asarray(image_emb) @ np.asarray(text_emb).T).argmax(axis=1)


def fit_linear_probe(train_X, train_y, val_X, val_y, Cs: Sequence[float] = (0.1, 1, 10, 100)):
    """Logistic regression on frozen embeddings. The regularisation C is chosen on the validation set.

    Returns (fitted classifier, chosen C). Ties go to the smaller C (the stronger regularisation).
    """
    from sklearn.linear_model import LogisticRegression

    best = None
    for C in sorted(Cs):
        clf = LogisticRegression(C=C, max_iter=3000).fit(train_X, train_y)
        score = clf.score(val_X, val_y)
        if best is None or score > best[0]:
            best = (score, clf, C)
    return best[1], best[2]


def cache_key(model_name: str, paths: Sequence[Path | str]) -> str:
    """A short fingerprint of the model and the ordered list of images."""
    h = hashlib.sha1(model_name.encode())
    for p in paths:
        h.update(b"\0" + str(p).encode())
    return h.hexdigest()


def cached_embeddings(
    cache_file: Path | str, model_name: str, paths: Sequence[Path | str], compute: Callable[[], np.ndarray]
) -> np.ndarray:
    """Load the embeddings from `cache_file` if it matches (model, paths); otherwise compute and save them."""
    cache_file, key = Path(cache_file), cache_key(model_name, paths)
    if cache_file.exists():
        with np.load(cache_file) as f:
            if str(f["key"]) == key:
                return f["embeddings"]
    emb = compute()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, key=key, embeddings=emb)
    return emb


class ClipEncoder:
    """One loaded CLIP model that turns photos into vectors (`images`) and questions into vectors (`texts`).

    The guide loads CLIP once and shares it between recognition, similar paintings and search; tests pass
    a fake object with the same two methods instead.
    """

    def __init__(self, model, processor, model_name: str = MODEL):
        self.model, self.processor, self.model_name = model, processor, model_name

    def images(self, images: Sequence, batch_size: int = 32) -> np.ndarray:
        return embed_pil_images(self.model, self.processor, images, batch_size)

    def texts(self, texts: Sequence[str]) -> np.ndarray:
        return embed_texts(self.model, self.processor, texts)


def load_encoder(model_name: str = MODEL) -> ClipEncoder:
    """Load CLIP (downloads the weights on first use) and wrap it as a `ClipEncoder`."""
    model, processor = load_clip(model_name)
    return ClipEncoder(model, processor, model_name)
