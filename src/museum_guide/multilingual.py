"""A multilingual text encoder in the SAME space as the CLIP image vectors of the index.

`sentence-transformers/clip-ViT-B-32-multilingual-v1` is a student text encoder trained to put sentences in many
languages (French included) where the original OpenAI CLIP ViT-B/32 text encoder puts their English translation.
Its image side is the unchanged OpenAI CLIP, so the image vectors already in the index can be searched with it.

It is an optional dependency (`pip install -e ".[multilingual]"`), imported only when used.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from museum_guide.clip import normalise

MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"


def load_multilingual(model_name: str = MODEL):
    """The multilingual text encoder (downloads about 500 MB on first use)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as err:
        raise ImportError(
            "the multilingual encoder needs sentence-transformers: pip install -e '.[multilingual]'"
        ) from err
    return SentenceTransformer(model_name)


def embed_texts_multilingual(model, texts: Sequence[str]) -> np.ndarray:
    """Normalised text vectors, one row per sentence, comparable with the CLIP image vectors."""
    return normalise(model.encode(list(texts), convert_to_numpy=True, show_progress_bar=False))
