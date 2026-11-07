"""Loaders and fallbacks that need a model, the network or specific hardware, exercised with fakes."""
import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from museum_guide import clip as clip_module
from museum_guide import data as data_module
from museum_guide import search as search_module
from museum_guide.benchmark import select_quantized_engine
from museum_guide.search import load_index, load_or_build_index


class FakeClip(nn.Module):
    def get_image_features(self, pixel_values):
        m = pixel_values.mean(dim=(2, 3))
        return torch.cat([m, torch.ones(m.shape[0], 1)], dim=1)

    def get_text_features(self, input_ids, **_):
        return torch.stack([input_ids.float().sum(1), torch.ones(input_ids.shape[0]), input_ids.float().mean(1), input_ids.float().max(1).values], dim=1)


class FakeClipProcessor:
    def __call__(self, images=None, text=None, return_tensors="pt", padding=False):
        if images is not None:
            means = torch.tensor([float(np.asarray(im).mean()) for im in images])
            return {"pixel_values": means.view(-1, 1, 1, 1).expand(-1, 3, 4, 4).clone()}
        return {"input_ids": torch.tensor([[len(t), len(t.split()), 3] for t in text])}


@pytest.fixture
def fake_clip_package(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.CLIPModel = types.SimpleNamespace(from_pretrained=lambda name: FakeClip())
    fake.CLIPProcessor = types.SimpleNamespace(from_pretrained=lambda name: FakeClipProcessor())
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _jpeg(path, level):
    Image.new("RGB", (8, 8), (level, 255 - level, 90)).save(path)
    return path


def test_load_clip_returns_an_eval_model_and_its_processor(fake_clip_package):
    model, processor = clip_module.load_clip("some/clip")
    assert isinstance(model, FakeClip) and not model.training and isinstance(processor, FakeClipProcessor)


def test_clip_encoder_through_load_encoder(fake_clip_package, tmp_path):
    encoder = clip_module.load_encoder("some/clip")
    assert encoder.model_name == "some/clip"
    photos = [Image.new("RGB", (8, 8), (level, 0, 0)) for level in (30, 200)]
    vectors = encoder.images(photos)
    assert vectors.shape == (2, 4) and np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)
    texts = encoder.texts(["a portrait", "a still life"])
    assert texts.shape == (2, 4) and np.linalg.norm(texts, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_embed_images_reads_files_in_batches(fake_clip_package, tmp_path):
    paths = [_jpeg(tmp_path / f"{i}.jpg", 40 * i) for i in range(5)]
    model, processor = clip_module.load_clip()
    assert clip_module.embed_images(model, processor, paths, batch_size=2).shape == (5, 4)


# ----------------------------------------------------------------------------------- the device


@pytest.mark.parametrize(
    "cuda, mps, expected", [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")]
)
def test_device_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    assert data_module.device().type == expected


# ----------------------------------------------------------------------------------- the search index


def _table(n=6):
    return pd.DataFrame({"id": [f"id{i}" for i in range(n)], "path": [f"portrait/{i}.jpg" for i in range(n)],
                         "label": ["portrait"] * n, "style": ["Realism"] * n, "artist": ["a"] * n})


@pytest.fixture
def counting_clip(monkeypatch):
    calls = {"load": 0, "embed": 0}

    def fake_load_clip(name):
        calls["load"] += 1
        return object(), object()

    def fake_embed_images(model, processor, paths):
        calls["embed"] += 1
        return np.stack([np.array([1.0 + i, 2.0, 0.5, 0.1]) for i in range(len(paths))])

    monkeypatch.setattr(search_module, "load_clip", fake_load_clip)
    monkeypatch.setattr(search_module, "embed_images", fake_embed_images)
    return calls


def test_build_index_without_a_cache_embeds_every_time(counting_clip, tmp_path):
    index = search_module.build_index(_table(), tmp_path)
    assert len(index) == 6 and counting_clip == {"load": 1, "embed": 1}
    search_module.build_index(_table(), tmp_path)
    assert counting_clip["embed"] == 2


def test_build_index_with_a_cache_embeds_once(counting_clip, tmp_path):
    cache = tmp_path / "vectors.npz"
    search_module.build_index(_table(), tmp_path, cache)
    search_module.build_index(_table(), tmp_path, cache)
    assert counting_clip["embed"] == 1 and cache.exists()


def test_load_or_build_index_saves_then_reloads_unless_asked_to_rebuild(counting_clip, tmp_path):
    file = tmp_path / "idx" / "index.npz"
    built = load_or_build_index(_table(), tmp_path, file)
    assert file.exists() and counting_clip["embed"] == 1
    again = load_or_build_index(_table(), tmp_path, file)
    assert counting_clip["embed"] == 1 and np.allclose(again.vectors, built.vectors)
    load_or_build_index(_table(), tmp_path, file, rebuild=True)
    assert counting_clip["embed"] == 2
    assert load_index(file).meta["id"].tolist() == [f"id{i}" for i in range(6)]


# ----------------------------------------------------------------------------------- engines and optional packages


def test_quantised_engine_error_when_the_build_has_none(monkeypatch):
    monkeypatch.setattr(torch.backends, "quantized", types.SimpleNamespace(supported_engines=["none"], engine="none"))
    with pytest.raises(RuntimeError, match="no quantised engine"):
        select_quantized_engine()


def test_quantised_engine_is_activated_when_one_is_supported(monkeypatch):
    fake = types.SimpleNamespace(supported_engines=["none", "fbgemm"], engine="none")
    monkeypatch.setattr(torch.backends, "quantized", fake)
    assert select_quantized_engine() == "fbgemm" and fake.engine == "fbgemm"


def test_multilingual_encoder_says_how_to_install_the_extra(monkeypatch):
    from museum_guide.multilingual import load_multilingual

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes `import sentence_transformers` fail
    with pytest.raises(ImportError, match=r"\.\[multilingual\]"):
        load_multilingual()


def test_multilingual_encoder_is_built_from_sentence_transformers(monkeypatch):
    from museum_guide.multilingual import load_multilingual

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = lambda name: ("encoder", name)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    assert load_multilingual("some/multilingual-model") == ("encoder", "some/multilingual-model")


class _FakeVisionConfig:
    pass


class _FakeVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lin = nn.Linear(3, 2)

    def forward(self, pixel_values):
        return types.SimpleNamespace(image_embeds=self.lin(pixel_values.mean(dim=(2, 3))))


def _fake_clip_config_package(monkeypatch, vision_config):
    fake = types.ModuleType("transformers")
    fake.CLIPConfig = types.SimpleNamespace(from_pretrained=lambda name: types.SimpleNamespace(vision_config=vision_config))
    fake.CLIPVisionConfig = _FakeVisionConfig
    fake.CLIPVisionModelWithProjection = _FakeVisionModel
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_clip_image_tower_wraps_the_vision_model_with_random_weights(monkeypatch):
    from museum_guide.benchmark import clip_image_tower

    _fake_clip_config_package(monkeypatch, _FakeVisionConfig())
    tower = clip_image_tower("some/clip")
    assert not tower.training and tower(torch.rand(2, 3, 4, 4)).shape == (2, 2)


def test_clip_image_tower_refuses_a_checkpoint_without_a_vision_config(monkeypatch):
    from museum_guide.benchmark import clip_image_tower

    _fake_clip_config_package(monkeypatch, None)  # before: mypy-flagged, it would have failed later inside transformers
    with pytest.raises(TypeError, match="no CLIP vision configuration"):
        clip_image_tower("some/clip")
