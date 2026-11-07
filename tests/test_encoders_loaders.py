"""`load_encoder` and `image_tower` with a fake `transformers` module: no network, no download, no real weights.

The loaders import transformers inside the function, so a fake module put in `sys.modules` replaces it.
"""
import sys
import types
from dataclasses import replace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from museum_guide import encoders
from museum_guide.encoders import REGISTRY, image_tower, load_encoder

DIM = 4


class _Out:
    """An output object, as transformers 5.x sometimes returns instead of a tensor."""

    def __init__(self, pooler_output=None, last_hidden_state=None):
        self.pooler_output, self.last_hidden_state = pooler_output, last_hidden_state


class FakeTextImageModel(nn.Module):
    """get_image_features / get_text_features like CLIP and SigLIP; records the arguments of the text call."""

    text_calls: list = []

    def get_image_features(self, pixel_values):
        mean = pixel_values.mean(dim=(2, 3))  # (batch, 3)
        return torch.cat([mean, torch.ones(mean.shape[0], 1)], dim=1)  # (batch, 4), a plain tensor; the constant keeps directions distinct

    def get_text_features(self, **kwargs):
        FakeTextImageModel.text_calls.append(sorted(kwargs))
        ids = kwargs["input_ids"].float()
        return _Out(pooler_output=torch.stack([ids.sum(1), ids.mean(1), ids.max(1).values, ids.min(1).values], dim=1) + 1.0)


class FakeProcessor:
    def __call__(self, images, return_tensors="pt"):
        means = torch.tensor([float(np.asarray(im).mean()) for im in images])
        return {"pixel_values": means.view(-1, 1, 1, 1).expand(-1, 3, 6, 6).clone()}


class FakeTokenizer:
    calls: list = []

    def __call__(self, sentences, return_tensors="pt", **kwargs):
        FakeTokenizer.calls.append(kwargs)
        ids = torch.tensor([[len(s), len(s.split()), 7] for s in sentences])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class FakeDino(nn.Module):
    def forward(self, pixel_values):
        batch = pixel_values.shape[0]
        tokens = pixel_values.mean(dim=(1, 2, 3)).view(batch, 1, 1).expand(batch, 5, 3)  # CLS + 4 patches, dim 3
        return _Out(last_hidden_state=tokens + torch.arange(5.0).view(1, 5, 1))


def _auto(cls):
    return types.SimpleNamespace(from_pretrained=lambda hf_id: cls())


@pytest.fixture
def fake_transformers(monkeypatch):
    FakeTextImageModel.text_calls, FakeTokenizer.calls = [], []
    fake = types.ModuleType("transformers")
    fake.AutoModel = types.SimpleNamespace(from_pretrained=lambda hf_id: FakeDino() if "dinov2" in hf_id else FakeTextImageModel())
    fake.AutoImageProcessor = _auto(FakeProcessor)
    fake.AutoTokenizer = _auto(FakeTokenizer)
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


@pytest.fixture
def images(tmp_path):
    paths = []
    for i, level in enumerate((20, 120, 220)):
        p = tmp_path / f"img{i}.jpg"
        Image.new("RGB", (16, 12), (level, level, level)).save(p)
        paths.append(p)
    return paths


def test_clip_encoder_embeds_images_and_texts(fake_transformers, images):
    enc = load_encoder("clip_b32")
    vectors = enc.embed_images(images)
    assert vectors.shape == (3, DIM)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)
    assert not np.allclose(vectors[0], vectors[2])  # different images, different vectors
    texts = enc.embed_texts(["a portrait", "a still life of fruit"])
    assert texts.shape == (2, DIM) and np.linalg.norm(texts, axis=1) == pytest.approx(1.0, abs=1e-5)
    # CLIP pads to the longest sentence and uses the attention mask
    assert FakeTokenizer.calls[-1] == {"padding": True}
    assert FakeTextImageModel.text_calls[-1] == ["attention_mask", "input_ids"]


def test_siglip_encoder_pads_to_a_fixed_length_and_ignores_the_mask(fake_transformers, images):
    enc = load_encoder("siglip_b16")
    assert enc.embed_texts(["a painting"]).shape == (1, DIM)
    assert FakeTokenizer.calls[-1] == {"padding": "max_length", "max_length": 64}
    assert FakeTextImageModel.text_calls[-1] == ["input_ids"]
    assert enc.embed_images(images).shape == (3, DIM)


def test_dinov2_encoder_has_no_text_tower_and_concatenates_cls_and_mean(fake_transformers, images):
    enc = load_encoder("dinov2_s")
    assert enc.embed_texts is None
    vectors = enc.embed_images(images)
    assert vectors.shape == (3, 6)  # CLS (dim 3) + mean of the patch tokens (dim 3)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_resnet_encoder_uses_pooled_features(monkeypatch, images):
    import torchvision.models as tvm

    monkeypatch.setattr(tvm, "resnet18", lambda weights=None: _real_resnet18())  # the real one would download weights
    enc = load_encoder("resnet18_frozen")
    assert enc.embed_texts is None
    vectors = enc.embed_images(images)
    assert vectors.shape == (3, 512)
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(1.0, abs=1e-5)


def _real_resnet18():
    from torchvision.models.resnet import ResNet, BasicBlock

    return ResNet(BasicBlock, [2, 2, 2, 2])  # the architecture of resnet18, random weights, no download


def test_load_encoder_rejects_an_unknown_family(monkeypatch):
    monkeypatch.setitem(REGISTRY, "odd", replace(REGISTRY["clip_b32"], name="odd", family="alien"))
    with pytest.raises(ValueError, match="unknown encoder family"):
        load_encoder("odd")


def test_tensor_helper_accepts_tensors_and_output_objects():
    t = torch.ones(2, 3)
    assert encoders._tensor(t) is t
    assert encoders._tensor(_Out(pooler_output=t)) is t


# ------------------------------------------------------------------ image_tower with fake configurations


class FakeVisionConfig:
    pass


class FakeSiglipVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.lin = nn.Linear(3, 2)

    def forward(self, pixel_values):
        return _Out(pooler_output=self.lin(pixel_values.mean(dim=(2, 3))))


class FakeDinoBackbone(FakeDino):
    def __init__(self, config):
        super().__init__()
        self.config = config


@pytest.fixture
def fake_configs(monkeypatch):
    fake = types.ModuleType("transformers")
    fake.SiglipVisionConfig = FakeVisionConfig
    fake.SiglipConfig = types.SimpleNamespace(from_pretrained=lambda hf_id: types.SimpleNamespace(vision_config=FakeVisionConfig()))
    fake.SiglipVisionModel = FakeSiglipVision
    fake.AutoConfig = types.SimpleNamespace(from_pretrained=lambda hf_id: object())
    fake.Dinov2Model = FakeDinoBackbone
    monkeypatch.setitem(sys.modules, "transformers", fake)
    return fake


def test_siglip_tower_wraps_the_vision_model(fake_configs):
    tower, size = image_tower("siglip_b16")
    assert size == 224 and not tower.training
    assert tower(torch.rand(2, 3, 8, 8)).shape == (2, 2)


def test_siglip_tower_refuses_a_config_without_a_vision_part(fake_configs):
    fake_configs.SiglipConfig = types.SimpleNamespace(from_pretrained=lambda hf_id: types.SimpleNamespace(vision_config=None))
    with pytest.raises(TypeError, match="no SigLIP vision configuration"):
        image_tower("siglip_b16")


def test_dinov2_tower_pools_cls_and_mean(fake_configs):
    tower, size = image_tower("dinov2_b")
    assert size == 224 and tower(torch.rand(3, 3, 8, 8)).shape == (3, 6)


def test_clip_tower_comes_from_the_benchmark_helper(monkeypatch):
    monkeypatch.setattr("museum_guide.benchmark.clip_image_tower", lambda hf_id: ("tower", hf_id))
    assert image_tower("clip_b16") == (("tower", REGISTRY["clip_b16"].hf_id), 224)
