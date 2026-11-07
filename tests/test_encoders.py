import numpy as np
import pytest
import torch

from museum_guide import encoders
from museum_guide.encoders import (
    REGISTRY,
    EncoderSpec,
    embed_paths,
    feature_size_mb,
    image_tower,
    l2_normalise,
    paired_wins,
    pool_cls_and_mean,
)


def test_registry_is_consistent():
    assert "clip_b32" in REGISTRY  # the baseline every other encoder is compared with
    assert len({s.hf_id for s in REGISTRY.values()}) == len(REGISTRY)
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.family in {"clip", "siglip", "dinov2", "resnet"}
        assert spec.input_size in {128, 224}
        assert spec.convention
        assert spec.has_text == (spec.family in {"clip", "siglip"})  # only the contrastive models can do zero-shot


def test_l2_normalise_rows_and_zero_rows():
    x = np.array([[3.0, 4.0], [0.0, 0.0]])
    out = l2_normalise(x)
    assert out[0] == pytest.approx([0.6, 0.8])
    assert not np.isnan(out).any()
    assert (out[1] == 0).all()
    assert out.dtype == np.float32


def test_pool_cls_and_mean_numpy_and_torch_agree():
    tokens = np.arange(2 * 5 * 3, dtype=np.float32).reshape(2, 5, 3)  # batch 2, CLS + 4 patches, dim 3
    out = pool_cls_and_mean(tokens)
    assert out.shape == (2, 6)
    assert out[0, :3] == pytest.approx(tokens[0, 0])  # first half: the CLS token
    assert out[0, 3:] == pytest.approx(tokens[0, 1:].mean(0))  # second half: mean of the patch tokens
    assert torch.allclose(pool_cls_and_mean(torch.from_numpy(tokens)), torch.from_numpy(out))


def test_pool_cls_and_mean_ignores_nothing_but_the_cls_position():
    tokens = np.zeros((1, 4, 2), dtype=np.float32)
    tokens[0, 0] = [1, 2]
    tokens[0, 1:] = [[3, 3], [6, 6], [9, 9]]
    assert pool_cls_and_mean(tokens)[0] == pytest.approx([1, 2, 6, 6])


def test_paired_wins_counts_wins_ties_losses():
    assert paired_wins([0.9, 0.8, 0.7], [0.8, 0.8, 0.9]) == {"wins": 1, "ties": 1, "losses": 1}


def test_feature_size_mb():
    assert feature_size_mb(1_000_000) == 4.0
    assert feature_size_mb(1_000_000, 2) == 2.0


def _images(tmp_path, n=5):
    from PIL import Image

    paths = []
    for i in range(n):
        p = tmp_path / f"im{i}.jpg"
        Image.new("RGB", (20, 12), (i * 40, 10, 10)).save(p)
        paths.append(p)
    return paths


def test_embed_paths_batches_keep_order_and_normalise(tmp_path):
    paths = _images(tmp_path)
    seen_batches = []

    def preprocess(images):  # one number per image: its red level
        seen_batches.append(len(images))
        return torch.tensor([[float(np.asarray(im)[0, 0, 0]) + 1.0, 1.0] for im in images])

    out = embed_paths(preprocess, lambda x: x, paths, batch_size=2)
    assert seen_batches == [2, 2, 1]
    assert out.shape == (5, 2)
    assert np.linalg.norm(out, axis=1) == pytest.approx(np.ones(5), abs=1e-5)
    assert out[0, 0] < out[4, 0]  # the order of the paths is kept (red level increases with i)


def test_embed_paths_does_not_track_gradients(tmp_path):
    paths = _images(tmp_path, 2)
    layer = torch.nn.Linear(2, 3)
    out = embed_paths(lambda ims: torch.ones(len(ims), 2), layer, paths)
    assert out.shape == (2, 3)
    assert layer.weight.grad is None


def test_resnet_image_tower_has_random_weights_and_the_right_output():
    tower, size = image_tower("resnet18_frozen")  # torchvision, weights=None: no download
    assert size == 128
    with torch.no_grad():
        out = tower(torch.randn(2, 3, size, size))
    assert out.shape == (2, 512)  # the pooled features, the classifier is removed
    assert not tower.training


def test_image_tower_rejects_an_unknown_family(monkeypatch):
    monkeypatch.setitem(REGISTRY, "weird", EncoderSpec("weird", "transformer-xl", "x/y", 224, False, "none"))
    with pytest.raises(ValueError, match="unknown encoder family"):
        image_tower("weird")
    with pytest.raises(ValueError, match="unknown encoder family"):
        encoders.load_encoder("weird")
