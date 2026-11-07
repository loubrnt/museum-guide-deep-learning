from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from museum_guide.clip_adapt import (
    ClipClassifier,
    LoRALinear,
    MultiLayerFeatures,
    add_lora,
    count_parameters,
    fit_adapt,
    fit_standardised_probe,
    freeze,
    last_blocks,
    load_trainable_state,
    model_size_mb,
    paired_differences,
    parameter_groups,
    trainable_state,
    unfreeze,
    warm_start_head,
)


class Block(nn.Module):
    def __init__(self, d=8):
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(d, d)
        self.self_attn.v_proj = nn.Linear(d, d)
        self.self_attn.k_proj = nn.Linear(d, d)
        self.mlp = nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, x):
        a = self.self_attn
        return self.norm(x + self.mlp(a.q_proj(x) + a.v_proj(x)))


class FakeVision(nn.Module):
    """A tiny stand-in for CLIPVisionModel: a patch embedding, a stack of blocks, hidden states, a pooled vector."""

    def __init__(self, d=8, n_blocks=4):
        super().__init__()
        self.embed = nn.Linear(4, d)
        self.blocks = nn.ModuleList(Block(d) for _ in range(n_blocks))
        self.post_layernorm = nn.LayerNorm(d)

    def forward(self, pixel_values, output_hidden_states=False):
        x = self.embed(pixel_values.flatten(2))  # (batch, 3 tokens of 4 values, d)
        hidden = [x]
        for block in self.blocks:
            x = block(x)
            hidden.append(x)
        return SimpleNamespace(hidden_states=tuple(hidden), pooler_output=self.post_layernorm(x[:, 0]))


def fake_images(n=6):
    return torch.randn(n, 3, 2, 2)  # flattened to 3 tokens of 4 values


def test_last_blocks():
    blocks = nn.ModuleList(Block() for _ in range(4))
    assert last_blocks(blocks, 2) == [blocks[2], blocks[3]]
    assert last_blocks(blocks, 0) == []
    with pytest.raises(ValueError):
        last_blocks(blocks, 5)


def test_freeze_unfreeze_and_count():
    vision = FakeVision()
    total = count_parameters(vision, trainable_only=False)
    assert count_parameters(vision) == total
    freeze(vision)
    assert count_parameters(vision) == 0
    unfreeze(last_blocks(vision.blocks, 1))
    assert count_parameters(vision) == count_parameters(vision.blocks[3], trainable_only=False)
    assert not vision.blocks[0].mlp.weight.requires_grad and vision.blocks[3].mlp.weight.requires_grad
    # a module listed twice is counted once
    assert count_parameters([vision.blocks[3], vision.blocks[3]]) == count_parameters(vision.blocks[3])


def test_parameter_groups_split_decay_and_skip_frozen():
    vision = FakeVision()
    head = nn.Linear(8, 3)
    freeze(vision)
    unfreeze(last_blocks(vision.blocks, 1))
    groups = parameter_groups([(last_blocks(vision.blocks, 1), 1e-5), ([head], 1e-3)], weight_decay=0.01)
    by_lr = {}
    for g in groups:
        by_lr.setdefault(g["lr"], []).append(g)
    assert set(by_lr) == {1e-5, 1e-3}
    for g in groups:
        assert all(p.requires_grad for p in g["params"])
        if g["weight_decay"] == 0.0:
            assert all(p.ndim == 1 for p in g["params"])  # biases and layer norms
        else:
            assert all(p.ndim > 1 for p in g["params"])
    n_in_groups = sum(p.numel() for g in groups for p in g["params"])
    assert n_in_groups == count_parameters(vision) + count_parameters(head)


def test_parameter_groups_do_not_repeat_a_parameter():
    layer = nn.Linear(4, 2)
    groups = parameter_groups([([layer], 1e-3), ([layer], 1e-5)], weight_decay=0.1)
    params = [p for g in groups for p in g["params"]]
    assert len(params) == len({id(p) for p in params}) == 2


def test_trainable_state_round_trip():
    layer = nn.Linear(3, 2)
    layer.weight.requires_grad_(False)
    state = trainable_state(layer)
    assert set(state) == {"bias"}
    with torch.no_grad():
        layer.bias.add_(5.0)
    load_trainable_state(layer, state)
    assert torch.equal(layer.bias, state["bias"])


def test_multilayer_features_shape_and_last_layer_match():
    vision = FakeVision(d=8, n_blocks=4).eval()
    x = fake_images()
    feats = MultiLayerFeatures(vision, vision.post_layernorm, n_layers=3)(x)
    assert feats.shape == (6, 3 * 8)
    last = MultiLayerFeatures(vision, vision.post_layernorm, n_layers=1)(x)
    assert torch.allclose(last, vision(x).pooler_output, atol=1e-6)  # one layer = CLIP's pooled vector
    assert torch.allclose(feats[:, -8:], last, atol=1e-6)


def test_fit_standardised_probe_picks_c_on_validation():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 300)
    X = rng.normal(size=(300, 20)) + 2.0 * np.eye(3)[y] @ rng.normal(size=(3, 20))
    clf, C = fit_standardised_probe(X[:150], y[:150], X[150:225], y[150:225], Cs=(0.0001, 0.1, 1))
    assert C in (0.0001, 0.1, 1)
    assert clf.score(X[225:], y[225:]) > 0.8


def test_lora_starts_identical_and_trains_only_the_low_rank_matrices():
    torch.manual_seed(0)
    block = Block()
    x = torch.randn(5, 8)
    before = block(x)
    freeze(block)
    wrapped = add_lora([block], r=4)
    assert len(wrapped) == 2 and all(isinstance(w, LoRALinear) for w in wrapped)
    assert torch.allclose(block(x), before, atol=1e-6)  # B = 0: no change at the start
    trainable = {n for n, p in block.named_parameters() if p.requires_grad}
    assert trainable == {"self_attn.q_proj.A", "self_attn.q_proj.B", "self_attn.v_proj.A", "self_attn.v_proj.B"}
    assert not block.self_attn.k_proj.weight.requires_grad  # not targeted
    block(x).sum().backward()
    assert block.self_attn.q_proj.B.grad.abs().sum() > 0  # the LoRA matrices receive gradient
    assert block.self_attn.q_proj.base.weight.grad is None  # the base weights do not


def test_lora_parameter_count():
    block = Block(d=8)
    freeze(block)
    add_lora([block], r=2)
    assert count_parameters(block) == 2 * (2 * 8 + 8 * 2)  # q and v: A (2x8) + B (8x2)


def test_warm_start_head_reproduces_the_probe():
    rng = np.random.default_rng(1)
    X, y = rng.normal(size=(120, 6)), rng.integers(0, 3, 120)
    probe = LogisticRegression(max_iter=500).fit(X, y)
    head = nn.Linear(6, 3)
    warm_start_head(head, probe)
    logits = head(torch.as_tensor(X, dtype=torch.float32)).detach().numpy()
    assert np.allclose(logits, probe.decision_function(X), atol=1e-4)


def test_clip_classifier_forward_is_a_normalised_embedding_through_the_head():
    vision = FakeVision(d=8)
    model = ClipClassifier(vision, nn.Linear(8, 5, bias=False), nn.Linear(5, 3)).eval()
    out = model(fake_images(4))
    assert out.shape == (4, 3)


def _loaders(n=60, seed=0):
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = torch.randn(n, 3, 2, 2, generator=g) + 2.0 * y[:, None, None, None]
    ds = TensorDataset(x, y)
    return {"train": DataLoader(ds, batch_size=16, shuffle=True), "val": DataLoader(ds, batch_size=16)}


def _classifier():
    torch.manual_seed(0)
    vision = FakeVision(d=8)
    freeze(vision)
    unfreeze(last_blocks(vision.blocks, 1))
    proj, head = nn.Linear(8, 5, bias=False), nn.Linear(5, 2)
    return ClipClassifier(vision, proj, head), vision


def test_fit_adapt_learns_and_restores_the_best_validation_epoch():
    model, vision = _classifier()
    groups = parameter_groups([(last_blocks(vision.blocks, 1), 1e-2), ([model.projection, model.head], 1e-2)], 0.0)
    hist = fit_adapt(model, _loaders(), torch.device("cpu"), groups, epochs=8, patience=8)
    assert len(hist["val_acc"]) == hist["epochs_run"] + 1  # epoch 0 plus the epochs run
    assert hist["best_val_acc"] == max(hist["val_acc"])
    assert hist["best_val_acc"] > 0.9
    # the restored model is the best one: it scores that validation accuracy again
    from museum_guide.train import run_epoch

    _, acc, _ = run_epoch(model, _loaders()["val"], nn.CrossEntropyLoss(), torch.device("cpu"))
    assert acc == pytest.approx(hist["best_val_acc"])


def test_fit_adapt_keeps_the_start_when_nothing_helps_and_stops_early():
    model, vision = _classifier()
    before = {n: p.clone() for n, p in model.named_parameters()}
    groups = parameter_groups([([model.head], 0.0)], 0.0)  # lr 0: nothing can improve
    hist = fit_adapt(model, _loaders(), torch.device("cpu"), groups, epochs=20, patience=2)
    assert hist["best_epoch"] == 0
    assert hist["epochs_run"] == 2  # patience 2 without any gain
    assert all(torch.equal(before[n], p) for n, p in model.named_parameters())


def test_fit_adapt_does_not_touch_frozen_parameters():
    model, vision = _classifier()
    frozen = vision.blocks[0].mlp.weight.clone()
    groups = parameter_groups([(last_blocks(vision.blocks, 1), 1e-2), ([model.head], 1e-2)], 0.0)
    fit_adapt(model, _loaders(), torch.device("cpu"), groups, epochs=3)
    assert torch.equal(vision.blocks[0].mlp.weight, frozen)


def test_model_size_and_paired_differences():
    assert model_size_mb(7_087_872) == pytest.approx(28.35, abs=0.01)
    d = paired_differences([0.6, 0.5, 0.7], [0.5, 0.5, 0.6])
    assert d["mean"] == pytest.approx(0.2 / 3)
    assert d["seeds_a_higher"] == 2 and d["seeds_tied"] == 1 and d["n_seeds"] == 3


def test_clip_transforms_shapes_and_normalisation():
    from PIL import Image

    from museum_guide.clip_adapt import CLIP_MEAN, CLIP_STD, clip_transforms

    img = Image.new("RGB", (128, 128), (0, 0, 0))
    out = clip_transforms(augment=False)(img)
    assert out.shape == (3, 224, 224)
    expected = torch.tensor([-m / s for m, s in zip(CLIP_MEAN, CLIP_STD)])
    assert torch.allclose(out[:, 0, 0], expected, atol=1e-4)
    assert clip_transforms(augment=True)(img).shape == (3, 224, 224)


def test_fit_adapt_restores_the_best_epoch_even_when_the_last_one_is_worse():
    # Random validation labels: validation accuracy is noise, so the best epoch is not the last one and the
    # test fails if `fit_adapt` leaves the final weights in place instead of the best ones.
    model, vision = _classifier()
    base = _loaders()
    xv = base["val"].dataset.tensors[0]
    noisy_y = torch.randint(0, 2, (len(xv),), generator=torch.Generator().manual_seed(7))
    val = DataLoader(TensorDataset(xv, noisy_y), batch_size=16)
    groups = parameter_groups([(last_blocks(vision.blocks, 1), 1e-2), ([model.projection, model.head], 1e-2)], 0.0)
    hist = fit_adapt(model, {"train": base["train"], "val": val}, torch.device("cpu"), groups, epochs=10, patience=10)
    assert hist["best_val_acc"] > hist["val_acc"][-1]  # precondition: the last epoch is not the best
    from museum_guide.train import run_epoch

    _, acc, _ = run_epoch(model, val, nn.CrossEntropyLoss(), torch.device("cpu"))
    assert acc == pytest.approx(hist["best_val_acc"])


def test_parameter_groups_leave_frozen_modules_out_even_when_listed():
    vision = FakeVision()
    freeze(vision)
    unfreeze(last_blocks(vision.blocks, 1))
    groups = parameter_groups([([vision.blocks[0], vision.blocks[3]], 1e-5)], weight_decay=0.01)
    listed = {id(p) for g in groups for p in g["params"]}
    assert not listed & {id(p) for p in vision.blocks[0].parameters()}  # a frozen block must not reach the optimiser
    assert listed == {id(p) for p in vision.blocks[3].parameters()}
