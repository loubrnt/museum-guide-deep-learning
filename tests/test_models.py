import pytest
import torch
from torchvision.models import resnet18

import museum_guide.models as models
from museum_guide.models import Dense, FeedForward, pretrained_resnet18


def test_dense_shape_and_parameters():
    layer = Dense(10, 4)
    assert layer(torch.randn(3, 10)).shape == (3, 4)
    assert layer.weight.shape == (4, 10)
    assert torch.count_nonzero(layer.bias) == 0


def test_dense_matches_torch_linear():
    layer, ref = Dense(7, 3), torch.nn.Linear(7, 3)
    with torch.no_grad():
        ref.weight.copy_(layer.weight)
        ref.bias.copy_(layer.bias)
    x = torch.randn(5, 7)
    assert torch.allclose(layer(x), ref(x), atol=1e-6)


def test_dense_uses_he_initialisation():
    torch.manual_seed(0)
    layer = Dense(1000, 1000)
    assert layer.weight.std().item() == pytest.approx((2 / 1000) ** 0.5, rel=0.05)


def test_feedforward_flattens_images_and_returns_logits():
    net = FeedForward(3 * 8 * 8, (16, 8), n_classes=5)
    assert net(torch.randn(4, 3, 8, 8)).shape == (4, 5)


def test_feedforward_dropout_only_acts_in_training_mode():
    torch.manual_seed(0)
    net = FeedForward(12, (64,), n_classes=5, dropout=0.5)
    x = torch.randn(6, 12)
    net.eval()
    assert torch.equal(net(x), net(x))  # deterministic at inference
    net.train()
    assert not torch.equal(net(x), net(x))  # stochastic while training


def test_feedforward_can_take_a_gradient_step():
    net = FeedForward(12, (8,), n_classes=3)
    loss = torch.nn.functional.cross_entropy(net(torch.randn(4, 12)), torch.tensor([0, 1, 2, 0]))
    loss.backward()
    assert all(p.grad is not None for p in net.parameters())


@pytest.fixture
def offline_resnet(monkeypatch):
    """Same architecture as the ImageNet model, random weights: avoids downloading 45 MB in tests."""
    monkeypatch.setattr(models, "resnet18", lambda weights=None: resnet18(weights=None))


def test_pretrained_resnet_head_only_freezes_the_backbone(offline_resnet):
    model = pretrained_resnet18(5, train_layer4=False)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 512 * 5 + 5


def test_pretrained_resnet_layer4_is_also_trainable(offline_resnet):
    model = pretrained_resnet18(5, train_layer4=True)
    trainable = {n.split(".")[0] for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"layer4", "fc"}


def test_pretrained_resnet_outputs_one_logit_per_class(offline_resnet):
    model = pretrained_resnet18(5, train_layer4=False).eval()
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5)
