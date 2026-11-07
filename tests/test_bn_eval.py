"""`bn_eval=True` keeps the BatchNorm layers of the frozen part in eval mode; the default is unchanged."""
import copy

import pytest
import torch
from torch import nn
from torchvision.models import ResNet, resnet18

import museum_guide.models as models
from museum_guide.models import FrozenBatchNormResNet, pretrained_resnet18


@pytest.fixture(autouse=True)
def offline_resnet(monkeypatch):
    """Same architecture as the ImageNet model, random weights: avoids downloading 45 MB in tests."""
    monkeypatch.setattr(models, "resnet18", lambda weights=None: resnet18(weights=None))


def _frozen_bn(model):
    return [m for m in model.modules() if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters())]


def _stats(bn):
    return bn.running_mean.clone(), bn.running_var.clone()


def _train_step(model):
    model.train()
    model(torch.randn(4, 3, 64, 64) + 1.0)


@pytest.mark.parametrize("train_layer4", [False, True])
def test_frozen_batchnorm_statistics_do_not_move_with_bn_eval(train_layer4):
    model = pretrained_resnet18(5, train_layer4=train_layer4, bn_eval=True)
    frozen = _frozen_bn(model)
    before = [_stats(m) for m in frozen]
    _train_step(model)
    assert frozen
    for m, (mean, var) in zip(frozen, before):
        assert torch.equal(m.running_mean, mean) and torch.equal(m.running_var, var)
        assert not m.training


@pytest.mark.parametrize("train_layer4", [False, True])
def test_frozen_batchnorm_statistics_move_by_default(train_layer4):
    model = pretrained_resnet18(5, train_layer4=train_layer4)  # default behaviour, as in the reported results
    frozen = _frozen_bn(model)
    before = [_stats(m) for m in frozen]
    _train_step(model)
    assert all(m.training for m in frozen)
    assert any(not torch.equal(m.running_mean, mean) for m, (mean, _) in zip(frozen, before))


def test_head_only_variant_freezes_every_batchnorm():
    model = pretrained_resnet18(5, train_layer4=False, bn_eval=True)
    model.train()
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert len(bns) == 20  # ResNet-18
    assert not any(m.training for m in bns)


def test_layer4_batchnorm_keeps_training_when_layer4_is_trained():
    model = pretrained_resnet18(5, train_layer4=True, bn_eval=True)
    model.train()
    layer4_bn = [m for m in model.layer4.modules() if isinstance(m, nn.BatchNorm2d)]
    other_bn = [m for n, m in model.named_modules() if isinstance(m, nn.BatchNorm2d) and not n.startswith("layer4")]
    assert layer4_bn and all(m.training for m in layer4_bn)
    assert other_bn and not any(m.training for m in other_bn)
    before = _stats(layer4_bn[0])
    _train_step(model)
    assert not torch.equal(layer4_bn[0].running_mean, before[0])  # layer4 statistics still adapt


def test_eval_then_train_keeps_frozen_batchnorm_in_eval():
    model = pretrained_resnet18(5, train_layer4=True, bn_eval=True)
    model.eval()
    assert not any(m.training for m in model.modules())
    model.train()
    assert model.training and model.fc.training
    assert not any(m.training for m in _frozen_bn(model))


def test_the_model_is_trainable_and_gradients_reach_the_unfrozen_parts():
    model = pretrained_resnet18(5, train_layer4=True, bn_eval=True)
    model.train()
    nn.functional.cross_entropy(model(torch.randn(4, 3, 64, 64)), torch.tensor([0, 1, 2, 3])).backward()
    assert model.fc.weight.grad is not None
    assert model.layer4[1].bn2.weight.grad is not None
    assert model.conv1.weight.grad is None  # still frozen


def test_default_is_a_plain_resnet_in_train_mode():
    model = pretrained_resnet18(5, train_layer4=False)
    assert type(model) is ResNet
    model.train()
    assert all(m.training for m in model.modules())


def test_bn_eval_changes_the_class_not_the_weights_or_the_forward_pass_in_eval_mode():
    torch.manual_seed(0)
    a = pretrained_resnet18(5, train_layer4=True)
    torch.manual_seed(0)
    b = pretrained_resnet18(5, train_layer4=True, bn_eval=True)
    assert isinstance(b, FrozenBatchNormResNet)
    assert a.state_dict().keys() == b.state_dict().keys()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 64, 64)
    assert torch.allclose(a.eval()(x), b.eval()(x))


def test_copies_keep_the_behaviour():
    model = pretrained_resnet18(5, train_layer4=False, bn_eval=True)
    clone = copy.deepcopy(model)
    clone.train()
    assert type(clone) is FrozenBatchNormResNet
    assert not any(m.training for m in clone.modules() if isinstance(m, nn.BatchNorm2d))
    assert clone.fc.training
