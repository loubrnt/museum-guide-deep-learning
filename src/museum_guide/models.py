"""Models: a feed-forward network written from PyTorch primitives, and a pretrained ResNet-18."""
from __future__ import annotations

import math

import torch
from torch import nn
from torchvision.models import ResNet, ResNet18_Weights, resnet18


class Dense(nn.Module):
    """y = x W^T + b, with He initialisation (suited to ReLU)."""

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_out, n_in) * math.sqrt(2.0 / n_in))
        self.bias = nn.Parameter(torch.zeros(n_out))

    def forward(self, x):
        return x @ self.weight.T + self.bias


class FeedForward(nn.Module):
    """Flatten the image, then Dense -> ReLU -> Dropout blocks, then a Dense layer to the class logits."""

    def __init__(self, n_in: int, hidden: tuple[int, ...], n_classes: int, dropout: float = 0.0):
        super().__init__()
        sizes = (n_in, *hidden)
        self.hidden = nn.ModuleList(Dense(a, b) for a, b in zip(sizes[:-1], sizes[1:]))
        self.out = Dense(sizes[-1], n_classes)
        self.dropout = dropout

    def forward(self, x):
        x = x.flatten(1)
        for layer in self.hidden:
            x = torch.relu(layer(x))
            if self.dropout:
                x = nn.functional.dropout(x, self.dropout, self.training)
        return self.out(x)


class FrozenBatchNormResNet(ResNet):
    """A ResNet whose frozen BatchNorm layers stay in eval mode, even after `.train()`.

    Freezing a layer with `requires_grad = False` does not freeze its BatchNorm *statistics*: in train mode
    they keep updating from the new data and each batch is normalised with its own statistics. A BatchNorm2d
    is treated as frozen when none of its parameters require a gradient, so the BatchNorm layers of a trained
    block (e.g. `layer4`) keep training normally.
    """

    def train(self, mode: bool = True):
        super().train(mode)
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        return self


def pretrained_resnet18(n_classes: int, train_layer4: bool, bn_eval: bool = False) -> nn.Module:
    """ImageNet ResNet-18 with a new head. Only the head (and optionally layer4) is trained.

    With `bn_eval=True` the BatchNorm layers of the frozen part also keep their ImageNet statistics
    (see `FrozenBatchNormResNet`). The default, False, is the behaviour of the reported results.
    """
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    for p in model.parameters():
        p.requires_grad = False
    if train_layer4:
        for p in model.layer4.parameters():
            p.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if bn_eval:
        model.__class__ = FrozenBatchNormResNet
        model.train(model.training)  # apply now: the model starts in train mode
    return model
