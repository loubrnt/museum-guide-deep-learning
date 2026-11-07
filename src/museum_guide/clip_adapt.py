"""Adapting CLIP's image encoder: richer frozen features, partial fine-tuning and a small LoRA.

The project's CLIP model is a frozen image encoder with a linear head (`museum_guide.clip`). These helpers
test whether adapting the encoder beats that. Everything here is written for a vision transformer shaped like
Hugging Face's `CLIPVisionModel` (a list of blocks, a post layer norm, a projection) but only relies on a few
attributes, so it is tested on tiny fake modules without downloading anything.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable, Sequence

import numpy as np
import torch
from torch import nn

from museum_guide.train import run_epoch

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ----- what is trained -------------------------------------------------------------------------------------------


def last_blocks(blocks: Sequence[nn.Module], n: int) -> list[nn.Module]:
    """The last `n` blocks (none for n=0)."""
    if n < 0 or n > len(blocks):
        raise ValueError(f"cannot take the last {n} of {len(blocks)} blocks")
    return list(blocks)[len(blocks) - n :]


def freeze(model: nn.Module) -> None:
    """No parameter of `model` is trained any more."""
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze(modules: Iterable[nn.Module]) -> None:
    """Train every parameter of the given modules."""
    for m in modules:
        for p in m.parameters():
            p.requires_grad_(True)


def count_parameters(model_or_modules, trainable_only: bool = True) -> int:
    """Number of parameters (counted once even if a module is listed twice)."""
    modules = [model_or_modules] if isinstance(model_or_modules, nn.Module) else list(model_or_modules)
    seen, total = set(), 0
    for m in modules:
        for p in m.parameters():
            if id(p) in seen or (trainable_only and not p.requires_grad):
                continue
            seen.add(id(p))
            total += p.numel()
    return total


def parameter_groups(spec: Sequence[tuple[Iterable[nn.Module], float]], weight_decay: float) -> list[dict]:
    """AdamW parameter groups: for each (modules, learning rate), the trainable parameters of those modules.

    Biases and layer-norm weights (1-D tensors) get no weight decay. A parameter listed in two groups is kept in
    the first only; frozen parameters are left out.
    """
    seen: set[int] = set()
    groups = []
    for modules, lr in spec:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for m in modules:
            for p in m.parameters():
                if not p.requires_grad or id(p) in seen:
                    continue
                seen.add(id(p))
                (decay if p.ndim > 1 else no_decay).append(p)
        if decay:
            groups.append({"params": decay, "lr": lr, "weight_decay": weight_decay})
        if no_decay:
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0})
    return groups


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Copy of the trainable parameters only (what early stopping has to remember)."""
    return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_trainable_state(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            params[name].copy_(value)


# ----- richer frozen features ------------------------------------------------------------------------------------


class MultiLayerFeatures(nn.Module):
    """Class-token vectors of the last `n_layers` blocks, each through the final layer norm, concatenated.

    With n_layers=1 this is the vector CLIP uses before its projection. Earlier blocks keep more low-level
    information (colour, brushwork), which may matter for a style label.
    """

    def __init__(self, vision_model: nn.Module, post_layernorm: nn.Module, n_layers: int = 4):
        super().__init__()
        self.vision_model = vision_model
        self.post_layernorm = post_layernorm
        self.n_layers = n_layers

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        hidden = self.vision_model(pixel_values=pixel_values, output_hidden_states=True).hidden_states
        return torch.cat([self.post_layernorm(h[:, 0]) for h in hidden[-self.n_layers :]], dim=-1)


def fit_standardised_probe(train_X, train_y, val_X, val_y, Cs: Sequence[float] = (0.001, 0.01, 0.1, 1)):
    """Standardise the features, then logistic regression with C chosen on the validation set.

    Returns (fitted pipeline, chosen C). Ties go to the smaller C. High-dimensional concatenated features need
    stronger regularisation than the 512-d normalised vectors, hence the smaller grid.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    best = None
    for C in sorted(Cs):
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)).fit(train_X, train_y)
        score = clf.score(val_X, val_y)
        if best is None or score > best[0]:
            best = (score, clf, C)
    assert best is not None  # Cs is not empty
    return best[1], best[2]


# ----- a small LoRA -----------------------------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A x, with W and b frozen and only the low-rank A (r x in), B (out x r) trained.

    B starts at zero, so the wrapped layer initially behaves exactly like the original one.
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scale


def add_lora(blocks: Iterable[nn.Module], r: int = 8, alpha: float = 16.0, targets=("q_proj", "v_proj")) -> list[LoRALinear]:
    """Wrap the attention projections `targets` of every block (`block.self_attn.<name>`) with LoRA, in place.

    Call it AFTER `freeze(model)`: the new low-rank matrices are created trainable.
    """
    wrapped = []
    for block in blocks:
        for name in targets:
            layer = LoRALinear(getattr(block.self_attn, name), r, alpha)
            setattr(block.self_attn, name, layer)
            wrapped.append(layer)
    return wrapped


# ----- the classifier and its training ---------------------------------------------------------------------------


class ClipClassifier(nn.Module):
    """Vision tower -> projection -> L2 normalisation -> linear head: the model of the project's CLIP probe."""

    def __init__(self, vision_model: nn.Module, projection: nn.Module, head: nn.Linear):
        super().__init__()
        self.vision_model = vision_model
        self.projection = projection
        self.head = head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        emb = self.projection(self.vision_model(pixel_values=pixel_values).pooler_output)
        return self.head(nn.functional.normalize(emb, dim=-1))


def warm_start_head(head: nn.Linear, probe) -> None:
    """Start the head from a fitted sklearn logistic regression, so that fine-tuning begins at the probe's accuracy."""
    with torch.no_grad():
        head.weight.copy_(torch.as_tensor(probe.coef_, dtype=head.weight.dtype))
        head.bias.copy_(torch.as_tensor(probe.intercept_, dtype=head.bias.dtype))


def fit_adapt(
    model: nn.Module, loaders: dict, device, groups: list[dict], epochs: int = 6, patience: int = 2, on_epoch: Callable | None = None
) -> dict:
    """AdamW + cross-entropy on `loaders['train']`; the model of the best VALIDATION epoch is restored.

    Epoch 0 (the model as given, e.g. warm-started from the probe) is a candidate: if no epoch beats it on the
    validation set, `best_epoch` is 0 and the model is left unchanged. The test set is never read here.
    """
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(groups)
    _, val_acc, _ = run_epoch(model, loaders["val"], loss_fn, device)
    best_acc, best_epoch, best_state, waited = val_acc, 0, trainable_state(model), 0
    history = {"val_acc": [val_acc], "train_loss": [], "epoch_seconds": []}
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        train_loss, _, _ = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        _, val_acc, _ = run_epoch(model, loaders["val"], loss_fn, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["epoch_seconds"].append(time.perf_counter() - start)
        if on_epoch:
            on_epoch(epoch, train_loss, val_acc)
        if val_acc > best_acc:
            best_acc, best_epoch, best_state, waited = val_acc, epoch, trainable_state(model), 0
        else:
            waited += 1
            if waited >= patience:
                break
    load_trainable_state(model, best_state)
    return {**history, "best_epoch": best_epoch, "best_val_acc": best_acc, "epochs_run": len(history["train_loss"])}


def clip_transforms(augment: bool, size: int = 224):
    """Image -> normalised tensor as CLIP's own processor does (bicubic resize, centre crop, CLIP mean/std).

    With `augment`, the project's training augmentation: RandomResizedCrop (scale 0.7-1) and a horizontal flip.
    """
    from torchvision import transforms as T

    bicubic = T.InterpolationMode.BICUBIC
    steps = (
        [T.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=bicubic), T.RandomHorizontalFlip()]
        if augment
        else [T.Resize(size, interpolation=bicubic), T.CenterCrop(size)]
    )
    return T.Compose([*steps, T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])


def model_size_mb(n_parameters: int, bytes_per_parameter: int = 4) -> float:
    """Weight file size in MB (1 MB = 10^6 bytes) of `n_parameters` parameters."""
    return n_parameters * bytes_per_parameter / 1e6


def paired_differences(a: Sequence[float], b: Sequence[float]) -> dict[str, float]:
    """Mean and standard deviation of the per-seed differences a - b, and the number of seeds where a is higher."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return {
        "mean": float(d.mean()),
        "std": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
        "seeds_a_higher": int((d > 0).sum()),
        "seeds_tied": int((d == 0).sum()),
        "n_seeds": int(len(d)),
    }
