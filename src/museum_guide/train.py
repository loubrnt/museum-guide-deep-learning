"""Training loop with monitoring, evaluation and CPU latency measurement."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn

REPORTS = Path("reports")


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    grad_norm: list[float] = field(default_factory=list)  # mean gradient norm over the epoch's steps


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """One pass over `loader`; trains if an optimizer is given. Returns (loss, acc, mean grad norm)."""
    model.train(optimizer is not None)
    total_loss, correct, n, norms = 0.0, 0, 0, []
    with torch.set_grad_enabled(optimizer is not None):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                norms.append(torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])).item())
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / n, correct / n, float(np.mean(norms)) if norms else 0.0


def fit(model, loaders, device, epochs, lr, weight_decay=0.0, patience=None, verbose=True) -> History:
    """Adam + cross-entropy. Keeps the weights of the best validation epoch (early stopping)."""
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    hist, best, best_state, waited = History(), -1.0, None, 0
    for epoch in range(epochs):
        tl, ta, gn = run_epoch(model, loaders["train"], loss_fn, device, optimizer)
        vl, va, _ = run_epoch(model, loaders["val"], loss_fn, device)
        for k, v in zip(("train_loss", "train_acc", "val_loss", "val_acc", "grad_norm"), (tl, ta, vl, va, gn)):
            getattr(hist, k).append(v)
        if verbose:
            print(f"epoch {epoch + 1:3d}  train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}  |grad| {gn:.2f}")
        if va > best:
            best, best_state, waited = va, {k: v.detach().clone() for k, v in model.state_dict().items()}, 0
        else:
            waited += 1
            if patience is not None and waited >= patience:
                break
    model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict(model, loader, device):
    model.eval().to(device)
    ys, preds = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).argmax(1).cpu())
        ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(preds).numpy()


@torch.no_grad()
def cpu_latency_ms(model, input_shape, repeats=50) -> float:
    """Median time to classify one image on CPU, single thread (a modest kiosk)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model = model.cpu().eval()
    x = torch.randn(1, *input_shape)
    for _ in range(5):
        model(x)
    times = []
    for _ in range(repeats):
        t = time.perf_counter()
        model(x)
        times.append((time.perf_counter() - t) * 1000)
    torch.set_num_threads(threads)
    return float(np.median(times))


def plot_history(hist: History, title: str, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    ep = range(1, len(hist.train_loss) + 1)
    axes[0].plot(ep, hist.train_loss, label="train")
    axes[0].plot(ep, hist.val_loss, label="val")
    axes[0].set_title("loss")
    axes[1].plot(ep, hist.train_acc, label="train")
    axes[1].plot(ep, hist.val_acc, label="val")
    axes[1].set_title("accuracy")
    axes[2].plot(ep, hist.grad_norm)
    axes[2].set_title("mean gradient norm")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion(y, pred, classes, title: str, path: Path):
    cm = confusion_matrix(y, pred, labels=range(len(classes)), normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(ax=ax, values_format=".2f", colorbar=False, xticks_rotation=35)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_results(name: str, results: dict, hist: History | None = None):
    REPORTS.mkdir(exist_ok=True)
    if hist is not None:
        results = {**results, "history": asdict(hist)}
    (REPORTS / f"{name}.json").write_text(json.dumps(results, indent=2))


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())
