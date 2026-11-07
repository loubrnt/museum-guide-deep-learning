"""Plots and logging of the training module, written to a temporary folder (matplotlib's Agg backend)."""
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from museum_guide.train import History, fit, plot_confusion, plot_history

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _history(n=4):
    return History(
        train_loss=list(np.linspace(1.0, 0.2, n)), train_acc=list(np.linspace(0.4, 0.95, n)),
        val_loss=list(np.linspace(1.1, 0.7, n)), val_acc=list(np.linspace(0.35, 0.7, n)), grad_norm=list(np.linspace(3, 1, n)),
    )


def test_plot_history_writes_a_png_and_creates_the_folder(tmp_path):
    path = tmp_path / "figures" / "nested" / "curves.png"
    plot_history(_history(), "MLP, demo", path)
    assert path.read_bytes().startswith(PNG_MAGIC) and path.stat().st_size > 5_000


def test_plot_history_handles_a_single_epoch(tmp_path):
    plot_history(_history(1), "one epoch", tmp_path / "one.png")
    assert (tmp_path / "one.png").exists()


def test_plot_confusion_writes_a_png(tmp_path):
    y = np.array([0, 0, 1, 1, 2, 2])
    pred = np.array([0, 1, 1, 1, 2, 0])
    path = tmp_path / "cm.png"
    plot_confusion(y, pred, ["cityscape", "landscape", "portrait"], "demo - test", path)
    assert path.read_bytes().startswith(PNG_MAGIC)


def test_plot_confusion_keeps_a_class_that_never_appears(tmp_path):
    plot_confusion(np.array([0, 1, 1]), np.array([0, 1, 0]), ["a", "b", "never"], "three classes, two seen", tmp_path / "cm.png")
    assert (tmp_path / "cm.png").exists()


def test_fit_prints_one_line_per_epoch_when_verbose(capsys):
    torch.manual_seed(0)
    x = torch.randn(32, 4)
    y = (x[:, 0] > 0).long()
    loader = DataLoader(TensorDataset(x, y), batch_size=16)
    fit(nn.Linear(4, 2), {"train": loader, "val": loader}, torch.device("cpu"), epochs=3, lr=1e-2, verbose=True)
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("epoch")]
    assert len(lines) == 3 and "train" in lines[0] and "|grad|" in lines[0]


def test_fit_with_no_epochs_is_a_clear_error():
    # before: "TypeError: Expected state_dict to be dict-like, got NoneType", raised from load_state_dict
    x = torch.randn(8, 4)
    loader = DataLoader(TensorDataset(x, torch.zeros(8, dtype=torch.long)), batch_size=4)
    with pytest.raises(ValueError, match="epochs must be at least 1"):
        fit(nn.Linear(4, 2), {"train": loader, "val": loader}, torch.device("cpu"), epochs=0, lr=1e-2, verbose=False)
