import json

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import museum_guide.train as train
from museum_guide.train import History, cpu_latency_ms, fit, n_params, predict, run_epoch, save_results

CPU = torch.device("cpu")


def _blobs(n=60, seed=0):
    """Two well-separated Gaussian blobs: any working training loop reaches ~100%."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % 2
    x = torch.randn(n, 4, generator=g) + 4 * y[:, None]
    return x, y


@pytest.fixture
def loaders():
    xt, yt = _blobs(60, 0)
    xv, yv = _blobs(30, 1)
    return {
        "train": DataLoader(TensorDataset(xt, yt), batch_size=16, shuffle=True),
        "val": DataLoader(TensorDataset(xv, yv), batch_size=16),
    }


def _model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


def test_run_epoch_evaluation_does_not_touch_the_weights(loaders):
    model = _model()
    before = [p.clone() for p in model.parameters()]
    loss, acc, grad_norm = run_epoch(model, loaders["val"], nn.CrossEntropyLoss(), CPU)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert 0 <= acc <= 1 and loss > 0
    assert grad_norm == 0.0  # no optimiser: no gradient norm recorded


def test_run_epoch_training_updates_weights_and_reports_grad_norm(loaders):
    model = _model()
    before = [p.clone() for p in model.parameters()]
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    _, _, grad_norm = run_epoch(model, loaders["train"], nn.CrossEntropyLoss(), CPU, opt)
    assert grad_norm > 0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_fit_learns_a_separable_problem(loaders):
    hist = fit(_model(), loaders, CPU, epochs=30, lr=1e-2, verbose=False)
    assert isinstance(hist, History)
    assert len(hist.train_loss) == len(hist.val_acc) == len(hist.grad_norm) == 30
    assert hist.train_loss[-1] < hist.train_loss[0]
    assert max(hist.val_acc) > 0.95


def test_fit_restores_the_weights_of_the_best_validation_epoch(loaders):
    # Random validation labels: val accuracy is pure noise, so the best epoch is not the last one
    # and the test would fail if `fit` kept the final weights instead of the best ones.
    xv, _ = _blobs(40, 1)
    noisy_y = torch.randint(0, 2, (40,), generator=torch.Generator().manual_seed(4))
    loaders = {**loaders, "val": DataLoader(TensorDataset(xv, noisy_y), batch_size=16)}
    model = _model()
    hist = fit(model, loaders, CPU, epochs=20, lr=1e-2, verbose=False)
    assert max(hist.val_acc) > hist.val_acc[-1]  # precondition: the last epoch is not the best
    _, val_acc, _ = run_epoch(model, loaders["val"], nn.CrossEntropyLoss(), CPU)
    assert val_acc == pytest.approx(max(hist.val_acc))


def test_fit_early_stopping_respects_patience(loaders):
    hist = fit(_model(), loaders, CPU, epochs=50, lr=0.0, patience=3, verbose=False)  # lr=0: nothing improves
    assert len(hist.val_acc) == 4  # first epoch sets the best, then 3 epochs without gain


def test_fit_ignores_frozen_parameters(loaders):
    model = _model()
    for p in model[0].parameters():
        p.requires_grad = False
    frozen = model[0].weight.clone()
    fit(model, loaders, CPU, epochs=3, lr=1e-2, verbose=False)
    assert torch.equal(model[0].weight, frozen)


def test_predict_returns_labels_and_predictions(loaders):
    model = _model()
    y, pred = predict(model, loaders["val"], CPU)
    assert y.shape == pred.shape == (30,)
    assert set(pred) <= {0, 1}


def test_cpu_latency_is_positive_and_restores_thread_count():
    threads = torch.get_num_threads()
    ms = cpu_latency_ms(_model(), (4,), repeats=5)
    assert ms > 0
    assert torch.get_num_threads() == threads


def test_n_params():
    assert n_params(nn.Linear(10, 3)) == 33


def test_save_results_writes_json_with_history(tmp_path, monkeypatch):
    monkeypatch.setattr(train, "REPORTS", tmp_path / "reports")
    hist = History(train_loss=[1.0], train_acc=[0.5], val_loss=[1.1], val_acc=[0.4], grad_norm=[2.0])
    save_results("demo", {"test_acc": 0.5}, hist)
    saved = json.loads((tmp_path / "reports" / "demo.json").read_text())
    assert saved["test_acc"] == 0.5
    assert saved["history"]["val_acc"] == [0.4]
