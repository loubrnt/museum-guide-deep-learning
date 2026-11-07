import numpy as np
import pytest
import torch
from torch import nn

from museum_guide.benchmark import (
    ClipImageTower,
    n_params,
    peak_rss_mb,
    quantize_linear_int8,
    serialized_mb,
    summarize_ms,
    time_ms,
)


def test_summarize_ms():
    s = summarize_ms([1, 2, 3, 4, 100])
    assert s["median"] == 3
    assert s["mean"] == pytest.approx(22.0)
    assert s["n"] == 5
    assert s["median"] < s["p95"] <= 100


def test_time_ms_counts_calls_and_skips_warmup():
    calls = []
    times = time_ms(lambda: calls.append(1), repeats=7, warmup=3)
    assert len(times) == 7
    assert len(calls) == 10
    assert all(t >= 0 for t in times)


def test_serialized_size_follows_the_number_of_bytes():
    layer = nn.Linear(1000, 1000, bias=False)  # 10^6 float32 values = 4 MB
    assert serialized_mb(layer.state_dict()) == pytest.approx(4.0, rel=0.02)
    assert serialized_mb(layer.half().state_dict()) == pytest.approx(2.0, rel=0.02)


def test_peak_rss_is_a_plausible_number_of_megabytes():
    rss = peak_rss_mb()
    assert 10 < rss < 1e6


def test_n_params():
    assert n_params(nn.Linear(10, 3)) == 33


def test_int8_quantisation_shrinks_linear_layers_and_keeps_outputs_close():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 64)).eval()
    try:
        quantised = quantize_linear_int8(model)
    except RuntimeError as err:  # a PyTorch build without any quantised engine
        pytest.skip(str(err))
    assert serialized_mb(quantised.state_dict()) < 0.4 * serialized_mb(model.state_dict())
    x = torch.randn(8, 256)
    with torch.no_grad():
        reference, approx = model(x), quantised(x)
    cosine = nn.functional.cosine_similarity(reference, approx, dim=1)
    assert cosine.min() > 0.98
    assert not np.array_equal(reference.numpy(), approx.numpy())  # it is an approximation
    assert isinstance(model[0], nn.Linear)  # the original model is untouched


def test_clip_tower_wrapper_returns_the_vision_embedding():
    class FakeOutput:
        image_embeds = torch.ones(2, 4)

    class FakeVision(nn.Module):
        def forward(self, pixel_values):
            assert pixel_values.shape == (2, 3, 8, 8)
            return FakeOutput()

    out = ClipImageTower(FakeVision())(torch.zeros(2, 3, 8, 8))
    assert out.shape == (2, 4)
