"""What does a model cost on a kiosk? Size on disk, peak memory, latency, int8 quantisation.

The numbers depend on the machine: report them with the hardware, and trust ratios between models
more than absolute milliseconds. Everything here is measured in the calling process; the benchmark
script runs each model in a fresh process so that memory figures do not leak from one model to the next.
"""
from __future__ import annotations

import io
import resource
import sys
import time
from collections.abc import Callable

import numpy as np
import torch
from torch import nn


def summarize_ms(times_ms) -> dict[str, float]:
    """Median, 95th percentile, mean and count of a list of durations in milliseconds."""
    t = np.asarray(times_ms, dtype=float)
    return {"median": float(np.median(t)), "p95": float(np.percentile(t, 95)), "mean": float(t.mean()), "n": int(t.size)}


def time_ms(fn: Callable[[], object], repeats: int = 50, warmup: int = 5) -> list[float]:
    """Call `fn` `warmup + repeats` times and return the durations (ms) of the last `repeats` calls."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append((time.perf_counter() - start) * 1000)
    return times


def serialized_mb(obj) -> float:
    """Size in MB (1 MB = 10^6 bytes) of `torch.save(obj)`: the weight file a kiosk would have to ship."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getbuffer().nbytes / 1e6


def peak_rss_mb() -> float:
    """Peak resident memory of this process in MB (macOS reports bytes, Linux kilobytes)."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1e3


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def select_quantized_engine() -> str:
    """Pick a quantised-kernel engine this build supports (qnnpack on ARM, x86/fbgemm on Intel) and activate it."""
    supported = [e for e in torch.backends.quantized.supported_engines if e != "none"]
    for engine in ("qnnpack", "x86", "fbgemm"):
        if engine in supported:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"no quantised engine available (supported: {supported})")


def quantize_linear_int8(model: nn.Module) -> nn.Module:
    """A copy of `model` whose nn.Linear layers store int8 weights (dynamic quantisation, activations stay float).

    Convolutions are left as they are, so this suits transformers (almost all Linear) much better than a ResNet.
    """
    select_quantized_engine()
    return torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8, inplace=False)


class ClipImageTower(nn.Module):
    """CLIP's image encoder alone (vision transformer + projection): pixel values in, image vectors out."""

    def __init__(self, vision_model: nn.Module):
        super().__init__()
        self.vision = vision_model

    def forward(self, pixel_values):
        return self.vision(pixel_values=pixel_values).image_embeds


def clip_image_tower(model_name: str = "openai/clip-vit-base-patch32") -> ClipImageTower:
    """The CLIP image tower with RANDOM weights: same shapes, size, memory and speed as the trained one.

    Only the configuration is read (a few KB, cached after the first use); no checkpoint is loaded, so the
    peak memory measured afterwards is that of serving the model, not of loading a 600 MB checkpoint.
    """
    from transformers import CLIPConfig, CLIPVisionModelWithProjection

    config = CLIPConfig.from_pretrained(model_name).vision_config
    return ClipImageTower(CLIPVisionModelWithProjection(config).eval())
