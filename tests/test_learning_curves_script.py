"""The plotting and summary code of scripts/learning_curves.py, on a made-up result (no data, no model)."""
import importlib.util
import sys
from pathlib import Path

import pytest

from museum_guide.curves import images_needed

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def script():
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location("learning_curves_script", SCRIPTS / "learning_curves.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


def fake_task(n_classes=5):
    sizes = [25.0, 50.0, 100.0, 200.0, 400.0, 972.0]
    curves = {}
    for name, base in (("clip_probe", 0.8), ("resnet18_layer4", 0.4), ("random_forest", 0.3)):
        mean = [min(0.95, base + 0.02 * i + 0.1 * (name == "clip_probe") * (i > 2)) for i in range(6)]
        entry = {"mean": mean, "std": [0.02] * 6, "per_seed": [[m] for m in mean]}
        entry["images_needed"] = {f"{t:.2f}": images_needed(sizes, mean, t) for t in (0.7, 0.8, 0.85)}
        for r in entry["images_needed"].values():
            r["per_class"] = None if r["images"] is None else r["images"] / n_classes
        curves[name] = entry
    ks = [1, 2, 5]
    return {
        "chance": 1 / n_classes, "n_seeds": 1, "actual_size_mean": sizes, "curves": curves,
        "zero_shot": {"mean": 0.85, "std": 0.02, "template": "a painting of a {name}"},
        "few_shot": {"ks": ks, "draws_per_seed": 2,
                     "probe": {str(k): {"mean": 0.6 + 0.05 * i, "std": 0.03, "n_runs": 2} for i, k in enumerate(ks)},
                     "zero_shot": {str(k): {"mean": 0.85, "std": 0.0} for k in ks}},
    }


def test_plot_writes_a_figure_for_one_or_two_tasks(script, tmp_path):
    for tasks in (["genre"], ["genre", "style"]):
        path = tmp_path / f"{len(tasks)}.png"
        script.plot({"tasks": {t: fake_task() for t in tasks}}, path)
        assert path.stat().st_size > 5_000


def test_print_summary_reports_sizes_thresholds_and_few_shot(script, capsys):
    script.print_summary({"tasks": {"genre": fake_task()}})
    out = capsys.readouterr().out
    assert "genre" in out and "972" in out
    assert "images needed, clip_probe" in out
    assert "few-shot probe: k=1" in out
    assert "not reached" in out  # the weak models never reach 70%


def test_the_script_defines_the_documented_defaults(script):
    assert script.SIZES == (25, 50, 100, 200, 400)
    assert script.FEW_SHOT_KS == (1, 2, 5, 10, 20)
    assert script.TARGETS == (0.70, 0.80, 0.85)
    assert set(script.TASK_DATA) == {"genre", "style"}
