"""Where figures that show paintings are written, and that the two scripts that draw them use it."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

import museum_guide.data as data
from museum_guide import figures
from museum_guide.figures import ENV_DIR, ENV_TRACK, painting_figure_path, painting_figures_dir

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv(ENV_DIR, raising=False)
    monkeypatch.delenv(ENV_TRACK, raising=False)
    monkeypatch.setattr(data, "DATA_DIR", Path("data/wikiart_genre5"))


def test_default_is_a_gitignored_folder_next_to_the_data():
    assert painting_figures_dir() == Path("data/figures_local")


def test_the_default_follows_the_data_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "DATA_DIR", tmp_path / "wikiart_style6")
    assert painting_figures_dir() == tmp_path / "figures_local"


def test_environment_override(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "mine"))
    assert painting_figures_dir() == tmp_path / "mine"


def test_opt_in_to_the_old_behaviour(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "mine"))
    monkeypatch.setenv(ENV_TRACK, "1")
    assert painting_figures_dir() == figures.TRACKED_DIR == Path("reports/figures")


def test_only_the_value_1_enables_the_old_behaviour(monkeypatch):
    monkeypatch.setenv(ENV_TRACK, "0")
    assert painting_figures_dir() == Path("data/figures_local")


def test_painting_figure_path_creates_the_folder(monkeypatch, tmp_path):
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "a" / "b"))
    path = painting_figure_path("x.png")
    assert path == tmp_path / "a" / "b" / "x.png"
    assert path.parent.is_dir()


def test_default_folder_is_covered_by_gitignore():
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text().splitlines()
    assert "data/figures_local/" in ignore
    assert "/data" in ignore  # a symlink named `data` is not matched by `data/`


def _load_script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))  # scripts import their siblings
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script, figures_named", [
    ("search_clip", ["search_examples.png"]),
    ("error_analysis", ["error_confident_mistakes.png", "error_hesitant_correct.png"]),
])
def test_scripts_send_painting_figures_to_the_local_folder(script, figures_named):
    source = (SCRIPTS / f"{script}.py").read_text()
    assert "painting_figure_path(" in source
    for name in figures_named:
        assert f'painting_figure_path("{name}")' in source
        assert f'"figures" / "{name}"' not in source and f'FIG / "{name}"' not in source


def test_error_analysis_gallery_writes_where_it_is_told(tmp_path, monkeypatch):
    """A tiny synthetic run of the real gallery function: the image lands in the local folder, not in reports/."""
    module = _load_script("error_analysis", monkeypatch)
    images = tmp_path / "images"
    images.mkdir()
    rows = []
    for i in range(4):
        Image.fromarray(np.full((16, 16, 3), 40 * i, dtype=np.uint8)).save(images / f"{i}.png")
        rows.append({"path": f"{i}.png", "label": "portrait"})
    monkeypatch.setattr(module, "DATA_DIR", images)
    monkeypatch.setenv(ENV_DIR, str(tmp_path / "local"))
    target = module.painting_figure_path("error_confident_mistakes.png")
    probs = np.tile([0.1, 0.9], (4, 1))
    module.gallery(pd.DataFrame(rows), [0, 1, 2, 3], probs, ["portrait", "landscape"], "synthetic", target)
    assert target.exists() and target.parent == tmp_path / "local"
    assert not Path("reports/figures/error_confident_mistakes.png").exists()
