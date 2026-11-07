"""The walkthrough notebook reads committed results only, and must not publish any painting."""
import ast
import json
import re
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "museum_guide_walkthrough.ipynb"
IMAGE_CALLS = ("imshow", "Image.open", "imread", "PIL")


def _source(cell) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


@pytest.fixture(scope="module")
def nb():
    return json.loads(NOTEBOOK.read_text())


def _code(nb):
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _optional_cell(nb):
    cells = [c for c in _code(nb) if "RUN_MODELS" in _source(c)]
    assert len(cells) == 1, "exactly one cell (the optional section) may use RUN_MODELS"
    return cells[0]


def test_notebook_structure(nb):
    assert nb["nbformat"] == 4
    assert nb["metadata"]["kernelspec"]["name"] == "python3"
    ids = [c["id"] for c in nb["cells"]]
    assert len(ids) == len(set(ids)), "cell ids must be unique (and stable, for clean diffs)"
    assert {c["cell_type"] for c in nb["cells"]} == {"markdown", "code"}
    assert NOTEBOOK.stat().st_size < 1_000_000


def test_no_volatile_metadata(nb):
    """Execution counts, timestamps and cell metadata are stripped so that a rerun gives a clean diff."""
    assert set(nb["metadata"]) <= {"kernelspec", "language_info"}
    for cell in nb["cells"]:
        assert cell["metadata"] == {}
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            for out in cell["outputs"]:
                assert out.get("execution_count") is None
                assert out.get("metadata", {}) == {}


def test_valid_nbformat():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))


def test_every_code_cell_compiles(nb):
    for cell in _code(nb):
        ast.parse(_source(cell))


def test_every_report_it_loads_exists(nb):
    loaded = set()
    for cell in _code(nb):
        loaded.update(re.findall(r"""\bR\(\s*["']([A-Za-z0-9_]+)["']\s*\)""", _source(cell)))
        loaded.update(re.findall(r"""reports/([A-Za-z0-9_]+)\.json""", _source(cell)))
    assert len(loaded) >= 15, "the notebook is meant to cover the project's reports"
    missing = sorted(name for name in loaded if not (ROOT / "reports" / f"{name}.json").exists())
    assert not missing, f"reports loaded by the notebook but missing from the repository: {missing}"


def test_dataset_images_only_in_the_optional_section(nb):
    """No cell outside the guarded optional section may open an image or read the data folder."""
    optional = _optional_cell(nb)
    for cell in _code(nb):
        if cell is optional:
            continue
        src = _source(cell)
        for pattern in IMAGE_CALLS:
            assert pattern not in src, f"cell {cell['id']} uses {pattern}"
        assert "data/" not in src and '"data"' not in src, f"cell {cell['id']} reads the data folder"


def test_optional_section_is_off_and_guarded(nb):
    src = _source(_optional_cell(nb))
    assert re.search(r"^RUN_MODELS = False\b", src, re.M), "the models must not run by default"
    src = re.sub(r"#.*$", "", src, flags=re.M)  # comments may mention the data folder
    guard = src.index("if RUN_MODELS:")
    for needle in ("data", "subprocess", "guide.py", "predict.py"):
        for m in re.finditer(needle, src):
            assert m.start() > guard, f"{needle!r} is used before the RUN_MODELS guard"
    assert not any(p in src for p in ("imshow", "Image.open", "plt.imread")), "the optional section prints text only"


def test_outputs_are_charts_tables_and_text_only(nb):
    for cell in _code(nb):
        for out in cell["outputs"]:
            assert out["output_type"] in {"stream", "display_data", "execute_result"}, f"cell {cell['id']}: {out['output_type']}"
            if out["output_type"] == "stream":
                text = "".join(out["text"]) if isinstance(out["text"], list) else out["text"]
                assert "/Users/" not in text and "/private/" not in text and "/tmp/" not in text, "a local path leaked into an output"
            if "image/png" in out.get("data", {}):
                src = _source(cell)
                assert "plt.show()" in src and "R(" in src, f"cell {cell['id']} shows an image that does not come from a plot of report data"
                assert not any(p in src for p in IMAGE_CALLS)


def test_markdown_has_no_hand_typed_results(nb):
    """Numbers come from the report files (printed by code cells): the prose carries none."""
    allowed = ("ResNet-18", "ViT-B/32", "B/32", "int8", "fp16", "fp32", "L/14", "B/16")
    for cell in nb["cells"]:
        if cell["cell_type"] != "markdown":
            continue
        text = _source(cell)
        text = re.sub(r"^#+ \d+\. ", "# ", text, flags=re.M)  # section numbering
        for token in allowed:
            text = text.replace(token, "")
        assert not re.search(r"\d+(\.\d+)?\s?%", text), f"a percentage is typed in cell {cell['id']}"
        assert not re.search(r"\d+\.\d+", text), f"a decimal number is typed in cell {cell['id']}"
        assert not re.search(r"\b\d{2,}\b", text), f"a number is typed in cell {cell['id']}"


def test_notebook_executes_quickly_from_the_committed_reports():
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    pytest.importorskip("ipykernel")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    start = time.perf_counter()
    client.execute()
    assert time.perf_counter() - start < 120
    skipped = "".join(
        o.get("text", "") for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "stream"
    )
    assert "Skipped: set RUN_MODELS = True" in skipped  # the model section stayed off
    for c in nb.cells:
        if c.cell_type == "code":
            assert not any(o.output_type == "error" for o in c.outputs)
