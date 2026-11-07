"""WikiArt paintings must not be published: a guard against committing images by accident.

Every image-like file in the repository must be on the allow-list `tests/tracked_images.json`, with a reason.
To add one, look at it (charts only: curves, bars, matrices, histograms), then add it to the list.
"""
import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ALLOW_LIST = ROOT / "tests" / "tracked_images.json"
# Formats that can show a painting (a deck or a PDF can embed images too).
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp", ".svg", ".ico", ".heic", ".pdf", ".pptx", ".ppt"}
# Used only when Git is not available (a copy of the tree, a container): look in the places figures live.
SCANNED_DIRS = ("reports", "notebooks", "docs", "presentation")
IMAGE_OUTPUT_KEYS = ("image/png", "image/jpeg", "image/gif", "image/svg+xml")


def tracked_files(root: Path = ROOT) -> list[str] | None:
    """Paths known to Git, or None when `root` is not a Git checkout."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return [p for p in out.stdout.decode().split("\0") if p]


def candidate_files(root: Path = ROOT) -> tuple[list[str], bool]:
    """(files to check, whether the list comes from Git)."""
    tracked = tracked_files(root)
    if tracked is not None:
        return tracked, True
    found = [str(p.relative_to(root)) for d in SCANNED_DIRS if (root / d).is_dir() for p in (root / d).rglob("*") if p.is_file()]
    return sorted(found), False


def is_image_like(path: str) -> bool:
    return Path(path).suffix.lower() in IMAGE_SUFFIXES


def allowed_images(allow_list: dict) -> set[str]:
    return set(allow_list["charts"]["files"]) | set(allow_list["needs_decision"])


def notebook_has_image_outputs(notebook: dict) -> bool:
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            if any(key in output.get("data", {}) for key in IMAGE_OUTPUT_KEYS):
                return True
    return False


def relative_image_links(markdown: str) -> list[str]:
    """Relative targets of `![alt](path)` and `<img src="path">` (web links, data URIs and anchors are skipped)."""
    from_markdown = [a or b for a, b in re.findall(r"!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^)\s]+))", markdown)]
    targets = from_markdown + re.findall(r"<img[^>]+src=[\"']([^\"']+)", markdown)
    keep = [t for t in targets if not re.match(r"^(?:[a-z][a-z0-9+.-]*:|//|#)", t, flags=re.IGNORECASE)]
    return [t.split("#")[0].split("?")[0] for t in keep]


@pytest.fixture(scope="module")
def allow_list():
    return json.loads(ALLOW_LIST.read_text())


def test_every_image_in_the_repository_is_on_the_allow_list(allow_list):
    files, _ = candidate_files()
    unknown = sorted(f for f in files if is_image_like(f) and f not in allowed_images(allow_list))
    assert not unknown, (
        f"image files not on tests/tracked_images.json: {unknown}. WikiArt paintings must not be published "
        "(see src/museum_guide/figures.py): write figures that show paintings to data/figures_local/; "
        "for a chart, look at it, then add it to the allow-list."
    )


def test_the_allow_list_has_no_stale_entries(allow_list):
    files, from_git = candidate_files()
    if not from_git:
        pytest.skip("not a Git checkout")
    stale = sorted(allowed_images(allow_list) - set(files))
    assert not stale, f"entries of tests/tracked_images.json that are no longer tracked: {stale}"


def test_the_allow_list_is_well_formed(allow_list):
    assert allow_list["charts"]["reason"].strip()
    assert all(p.startswith("reports/") for p in allow_list["charts"]["files"])
    assert all(reason.strip() for reason in allow_list["needs_decision"].values())
    assert set(allow_list["charts"]["files"]).isdisjoint(allow_list["needs_decision"])


def test_the_three_figures_with_paintings_are_no_longer_published(allow_list):
    for name in ("search_examples.png", "error_confident_mistakes.png", "error_hesitant_correct.png"):
        path = f"reports/figures/{name}"
        assert path not in allowed_images(allow_list)
        assert not (ROOT / path).exists() or tracked_files() is None  # a local copy may exist, but never a tracked one


def test_notebooks_store_image_outputs_only_when_allowed(allow_list):
    files, _ = candidate_files()
    allowed = set(allow_list.get("notebooks_with_plot_outputs", {}))
    offenders = []
    for f in files:
        if f.endswith(".ipynb") and (ROOT / f).exists():
            if notebook_has_image_outputs(json.loads((ROOT / f).read_text())) and f not in allowed:
                offenders.append(f)
    assert not offenders, (
        f"notebooks with stored image outputs not listed in 'notebooks_with_plot_outputs': {offenders}. "
        "Check that the outputs are charts built from reports/*.json, not paintings, then list the notebook with its reason."
    )


def test_every_relative_image_link_in_the_documentation_exists():
    files, from_git = candidate_files()
    tracked = set(files) if from_git else None
    docs = [f for f in files if f.endswith(".md")] if from_git else [p.name for p in ROOT.glob("*.md")]
    broken = []
    for doc in docs:
        text = (ROOT / doc).read_text(errors="ignore")
        for target in relative_image_links(text):
            resolved = (ROOT / doc).parent.joinpath(target).resolve()
            try:
                rel = str(resolved.relative_to(ROOT))
            except ValueError:
                broken.append((doc, target))
                continue
            if not resolved.exists() or (tracked is not None and rel not in tracked):
                broken.append((doc, target))
    assert not broken, f"image links that do not point to a tracked file: {broken}"


# ---- the checks themselves, on small inputs, so they cannot pass for the wrong reason ----


def test_relative_image_links_are_extracted():
    text = (
        "![chart](reports/figures/a.png) ![x](<docs/b c.png>) text [not image](c.png) "
        '<img src="d/e.png" width=10> ![web](https://x.org/f.png) ![data](data:image/png;base64,AAA) '
        "![anchor](#top) ![q](g.png?raw=1#frag)"
    )
    assert relative_image_links(text) == ["reports/figures/a.png", "docs/b c.png", "g.png", "d/e.png"]  # Markdown links, then <img> tags


def test_image_like_detection():
    assert is_image_like("reports/figures/x.PNG")
    assert is_image_like("presentation/deck.pptx")
    assert not is_image_like("reports/x.json")
    assert not is_image_like("src/museum_guide/figures.py")


def test_notebook_image_output_detection():
    plain = {"cells": [{"outputs": [{"output_type": "stream", "text": "ok"}]}, {"cell_type": "markdown"}]}
    plot = {"cells": [{"outputs": [{"output_type": "display_data", "data": {"image/png": "AAA", "text/plain": "<Figure>"}}]}]}
    assert not notebook_has_image_outputs(plain)
    assert notebook_has_image_outputs(plot)


def test_a_new_image_would_be_caught(tmp_path, allow_list):
    """Simulate an untracked-by-the-list file: the same predicate the repository test uses flags it."""
    files = ["README.md", "reports/figures/new_gallery.png", "reports/figures/robustness.png"]
    unknown = [f for f in files if is_image_like(f) and f not in allowed_images(allow_list)]
    assert unknown == ["reports/figures/new_gallery.png"]


def test_scanning_fallback_without_git(tmp_path):
    (tmp_path / "reports" / "figures").mkdir(parents=True)
    (tmp_path / "reports" / "figures" / "a.png").write_bytes(b"x")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "ignored.png").write_bytes(b"x")  # outside the scanned folders
    files, from_git = candidate_files(tmp_path)
    assert from_git is False
    assert files == ["reports/figures/a.png"]
