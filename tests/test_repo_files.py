"""The repository's own files: metadata that must parse, and a model card that must not drift from the reports."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
def _yaml(path):
    yaml = pytest.importorskip("yaml")  # PyYAML is not a dependency of the project: skip these tests where it is missing
    return yaml.safe_load((ROOT / path).read_text())


def _report(name):
    return json.loads((ROOT / "reports" / name).read_text())


def test_citation_file_is_valid_cff():
    cff = _yaml("CITATION.cff")
    assert cff["cff-version"] == "1.2.0"
    assert {"message", "title", "authors"} <= set(cff)
    names = {(a["given-names"], a["family-names"]) for a in cff["authors"]}
    assert names == {("Lou", "Brunet"), ("Mathis", "Meimoun"), ("Alexis", "Maeder")}
    assert cff["repository-code"].startswith("https://github.com/")
    assert "license" not in cff  # the authors have not chosen one yet (see CONTRIBUTING.md)


def test_dependabot_config_covers_pip_and_actions():
    config = _yaml(".github/dependabot.yml")
    assert config["version"] == 2
    ecosystems = {u["package-ecosystem"]: u for u in config["updates"]}
    assert set(ecosystems) == {"pip", "github-actions"}
    assert all(u["schedule"]["interval"] == "weekly" for u in ecosystems.values())


def test_pre_commit_config_guards_large_files_and_pins_ruff_like_pyproject():
    config = _yaml(".pre-commit-config.yaml")
    hooks = {h["id"]: (repo["rev"], h) for repo in config["repos"] for h in repo["hooks"]}
    assert {"ruff-check", "check-added-large-files", "check-yaml", "check-toml"} <= set(hooks)
    assert hooks["check-added-large-files"][1]["args"] == ["--maxkb=500"]
    pyproject = (ROOT / "pyproject.toml").read_text()
    major_minor = re.search(r"ruff>=(\d+\.\d+),<(\d+\.\d+)", pyproject)
    rev = hooks["ruff-check"][0].lstrip("v")
    assert major_minor, "pyproject.toml should pin a ruff range"
    low, high = (tuple(map(int, g.split("."))) for g in major_minor.groups())
    assert low <= tuple(map(int, rev.split(".")[:2])) < high  # same range as the dev extra


def test_pull_request_template_has_the_three_sections_and_a_checklist():
    text = (ROOT / ".github" / "PULL_REQUEST_TEMPLATE.md").read_text()
    for heading in ("## What", "## Results / Checked", "## Limits", "## Checklist"):
        assert heading in text
    assert text.count("- [ ]") >= 3


def test_contributing_states_the_rules_and_that_no_licence_is_chosen():
    text = (ROOT / "CONTRIBUTING.md").read_text()
    assert "not chosen yet" in text
    assert "pytest -q" in text and "ruff check ." in text
    assert "reports/*.json" in text
    assert not (ROOT / "LICENSE").exists()  # the choice belongs to the authors


def test_documents_point_to_each_other():
    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    card = (ROOT / "MODEL_CARD.md").read_text()
    for target in ("MODEL_CARD.md", "DATA_STATEMENT.md"):
        assert target in contributing
        assert (ROOT / target).exists()
    assert "DATA_STATEMENT.md" in card


def _pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def test_model_card_headline_numbers_match_the_reports():
    """If a report is regenerated and a headline figure moves, the card must be updated with it."""
    card = (ROOT / "MODEL_CARD.md").read_text()
    robust = _report("robustness.json")["summary"]["test_acc"]
    clip = _report("clip_classifier.json")["summary"]["test_acc"]
    probe = _report("clip_probe.json")
    style = _report("style_task.json")["test_accuracy"]
    kiosk = _report("kiosk_benchmark.json")["configs"]
    expected = [
        f"{_pct(robust['resnet18_layer4']['mean'])} ± {robust['resnet18_layer4']['std'] * 100:.1f}",
        f"{_pct(robust['random_forest']['mean'])} ± {robust['random_forest']['std'] * 100:.1f}",
        f"{_pct(clip['linear_probe']['mean'])} ± {clip['linear_probe']['std'] * 100:.1f}",
        f"{_pct(clip['zero_shot_painting']['mean'])} ± {clip['zero_shot_painting']['std'] * 100:.1f}",
        f"{_pct(style['clip_probe']['mean'])} ± {style['clip_probe']['std'] * 100:.1f}",
        f"{_pct(style['resnet18_layer4']['mean'])} ± {style['resnet18_layer4']['std'] * 100:.1f}",
        _pct(probe["test_accuracy"]),
        f"{probe['test_before']['ece']:.3f}",
        f"{probe['test_after']['ece']:.3f}",
        f"{probe['temperature']:.2f}",
        f"{probe['min_confidence']:.2f}",
        f"{kiosk['resnet18_fp32']['threads_1']['weights_mb']:.1f} MB",
        f"{kiosk['clip_fp32']['threads_1']['weights_mb']:.1f} MB",
        f"{kiosk['clip_fp32']['threads_1']['total_ms']['median_of_runs']:.1f} ms",
    ]
    missing = [e for e in expected if e not in card]
    assert not missing, f"MODEL_CARD.md no longer matches the reports for: {missing}"


def test_data_statement_quotes_the_pinned_revision_and_the_open_image_point():
    text = (ROOT / "DATA_STATEMENT.md").read_text()
    script = (ROOT / "project" / "make_wikiart_subset.py").read_text()
    revision = re.search(r'REVISION = "([0-9a-f]{40})"', script).group(1)
    assert revision in text
    for tracked in ("search_examples.png", "error_confident_mistakes.png", "error_hesitant_correct.png"):
        assert tracked in text
