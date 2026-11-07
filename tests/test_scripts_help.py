"""Every script in scripts/ starts and answers `--help`: a broken import or a typo is caught without running anything.

The scripts are experiments, not importable modules, so unit tests cannot reach them. Running `--help` imports
everything a script needs (PyTorch, scikit-learn, the project package) and parses its command line.

Scripts are started from an empty temporary folder: if one ever ran its experiment on `--help` again (some did,
before they got a minimal command line), it would fail at once for lack of data instead of overwriting `reports/`.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]), "PYTHONWARNINGS": "ignore"}


def _help(script: Path, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True, timeout=180, env=ENV, cwd=cwd)


@pytest.fixture(scope="module")
def help_results(tmp_path_factory):
    """`--help` of every script, run a few at a time (each one imports PyTorch, which takes a few seconds)."""
    cwd = tmp_path_factory.mktemp("empty_cwd")
    with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 2)) as pool:
        return dict(zip((s.name for s in SCRIPTS), pool.map(lambda s: _help(s, cwd), SCRIPTS)))


def test_there_are_scripts_to_check():
    assert len(SCRIPTS) >= 20  # a wrong path would make every test below pass vacuously


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_script_answers_help(script, help_results):
    result = help_results[script.name]
    assert result.returncode == 0, result.stderr[-800:]
    assert "usage:" in result.stdout.lower()
    assert script.stem in result.stdout  # the usage line names the script, so it is its own parser that answered


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda s: s.name)
def test_script_only_runs_when_executed_not_when_imported(script):
    assert '__name__ == "__main__"' in script.read_text(), f"{script.name} would run its experiment on import"


def test_a_typo_in_an_option_is_rejected_not_ignored(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "baseline_classical.py"), "--oops"], capture_output=True, text=True, timeout=180, env=ENV, cwd=tmp_path
    )
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr
