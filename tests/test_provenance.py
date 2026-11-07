"""docs/PROVENANCE.md, the Makefile and the repository must tell the same story.

The map says which script writes which report and which README number comes from which key. These tests fail when
it rots: a new script or report that nobody documented, a renamed file, a key that no longer exists, an order of
execution that breaks a dependency, or a command that differs from the one in the Makefile.

If you add a script or a report, add its row to docs/PROVENANCE.md (and the script to a `reproduce-*` target of the
Makefile unless it is one of the INTERACTIVE tools below).
"""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "PROVENANCE.md").read_text()
MAKEFILE = (ROOT / "Makefile").read_text()

# Tools to use the models: they write no report, so `make reproduce-all` does not run them.
INTERACTIVE = {"predict", "guide", "search", "demo"}
SCRIPTS = sorted(p.stem for p in (ROOT / "scripts").glob("*.py"))
REPORT_SUFFIXES = {".json", ".png", ".npz"}  # committed outputs; model weights (.pt) are not committed


def parse_makefile(text: str) -> dict[str, tuple[list[str], list[str]]]:
    """target -> (prerequisites, recipe commands), with `$(PYTHON)` read as `python` and line continuations joined."""
    rules: dict[str, tuple[list[str], list[str]]] = {}
    current = None
    previous_continues = False
    for raw in text.splitlines():
        if raw.startswith("\t"):
            line = raw.strip()
            if current is None:
                continue
            if previous_continues:
                rules[current][1][-1] += " " + line.rstrip("\\").strip()
            else:
                rules[current][1].append(line.rstrip("\\").strip())
            previous_continues = raw.rstrip().endswith("\\")
            continue
        previous_continues = False
        match = re.match(r"^([A-Za-z0-9_.-]+):(?!=)\s*(.*)$", raw)
        if match and not match.group(1).startswith("."):
            current = match.group(1)
            rules[current] = (match.group(2).replace("\\", " ").split(), [])
        elif raw.rstrip().endswith("\\") and current is not None and not raw.startswith("#"):
            continue
        else:
            current = None if not raw.startswith(" ") else current
    for target, (prereqs, recipe) in rules.items():
        rules[target] = (prereqs, [re.sub(r"\s+", " ", c.replace("$(PYTHON)", "python")) for c in recipe])
    return rules


RULES = parse_makefile(MAKEFILE)


def ordered_commands(target: str) -> list[str]:
    """The recipe commands `make target` runs, in order: prerequisites left to right, each target once."""
    seen: set[str] = set()
    commands: list[str] = []

    def visit(name: str) -> None:
        if name in seen or name not in RULES:
            return
        seen.add(name)
        for prerequisite in RULES[name][0]:
            visit(prerequisite)
        commands.extend(RULES[name][1])

    visit(target)
    return commands


def script_order(commands: list[str]) -> list[str]:
    """Names of the python files run, in order (a script name occurring once per command)."""
    return [m.group(1) for c in commands if (m := re.search(r"(?:scripts|project)/(\w+)\.py", c))]


def doc_rows() -> dict[str, list[str]]:
    """script name -> cells of its row in the PROVENANCE tables (the ones with a `scripts/x.py` first cell)."""
    rows = {}
    for line in DOC.splitlines():
        match = re.match(r"^\|\s*`scripts/(\w+)\.py`", line)
        if match:
            rows[match.group(1)] = [c.strip() for c in line.strip().strip("|").split("|")]
    return rows


def resolve(report: Path, key_path: str):
    """Follow `a.b.0.c` inside a JSON file (a dot separates keys, a number indexes a list)."""
    node = json.loads(report.read_text())
    for part in key_path.split("."):
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


# --- the scripts ------------------------------------------------------------------------------------------------


def test_the_makefile_parser_sees_the_expected_targets():
    assert "reproduce-all" in RULES and "reproduce" in RULES
    assert RULES["reproduce-genre"][0] == ["data"]
    assert "python scripts/robustness.py --seeds 10" in RULES["reproduce-robustness"][1]
    # a recipe continued over several lines is joined into one command
    assert any(c.startswith("python project/make_wikiart_subset.py --label style --classes") for c in RULES["data-style"][1])


@pytest.mark.parametrize("name", SCRIPTS)
def test_every_script_is_in_the_provenance_map(name):
    assert f"`scripts/{name}.py`" in DOC, f"scripts/{name}.py has no row in docs/PROVENANCE.md"


@pytest.mark.parametrize("name", sorted(set(SCRIPTS) - INTERACTIVE))
def test_every_experiment_is_rebuilt_by_reproduce_all(name):
    assert name in script_order(ordered_commands("reproduce-all")), (
        f"scripts/{name}.py is not run by `make reproduce-all`: add it to a reproduce-* target "
        f"(or to INTERACTIVE in this test if it writes no report)"
    )


def test_the_interactive_tools_exist_and_are_not_in_reproduce_all():
    assert INTERACTIVE <= set(SCRIPTS)
    assert not INTERACTIVE & set(script_order(ordered_commands("reproduce-all")))


@pytest.mark.parametrize("name", sorted(set(SCRIPTS) - INTERACTIVE))
def test_scripts_run_after_the_ones_they_depend_on(name):
    order = script_order(ordered_commands("reproduce-all"))
    cells = doc_rows()[name]
    assert len(cells) == 6, f"the row of scripts/{name}.py should have 6 cells (script, needs, writes, supports, run after, time)"
    for dependency in re.findall(r"`scripts/(\w+)\.py`", cells[4]):
        assert dependency in order, f"scripts/{name}.py runs after scripts/{dependency}.py, which `make reproduce-all` does not run"
        assert order.index(dependency) < order.index(name), f"scripts/{dependency}.py must run before scripts/{name}.py"


@pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
def test_our_ordering_agrees_with_what_make_prints():
    out = subprocess.run(["make", "-n", "reproduce-all", "PYTHON=python"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    printed = [c.strip() for c in out.splitlines() if re.search(r"(?:scripts|project)/\w+\.py", c)]
    assert script_order(printed) == script_order(ordered_commands("reproduce-all"))


def test_commands_in_the_map_are_the_ones_in_the_makefile():
    makefile_commands = {c for _, recipe in RULES.values() for c in recipe}
    shown = re.findall(r"`((?:MUSEUM_GUIDE_DATA=\S+ )?python scripts/\w+\.py[^`]*)`", DOC)
    assert shown, "no command found in the map"
    for command in shown:
        name = re.search(r"scripts/(\w+)\.py", command).group(1)
        if name in INTERACTIVE:
            continue
        assert command in makefile_commands, f"`{command}` is in docs/PROVENANCE.md but not in the Makefile"


# --- the reports --------------------------------------------------------------------------------------------------

NAMED = sorted(set(re.findall(r"`(reports/[^`#\s*]+\.(?:json|png|npz))`", DOC)))
KEYS = sorted(set(re.findall(r"`(reports/[^`#\s*]+\.json)#([^`\s]+)`", DOC)))
COMMITTED = sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / "reports").rglob("*") if p.is_file() and p.suffix in REPORT_SUFFIXES
)


@pytest.mark.parametrize("path", NAMED)
def test_every_file_named_in_the_map_exists(path):
    assert (ROOT / path).is_file(), f"{path} is named in docs/PROVENANCE.md but is not in the repository"


@pytest.mark.parametrize("path", COMMITTED)
def test_every_committed_report_is_in_the_map(path):
    assert path in DOC, f"{path} is committed but no script row of docs/PROVENANCE.md says which script writes it"


@pytest.mark.parametrize(("path", "key"), KEYS)
def test_every_key_in_the_map_resolves(path, key):
    try:
        value = resolve(ROOT / path, key)
    except (KeyError, IndexError, ValueError, TypeError) as err:
        pytest.fail(f"{path}#{key} does not resolve ({err!r})")
    assert isinstance(value, (int, float, str, bool, list, dict))


def test_the_map_has_at_least_one_key_for_most_experiments():
    rows = doc_rows()
    with_key = [n for n in set(SCRIPTS) - INTERACTIVE if "#" in rows[n][3]]
    assert len(with_key) >= len(set(SCRIPTS) - INTERACTIVE) - 2  # fetch_non_paintings writes no report
