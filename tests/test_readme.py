"""The README is long: its contents list and its internal links must stay correct as sections move."""
import re
from pathlib import Path

README = (Path(__file__).resolve().parents[1] / "README.md").read_text()


def _slug(title: str) -> str:
    """GitHub's anchor for a heading: lower case, no punctuation, spaces to hyphens."""
    title = re.sub(r"[`*_]", "", title.lower())
    return re.sub(r"[^\w\- ]", "", title).strip().replace(" ", "-")


def _headings(levels: str = "#{1,6}") -> list[tuple[int, str]]:
    """Headings outside code blocks (a `# comment` inside a shell block is not a heading)."""
    found, fenced = [], False
    for line in README.split("\n"):
        if line.startswith("```"):
            fenced = not fenced
        match = re.match(rf"^({levels}) (.+)$", line)
        if match and not fenced:
            found.append((len(match.group(1)), match.group(2).strip()))
    return found


def _anchors() -> set[str]:
    seen: dict[str, int] = {}
    anchors = set()
    for _, title in _headings():
        slug = _slug(title)
        n = seen.get(slug, 0)
        seen[slug] = n + 1
        anchors.add(slug if n == 0 else f"{slug}-{n}")
    return anchors


def test_every_internal_link_points_to_a_heading():
    broken = [a for a in re.findall(r"\]\(#([^)]+)\)", README) if a not in _anchors()]
    assert not broken, f"README links to anchors that do not exist: {broken}"


def test_the_contents_list_has_every_section_and_subsection():
    contents = README.split("## Contents", 1)[1].split("\n## ", 1)[0]
    listed = re.findall(r"\[([^\]]+)\]\(#", contents)
    expected = [t for lvl, t in _headings("#{2,3}") if t != "Contents"]
    assert listed == expected, "the Contents list of the README is out of date: regenerate it from the headings"
