"""Where to write figures that SHOW paintings.

The WikiArt dataset card says "license: unknown" and the course script (`project/make_wikiart_subset.py`)
asks not to publish the images. A figure that tiles painting thumbnails (a gallery of mistakes, the
answers to a question) is a redistribution of those images, so such figures are written to a gitignored
folder next to the data, `data/figures_local/`, and `reports/figures/` holds charts only.

    MUSEUM_GUIDE_FIGURES_LOCAL=/some/dir   write them there instead
    MUSEUM_GUIDE_TRACK_PAINTING_FIGURES=1  write them to reports/figures/ (the old behaviour; the team's decision)

`tests/test_tracked_figures.py` fails if an image that is not on its allow-list gets committed.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_DIR = "MUSEUM_GUIDE_FIGURES_LOCAL"
ENV_TRACK = "MUSEUM_GUIDE_TRACK_PAINTING_FIGURES"
TRACKED_DIR = Path("reports") / "figures"


def painting_figures_dir() -> Path:
    """Folder for figures that show paintings (not created here)."""
    if os.environ.get(ENV_TRACK) == "1":
        return TRACKED_DIR
    if os.environ.get(ENV_DIR):
        return Path(os.environ[ENV_DIR])
    from museum_guide.data import DATA_DIR  # next to the data, which is not versioned

    return DATA_DIR.parent / "figures_local"


def painting_figure_path(name: str) -> Path:
    """Path of the figure `name` that shows paintings; its folder is created."""
    folder = painting_figures_dir()
    folder.mkdir(parents=True, exist_ok=True)
    return folder / name
