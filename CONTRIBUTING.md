# Contributing

This is a three-person course project (group 5: Lou Brunet, Mathis Meimoun, Alexis Maeder), but it is run like a
small open-source repository: small pull requests, measured results, and a review before merging.

**Licence: not chosen yet.** No `LICENSE` file has been added; the choice belongs to the three authors.
Until they decide, the code is not offered under an open-source licence. The WikiArt images are covered by a separate warning (see below).

## Set up

```bash
pip install -e ".[dev]"      # the package, pytest and ruff
pytest -q                    # the tests: synthetic data, no download, no GPU
ruff check .                 # lint (configured in pyproject.toml)
```

`make test` and `make lint` do the same, and the [Makefile](Makefile) has the other targets (`install`, `data`, `reproduce`, `clean`).
Run everything from the repository root.

Optional extras: `pip install -e ".[demo]"` for the web demo and `pip install -e ".[multilingual]"` for the
multilingual search encoder. The tests and the CI need neither.

### Optional: checks before each commit

```bash
pip install pre-commit
pre-commit install
```

The hooks (see [.pre-commit-config.yaml](.pre-commit-config.yaml)) run ruff, fix trailing whitespace and end-of-file newlines,
check YAML and TOML, and **refuse any added file larger than 500 KB** — a quick guard against committing data, weights or big images by mistake.
The CI does not depend on pre-commit.

## Data

`data/` is not versioned. Rebuild the subsets with the commands in the README (`make data` rebuilds the genre subset; the README also
gives the style subset). Most scripts read `data/wikiart_genre5` relative to the working directory; set
`MUSEUM_GUIDE_DATA=/path/to/dir` to put it elsewhere.

## Pull requests

- **Keep them small and focused**: one idea per pull request. If it depends on another open pull request, say so at the top of the description
  ("Stacked on #N, merge that one first") and base it on that branch.
- **Use the template** (`.github/PULL_REQUEST_TEMPLATE.md`): *What*, *Results* or *Checked*, *Limits*, then the checklist.
- **Measure, then write.** Run the experiment on the real data, put the numbers in the README and in the description, and **cross-check every number against
  `reports/*.json`**. A figure typed by hand is how wrong numbers end up in a report. Report negative results honestly.
- **Say what was not checked**: one machine, few seeds, proxies instead of real photos, drafts waiting for human review.
- **Tests**: new reusable code lives in `src/museum_guide/` with tests in `tests/` that use synthetic data or injected fake encoders (no network, no model download).
  Scripts in `scripts/` run the experiments and write `reports/<name>.json` and `reports/figures/<name>.png`.
- **README edits are additive and small**, at the place where the reader expects the topic. Several pull requests touch the README; conflicts are easier to resolve when each adds its own subsection.
- **Commits**: English, imperative mood, a short first line ("Add a data audit: near-duplicates, split composition, metadata-only floors"), then a few lines of why when it helps.
  Pull requests are merged with *squash*, so the pull request title becomes the commit message. Do not rewrite the history of `main`.

## What must never be committed

- **Data**: anything under `data/`, including the downloaded subsets, the CLIP vector caches and the search index.
- **Model weights**: `*.pt`, `*.pth`, `*.ckpt` (already in `.gitignore`). The tiny `reports/clip_probe.npz` (a few KB) is the one exception, on purpose.
- **Images of paintings, and third-party photos.** WikiArt images remain the property of their authors or rights holders and the dataset card says "license: unknown":
  the course script itself says not to commit or publish them. Do not add screenshots, notebook outputs or figures that show them, and do not add photos of people.
  Some figures already tracked in `reports/figures/` and the slides show paintings; that is an open point (see [DATA_STATEMENT.md](DATA_STATEMENT.md)).
- **Secrets**: `.env` files, tokens, keys.

## Regenerating results

Every number in the README comes from a `reports/*.json` file written by a script. For example `python scripts/robustness.py --seeds 10` writes
`reports/robustness.json`. The README's "Reproduce" section lists the commands, and `make reproduce` runs the main ones in order (it needs the data and takes a while).
After regenerating, re-read the README sentences that quote the changed numbers.

## Models and data documentation

- [MODEL_CARD.md](MODEL_CARD.md): what the proposed model is, what it was measured on, and where it should not be used.
- [DATA_STATEMENT.md](DATA_STATEMENT.md): where the data comes from, how the subsets were built, and the open copyright point.

If a change moves a number quoted there, update the card; `tests/test_repo_files.py` fails when the headline figures of the card no longer match the reports.
