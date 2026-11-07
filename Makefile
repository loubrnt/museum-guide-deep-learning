# Shortcuts for the usual commands. Run from the repository root; use PYTHON=... to pick an interpreter.
PYTHON ?= python

.PHONY: install test coverage lint typecheck check data reproduce clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

# Tests with a coverage report of src/museum_guide (lines not covered are listed).
coverage:
	$(PYTHON) -m pytest -q --cov=museum_guide --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check .

# Static types of src/ (configured in pyproject.toml). Needs the project dependencies installed, not only mypy.
typecheck:
	$(PYTHON) -m mypy

# Everything CI runs.
check: lint typecheck coverage

# Rebuilds data/wikiart_genre5 (about 2.3 GB streamed from Hugging Face; resumes if interrupted).
data:
	$(PYTHON) project/make_wikiart_subset.py --label genre \
		--classes portrait landscape cityscape still_life religious_painting \
		--per-class 300 --size 128 --out data/wikiart_genre5

# Every experiment of the README, in order. Needs the data; takes a while (robustness alone is about 10 minutes).
reproduce:
	$(PYTHON) scripts/baseline_classical.py
	$(PYTHON) scripts/train_mlp.py
	$(PYTHON) scripts/train_resnet.py
	$(PYTHON) scripts/search_clip.py
	$(PYTHON) scripts/robustness.py --seeds 10

# Caches only: never touches data/ or reports/.
clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./data/*" -not -path "./.venv/*" -prune -exec rm -rf {} +
