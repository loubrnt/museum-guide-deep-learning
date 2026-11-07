# Shortcuts for the usual commands. Run from the repository root; use PYTHON=... to pick an interpreter.
PYTHON ?= python

.PHONY: install test coverage lint typecheck check data reproduce clean \
	install-reproduce data-style data-style-fewest data-ood check-provenance \
	reproduce-all reproduce-genre reproduce-robustness reproduce-search reproduce-clip reproduce-ood \
	reproduce-style reproduce-encoders reproduce-kiosk

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

# ---------------------------------------------------------------------------------------------------------------
# Complete reproduction, in dependency order. See docs/PROVENANCE.md for what each script needs and writes.
# `make reproduce-all` rebuilds every report of the repository: it needs more than 10 GB of downloads (see the data
# targets), and its total run time was not measured end to end. Run the groups one by one to go step by step.
# ---------------------------------------------------------------------------------------------------------------

# Extra packages used by some experiments: the multilingual encoder (search_french) and the SigLIP tokenizer
# (encoder_comparison reports SigLIP as skipped without them). Not run by the other targets.
install-reproduce:
	$(PYTHON) -m pip install -e ".[dev,multilingual]" sentencepiece protobuf

# Rebuilds data/wikiart_style6 (6 styles, about 3.3 GB streamed; the build used for the style results).
data-style:
	$(PYTHON) project/make_wikiart_subset.py --label style \
		--classes Impressionism Realism Romanticism Expressionism Post_Impressionism Symbolism \
		--per-class 300 --size 128 --strategy spread --out data/wikiart_style6

# Same classes with the script's default strategy (about 1.4 GB): only for the confounded comparison of style_task.
data-style-fewest:
	$(PYTHON) project/make_wikiart_subset.py --label style \
		--classes Impressionism Realism Romanticism Expressionism Post_Impressionism Symbolism \
		--per-class 300 --size 128 --out data/wikiart_style6_fewest

# Downloads about 400 MB of ordinary photos into data/ood/ (nothing is committed).
data-ood:
	$(PYTHON) scripts/fetch_non_paintings.py

# 1. The genre task: baseline, feed-forward network, ResNet, its calibration, error analysis, data audit.
reproduce-genre: data
	$(PYTHON) scripts/baseline_classical.py
	$(PYTHON) scripts/train_mlp.py
	$(PYTHON) scripts/train_resnet.py
	$(PYTHON) scripts/calibrate_resnet.py
	$(PYTHON) scripts/error_analysis.py
	$(PYTHON) scripts/data_audit.py

# 2. How solid the numbers are: seeds, BatchNorm option, camera-like perturbations.
reproduce-robustness: reproduce-genre
	$(PYTHON) scripts/robustness.py --seeds 10
	$(PYTHON) scripts/bn_study.py --seeds 10
	$(PYTHON) scripts/photo_robustness.py --seeds 5
	$(PYTHON) scripts/clip_photo_robustness.py

# 3. Natural-language search (the multilingual part needs `make install-reproduce`).
reproduce-search: data
	$(PYTHON) scripts/search_clip.py
	$(PYTHON) scripts/search_eval.py
	$(PYTHON) scripts/search_french.py

# 4. CLIP as a recognition model: needs reports/robustness.json (to compare with the ResNet).
reproduce-clip: reproduce-robustness reproduce-search
	$(PYTHON) scripts/clip_classifier.py --seeds 10
	$(PYTHON) scripts/train_clip_probe.py

# 5. What the guide does when the photo is not a painting.
reproduce-ood: reproduce-clip data-ood
	$(PYTHON) scripts/non_painting.py

# 6. The 6-style task, with the default build for the confounded comparison.
reproduce-style: data-style data-style-fewest
	$(PYTHON) scripts/style_task.py
	MUSEUM_GUIDE_DATA=data/wikiart_style6_fewest $(PYTHON) scripts/style_task.py --tag _fewest

# 7. Other image encoders (genre and style data; about 3.7 GB of weights).
reproduce-encoders: data data-style
	$(PYTHON) scripts/encoder_comparison.py

# 8. Kiosk budget: measures time and memory, so run it last and leave the machine idle (about 15 minutes).
reproduce-kiosk: reproduce-clip
	$(PYTHON) scripts/benchmark_kiosk.py

reproduce-all: reproduce-genre reproduce-robustness reproduce-search reproduce-clip reproduce-ood reproduce-style reproduce-encoders reproduce-kiosk

# Checks that docs/PROVENANCE.md and the targets above still match the scripts and the reports.
check-provenance:
	$(PYTHON) -m pytest -q tests/test_provenance.py

# Caches only: never touches data/ or reports/.
clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -not -path "./data/*" -not -path "./.venv/*" -prune -exec rm -rf {} +
