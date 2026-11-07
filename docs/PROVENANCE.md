# Where each number comes from

Every number in the README comes from a file in `reports/`, and every file in `reports/` is written by one script of `scripts/`.
This page is the map: for each script, the exact command, what it needs, what it writes, and which README claim it supports.
It is checked by `tests/test_provenance.py`, which fails when a script, a report or a figure is missing from it, when a file named here
does not exist, when a `file.json#key` below does not resolve, or when the commands here and the `Makefile` disagree.

## What is committed and what is rebuilt

| Committed in git | Rebuilt on your machine (not committed) |
|---|---|
| the `.json` files of `reports/`, `reports/clip_probe.npz`, the `.png` files of `reports/figures/` | `data/` in full: the WikiArt subsets, `data/ood/` photos, and every embedding cache (`clip_embeddings.npz`, `search_index.npz`, `enc_*.npz`) |
| the code, the tests, this page | `reports/resnet18.pt` and `reports/mlp.pt` (model weights), the Hugging Face model cache |

Re-running a script overwrites its reports. Seed 0 is the split of the README tables, and `scripts/robustness.py` is built to reproduce them (see its docstring).
The three data sets are built with the commands in the `Makefile` (`make data`, `make data-style`, `make data-style-fewest`, `make data-ood`); download sizes
below are from the README and the scripts.

| Data | Command | Download |
|---|---|---|
| Genre subset, 1500 images: `data/wikiart_genre5` | `make data` | about 2.3 GB |
| Style subset, 1800 images: `data/wikiart_style6` | `make data-style` | about 3.3 GB |
| Style subset, default strategy: `data/wikiart_style6_fewest` (only for the confounded comparison) | `make data-style-fewest` | about 1.4 GB |
| Non-painting photos: `data/ood/` | `make data-ood` | about 400 MB |

## Run everything

`make install-reproduce` once (adds the multilingual encoder and the SigLIP tokenizer), then `make reproduce-all`. Total run time was **not measured end to end**;
the per-script times below are only those already stated in the README or in the scripts. The groups can be run one at a time and run in this order:

1. `make reproduce-genre`: baseline, feed-forward network, ResNet, calibration, error analysis, data audit.
2. `make reproduce-robustness`: seeds and confidence intervals, BatchNorm option, camera-like perturbations.
3. `make reproduce-search`: search, its query set, French.
4. `make reproduce-clip`: CLIP as a recognition model and its saved head.
5. `make reproduce-ood`: the non-painting rule.
6. `make reproduce-style`: the 6-style task.
7. `make reproduce-encoders`: other image encoders.
8. `make reproduce-curves`: how many labelled paintings are needed.
9. `make reproduce-kiosk`: time and memory; leave the machine idle.

The older `make reproduce` (5 scripts) is kept as it was.

How to read the tables: **Run after** lists scripts whose output the script reads (`make reproduce-all` respects it). **Supports** gives the README claim and the
place it comes from as `file.json#key.path` inside `reports/` (a dot separates nested keys; a number indexes a list); the value is written in plain text next to it.

## The genre task

| Script and command | Needs | Writes | Supports (README) | Run after | Time |
|---|---|---|---|---|---|
| `scripts/baseline_classical.py`: logistic regression and random forest on colour and edge features. `python scripts/baseline_classical.py` | genre data | `reports/baseline_classical.json`, `reports/figures/cm_logistic_regression.png`, `reports/figures/cm_random_forest.png` | Results table: logistic regression 50.2% `reports/baseline_classical.json#logistic_regression.test_acc`, random forest 53.8% `reports/baseline_classical.json#random_forest.test_acc` | none | not measured |
| `scripts/train_mlp.py`: feed-forward network, sanity check, learning-rate and regularisation studies, shift test. `python scripts/train_mlp.py` | genre data | `reports/mlp.json`, `reports/figures/mlp_sanity.png`, `reports/figures/mlp_lr_0.1.png`, `reports/figures/mlp_lr_0.01.png`, `reports/figures/mlp_lr_0.001.png`, `reports/figures/mlp_lr_0.0001.png`, `reports/figures/mlp_lr_1e-05.png`, `reports/figures/mlp_reg_none.png`, `reports/figures/mlp_reg_dropout_wd.png`, `reports/figures/mlp_reg_dropout_wd_augment.png`, `reports/figures/cm_mlp.png`; weights `reports/mlp.pt` (not committed) | Results table: MLP 41.9% `reports/mlp.json#final.test_acc`; failure diagnosis: 63% of predictions change under a shift `reports/mlp.json#final.prediction_change_rate_shifted` | none | not measured |
| `scripts/train_resnet.py`: ResNet-18 from scratch, head only, and `layer4` fine-tuned. `python scripts/train_resnet.py` | genre data | `reports/resnet.json`, `reports/figures/resnet_scratch.png`, `reports/figures/resnet_pretrained_head_only.png`, `reports/figures/resnet_pretrained_layer4.png`, `reports/figures/cm_resnet.png`; weights `reports/resnet18.pt` (not committed) | Results table: fine-tuned ResNet-18 74.6% `reports/resnet.json#final.test_acc`; from scratch, validation 46.2% `reports/resnet.json#scratch.best_val_acc` | none | not measured |
| `scripts/calibrate_resnet.py`: temperature scaling of the ResNet and the "unsure" rule. `python scripts/calibrate_resnet.py` | `reports/resnet18.pt` | `reports/calibration.json`, `reports/figures/calibration.png` | Calibration: temperature 2.05 `reports/calibration.json#temperature`, error 0.151 `reports/calibration.json#test_before.ece` to 0.077 `reports/calibration.json#test_after.ece` | `scripts/train_resnet.py` | not measured |
| `scripts/error_analysis.py`: per-class metrics, accuracy by artist type and style, confident mistakes. `python scripts/error_analysis.py` | genre data | `reports/error_analysis.json`, and, locally only (they show paintings, so they are not committed), `data/figures_local/error_confident_mistakes.png` and `data/figures_local/error_hesitant_correct.png` | Per-class results: ResNet macro-F1 0.746 `reports/error_analysis.json#models.resnet18_layer4.macro_f1` | none | not measured |
| `scripts/data_audit.py`: near-duplicates, split composition, metadata-only floors. `python scripts/data_audit.py` | genre data | `reports/data_audit.json` | Sanity checks: the painting's style alone predicts the genre at 45.1% `reports/data_audit.json#metadata_floors.style.mean` | none | about 10 seconds (README) |

## How solid the numbers are

| Script and command | Needs | Writes | Supports (README) | Run after | Time |
|---|---|---|---|---|---|
| `scripts/robustness.py`: the four models over 10 seeds (split and training), artist-level confidence intervals, shift test. `python scripts/robustness.py --seeds 10` | genre data | `reports/robustness.json`, `reports/figures/robustness.png` | 10 seeds: ResNet-18 74.0% `reports/robustness.json#summary.test_acc.resnet18_layer4.mean`; gap to the random forest on seed 0, +20.8 points `reports/robustness.json#seed0_paired_differences_95.resnet18_layer4_minus_random_forest.diff` | none | about 10 minutes (README) |
| `scripts/bn_study.py`: frozen BatchNorm layers in eval mode, or not. `python scripts/bn_study.py --seeds 10` | genre data | `reports/bn_study.json`, `reports/figures/bn_study.png` | Frozen BatchNorm: head only gains 4.3 points `reports/bn_study.json#summary.head_only.paired_difference.mean`; `layer4` improves on 6 seeds of 10 `reports/bn_study.json#summary.layer4.seeds_improved` | none | not measured |
| `scripts/photo_robustness.py`: camera-like perturbations, with and without photo-style augmentation. `python scripts/photo_robustness.py --seeds 5` | genre data | `reports/photo_robustness.json`, `reports/figures/photo_robustness.png` | Robustness to camera conditions: a 15 degree rotation takes the ResNet to 51.7% `reports/photo_robustness.json#accuracy.standard.rotation_15deg.mean`, 62.7% with photo augmentation `reports/photo_robustness.json#accuracy.photo.rotation_15deg.mean` | none | not measured |
| `scripts/clip_photo_robustness.py`: the CLIP head under camera-like perturbations and shifts, with and without photo-style augmentation. `python scripts/clip_photo_robustness.py` | genre data, CLIP weights, the committed `reports/clip_probe.npz`; the ResNet side is read from `reports/photo_robustness.json` | `reports/clip_photo_robustness.json`, `reports/figures/clip_photo_robustness.png` | Robustness to camera conditions: the CLIP head scores 89.0% on clean images `reports/clip_photo_robustness.json#accuracy.standard.clean.mean` and 78.5% after a 15 degree rotation `reports/clip_photo_robustness.json#accuracy.standard.rotation_15deg.mean` | `scripts/photo_robustness.py` | not measured |

## CLIP and the search

| Script and command | Needs | Writes | Supports (README) | Run after | Time |
|---|---|---|---|---|---|
| `scripts/clip_classifier.py`: CLIP zero-shot and linear probe against the ResNet, 10 seeds. `python scripts/clip_classifier.py --seeds 10` | genre data, `reports/robustness.json`, CLIP weights (about 600 MB at first use); builds the cache `data/wikiart_genre5/clip_embeddings.npz` | `reports/clip_classifier.json`, `reports/figures/clip_classifier.png` | CLIP as a recognition model: linear probe 90.2% `reports/clip_classifier.json#summary.test_acc.linear_probe.mean`, zero-shot 85.1% `reports/clip_classifier.json#summary.test_acc.zero_shot_painting.mean`; image tower 351.4 MB `reports/clip_classifier.json#cost.fp32_mb_image_tower` | `scripts/robustness.py` | not measured |
| `scripts/train_clip_probe.py`: refits and calibrates the CLIP head saved for the guide. `python scripts/train_clip_probe.py` | the cache `clip_embeddings.npz` (computed, about 30 s, if missing) | `reports/clip_probe.npz`, `reports/clip_probe.json`, `reports/figures/clip_probe_calibration.png` | One model for recognition and search: the saved head scores 89.6% on seed 0 `reports/clip_probe.json#test_accuracy`, with temperature 0.44 `reports/clip_probe.json#temperature`; it checks that the seed-0 probe of `scripts/clip_classifier.py` is reproduced | `scripts/clip_classifier.py` | seconds (README) |
| `scripts/search_clip.py`: the first search experiment, 5 genre questions and 5 free ones. `python scripts/search_clip.py` | genre data, CLIP weights | `reports/search_clip.json`, and, locally only (it shows paintings, so it is not committed), `data/figures_local/search_examples.png` | Natural-language search: precision@10 = 1.00 on the genre questions `reports/search_clip.json#mean_precision@10` | none | not measured |
| `scripts/search_eval.py`: the 66-question query set labelled from the metadata. `python scripts/search_eval.py` | genre data, the search index `data/search_index.npz` (built on first use, about 30 s) | `reports/search_queries.json`, `reports/search_eval.json`, `reports/figures/search_eval.png` | Searching, and a measurable query set: 66 questions `reports/search_eval.json#n_queries`, precision@10 0.37 `reports/search_eval.json#summary.bare.all.precision@10` | none | not measured |
| `scripts/search_french.py`: the same questions in French, with a multilingual text encoder. `python scripts/search_french.py` | `pip install -e ".[multilingual]"`, the search index, `scripts/search_eval.py` (it imports it) | `reports/search_french.json`, `reports/search_queries_fr.json`, `reports/figures/search_french.png` | Questions in French: average precision falls by 0.064 `reports/search_french.json#paired_average_precision.clip_fr_minus_clip_en.mean_average_precision_difference` | none | not measured |
| `scripts/search_hybrid_eval.py`: the question parser (artist, style, genre) and the catalogue-then-CLIP search; reads the hand-made judgments in `reports/search_hybrid_manual.json`. `python scripts/search_hybrid_eval.py` | genre data, CLIP weights, the search index, `reports/search_hybrid_manual.json` (typed by hand, not written by the script; its image grids are saved with `--grids` under `data/hybrid_manual/`, not versioned) | `reports/search_hybrid.json`, `reports/figures/search_hybrid.png` | Using the catalogue metadata: the parser recovers every field on 95.5% of the English questions `reports/search_hybrid.json#parser.english.exact_match_rate` and 97.0% of the French ones `reports/search_hybrid.json#parser.french.exact_match_rate`; on 13 hand-checked questions, 63.1% of the top 5 satisfy both the catalogue and the visual part `reports/search_hybrid.json#manual_check.mean_top5.hybrid.both` | `scripts/search_eval.py`, `scripts/search_french.py` | not measured |

## Style and other encoders

| Script and command | Needs | Writes | Supports (README) | Run after | Time |
|---|---|---|---|---|---|
| `scripts/style_task.py`: the 6-style task, CLIP and ResNet and floors, 10 seeds. `python scripts/style_task.py` and, for the confounded comparison, `MUSEUM_GUIDE_DATA=data/wikiart_style6_fewest python scripts/style_task.py --tag _fewest` | style data (`data/wikiart_style6`; the second command needs `data/wikiart_style6_fewest`), CLIP weights | `reports/style_task.json`, `reports/figures/style_task.png`, `reports/figures/style_task_confusion.png`, `reports/style_task_fewest.json`, `reports/figures/style_task_fewest.png`, `reports/figures/style_task_confusion_fewest.png` | Recognising the style: CLIP probe 55.4% `reports/style_task.json#test_accuracy.clip_probe.mean`, ResNet 40.6% `reports/style_task.json#test_accuracy.resnet18_layer4.mean`; on the default build the ResNet reaches 46.4% `reports/style_task_fewest.json#test_accuracy.resnet18_layer4.mean` | none | about 5 minutes (README) |
| `scripts/encoder_comparison.py`: eight frozen encoders, accuracy against cost. `python scripts/encoder_comparison.py` | genre data and style data, about 3.7 GB of weights (SigLIP also needs `sentencepiece` and `protobuf`), an idle machine for the timings | `reports/encoder_comparison.json`, `reports/figures/encoder_comparison.png` | Other image encoders: CLIP B/32 probe 90.2% on genre `reports/encoder_comparison.json#accuracy.genre.clip_b32.probe_acc.mean` | none | not measured |
| `scripts/clip_adapt.py`: does adapting CLIP (richer frozen features, fine-tuning the last blocks, LoRA) beat the frozen linear probe? `python scripts/clip_adapt.py` (style), `python scripts/clip_adapt.py --task style --lr-encoder 3e-5 --lr-head 3e-4 --lr-lora 1e-3 --patience 3 --epochs 8 --tag _lr3e-5 --variants probe finetune_1 finetune_2 lora_last4` (a second setting) and `python scripts/clip_adapt.py --task genre` | genre data and style data, CLIP weights (about 600 MB at first use), the Apple GPU or a patient CPU; builds the cache `clip_last4.npz` next to each `clip_embeddings.npz` | `reports/clip_adapt_style.json`, `reports/clip_adapt_style_lr3e-5.json`, `reports/clip_adapt_genre.json`, `reports/figures/clip_adapt_style.png`, `reports/figures/clip_adapt_style_lr3e-5.png`, `reports/figures/clip_adapt_genre.png` | Adapting CLIP: on style the frozen probe scores 52.0% `reports/clip_adapt_style.json#summary.probe.test_acc.mean` and no variant beats it by more than the noise, best frozen last-4 `reports/clip_adapt_style.json#paired_vs_probe.frozen_last4_minus_probe.mean`, fine-tuning the last block `reports/clip_adapt_style.json#paired_vs_probe.finetune_1_minus_probe.mean`; second setting `reports/clip_adapt_style_lr3e-5.json#paired_vs_probe.finetune_1_minus_probe.mean`; genre `reports/clip_adapt_genre.json#summary.finetune_2.test_acc.mean`; weights to ship `reports/clip_adapt_style.json#summary.finetune_1.shipped_mb_besides_head_fp32` | none (it builds the CLIP vectors it needs; the style probe matches `reports/style_task.json` per seed) | about 1 hour on a shared Apple M4 (this PR) |

## How much data is needed

| Script and command | Needs | Writes | Supports (README) | Run after | Time |
|---|---|---|---|---|---|
| `scripts/learning_curves.py`: CLIP probe, ResNet-18 and random forest on growing training sets, and few-shot CLIP, on the genre and the style task. `python scripts/learning_curves.py` | genre data and style data; CLIP weights (the caches `clip_embeddings.npz` are built if missing) | `reports/learning_curves.json`, `reports/figures/learning_curves.png` | How many labelled paintings are needed: the CLIP probe scores 77.3% with 25 images `reports/learning_curves.json#tasks.genre.curves.clip_probe.mean.0`, 88.2% with all of them `reports/learning_curves.json#tasks.genre.curves.clip_probe.mean.5`; it reaches 80% with about 38 images `reports/learning_curves.json#tasks.genre.curves.clip_probe.images_needed.80.images`; zero-shot gets 84.4% `reports/learning_curves.json#tasks.genre.zero_shot.mean`; 20 labelled images per class give 85.9% `reports/learning_curves.json#tasks.genre.few_shot.probe.20.mean`; style: 50.2% with all images `reports/learning_curves.json#tasks.style.curves.clip_probe.mean.5` | none | about 25 minutes (script docstring) |

## What the guide does in practice

| Script and command | Needs | Writes | Supports (README) | Run after | Time |
|---|---|---|---|---|---|
| `scripts/fetch_non_paintings.py`: downloads ordinary photos for the non-painting test. `python scripts/fetch_non_paintings.py` | internet, about 400 MB | `data/ood/` (not committed); no report | none | none | a minute or two (script) |
| `scripts/non_painting.py`: can the guide tell a painting from a wall, a face or an object? `python scripts/non_painting.py` | genre data, `data/ood/`, the committed `reports/clip_probe.npz`; reads `reports/calibration.json` for the ResNet threshold (uses a temperature of 1 if it is missing) | `reports/non_painting.json`, `reports/non_painting_rule.json`, `reports/figures/non_painting_scores.png`; `data/ood/non_painting_failures.png` (thumbnails, not committed) | What if it is not a painting?: the rule's threshold 0.071 `reports/non_painting_rule.json#threshold`; the ResNet still answers 30.7% of real non-paintings `reports/non_painting.json#answered_at_documented_thresholds.resnet.all_real_photos` | `scripts/fetch_non_paintings.py`, `scripts/calibrate_resnet.py`, `scripts/train_clip_probe.py` | a few minutes (script) |
| `scripts/benchmark_kiosk.py`: weight size, memory, latency, int8 and fp16. `python scripts/benchmark_kiosk.py` | an idle machine, genre data (for the int8 and fp16 accuracy), CLIP weights | `reports/kiosk_benchmark.json`, `reports/figures/kiosk_benchmark.png` | Kiosk budget: CLIP image encoder 20.9 ms `reports/kiosk_benchmark.json#configs.clip_fp32.threads_1.total_ms.median_of_runs`, ResNet-18 7.2 ms `reports/kiosk_benchmark.json#configs.resnet18_fp32.threads_1.total_ms.median_of_runs`, int8 accuracy 89.5% `reports/kiosk_benchmark.json#clip_precision_fp16_int8.test_accuracy.int8.mean` | `scripts/train_clip_probe.py` | about 15 minutes (README) |

## Tools to use the models (no report written)

These five scripts are not experiments, so they are not part of `make reproduce-all`. Their prerequisites:

| Script | Needs |
|---|---|
| `scripts/predict.py` | `--model resnet`: `reports/resnet18.pt` from `scripts/train_resnet.py`, and `reports/calibration.json` if present; `--model clip`: the committed `reports/clip_probe.npz`; `--reject-non-paintings`: `reports/non_painting_rule.json` and the search index |
| `scripts/guide.py` | `reports/clip_probe.npz`, the search index (built on first use, about 30 s), optionally `reports/non_painting_rule.json` |
| `scripts/search.py` | genre data and the search index; `--encoder multilingual` needs `pip install -e ".[multilingual]"` |
| `scripts/demo.py` | `pip install -e ".[demo]"`, everything `scripts/guide.py` needs |
| `scripts/evaluate_photos.py` | the team's own photos in `data/photos/<label>/` (see `docs/REAL_PHOTO_PROTOCOL.md`), `reports/clip_probe.npz`; writes only to `data/photo_eval/`, never to `reports/`; `--demo` needs the genre data |
