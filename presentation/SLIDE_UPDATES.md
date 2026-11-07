# Updating the presentation

`museum-guide.pptx` was made before most of the measurements in this repository. It is a deck of 11 full-slide images, so its text cannot be edited in place: the slides that change have to be rebuilt in your slide tool. This file gives the replacement text, slide by slide, with the report each number comes from. **It is a proposal for the three authors; the deck itself is untouched.**

Every figure below is a value in `reports/*.json` (the key is given), and matches the [README](../README.md). The latencies are from one laptop (Apple M4, one thread); say so on the slide.

## What to change

| Slide | Status | What to do |
|---|---|---|
| 1 Title | shows 5 painting thumbnails | WikiArt's licence is "unknown": decide whether to keep them (see [DATA_STATEMENT.md](../DATA_STATEMENT.md)). |
| 2 The brief | fine | Nothing. |
| 3 The data | fine | Optional line: "10 seeds, not one split". |
| 4 The network, 5 Monitoring | fine | If you need a slot for a new slide, merge them into one. |
| 6 Failure diagnosis | **out of date** | The orange line says the effect on the CNN is "not measured yet". It is measured: see below. |
| 7 Pretrained CNN | add one line | "74.6% is one split; over 10 seeds it is 74.0% ± 3.8." |
| 8 Accuracy against cost | **out of date** | Add CLIP and use 10 seeds: see below. |
| 9 Search | shows paintings | Replace the thumbnails by a chart: see below. |
| 10 Recommendation | **out of date** | The recommendation changed: see below. |
| 11 Limits | **out of date** | Several limits were lifted: see below. |

## Slide 6: failure diagnosis

Replace *"A convolution reuses the same weights at every position. Effect on our CNN: not measured yet."* by:

> A convolution reuses the same weights at every position, and it helps. The same shift (12.5% of the width, circular) changes **19.7%** of the ResNet-18's predictions, against **60.9%** for the feed-forward network. CLIP with a linear head: **8.6%**. A shift still costs the ResNet 5 points (74.0% to 69.0%).

Sources: in `reports/robustness.json`, section `summary.shift`, the entries `mlp` and `resnet18_layer4`, key `"0.125"` (prediction change rate 0.609 and 0.197; ResNet accuracy 0.740 to 0.690); in `reports/clip_photo_robustness.json#kiosk_behaviour.standard`, the entry `shift_12.5pct` (prediction change rate 0.086). MLP and ResNet: 10 seeds; CLIP: 5 seeds.

## Slide 8: accuracy against cost

Suggested title: **"A pretrained encoder with a linear head: 90.2% against 53.9% for the best classical model"**.

| Model | Test accuracy, 10 seeds (split by artist) | Source |
|---|---|---|
| Feed-forward network | 40.9% ± 6.0 | `reports/robustness.json#summary.test_acc.mlp` |
| Logistic regression | 49.0% ± 3.7 | `...test_acc.logistic_regression` |
| Random forest | 53.9% ± 5.2 | `...test_acc.random_forest` |
| ResNet-18, fine-tuned | 74.0% ± 3.8 | `...test_acc.resnet18_layer4` |
| **CLIP image encoder + linear head** | **90.2% ± 2.5** | `reports/clip_classifier.json#summary.test_acc.linear_probe` |

Cost column: keep only what one benchmark measured, **ResNet-18 7.2 ms and CLIP 20.9 ms per image on one core** (`reports/kiosk_benchmark.json#configs.resnet18_fp32.threads_1.total_ms.median_of_runs`, `...clip_fp32...`). The latencies of the other three models come from an earlier, separate run: either drop them or footnote them. Footer: "Rough comparison, one laptop; ratios matter more than milliseconds."

## Slide 9: search

Replace the thumbnails with numbers, which also removes the paintings from the slide:

> **66 questions labelled from the catalogue** (genre, style x genre, artist). Precision@10: **1.00** on genre questions, **0.27** on style x genre, **0.40** on artists, against 0.19, 0.02 and 0.01 by chance. In French, the average precision falls from 0.327 to 0.263, mostly through artists' spellings. Questions about fine content ("a stormy sea") are still judged by eye.

Sources: `reports/search_eval.json#summary.bare.genre.precision@10` and the other types; French figures from `reports/search_french.json` (see the README, "Questions in French").

## Slide 10: recommendation

The recommendation is no longer "the ResNet and CLIP". Suggested slide:

> **Title:** Deploy CLIP's image encoder with a linear head; keep the ResNet-18 as the fallback.
> **90.2%** against 53.9% for the best classical model (10 seeds, split by artist).
> **about 21 ms** per image on one core (ResNet-18: 7 ms), no GPU, no API fee.
> **351 MB** of weights for the image encoder plus a 13 KB head (ResNet-18: 45 MB). The old "605 MB" was the whole CLIP checkpoint, text half included.
> **It can say "not sure"**: calibrated confidence, answers only above 0.83.
> Not the feed-forward network: less accurate than the classical model, and fragile to framing.

Sources: `reports/clip_classifier.json`, `reports/kiosk_benchmark.json#configs.clip_fp32.threads_1.weights_mb` (351.5), `reports/clip_probe.json#temperature` and the README ("Classify a photo", threshold 0.83). The "2 GB of memory" condition in the README's recommendation is a judgement, not a measurement.

## Slide 11: limits

| Limit | Effect | Next step |
|---|---|---|
| No real camera photos | accuracy on visitors' photos is unknown; with synthetic perturbations CLIP keeps 82.5% on average, the ResNet 64.3% | field test: the harness and the protocol exist (`docs/REAL_PHOTO_PROTOCOL.md`) |
| 48% of images: unknown artist | a new painter is not guaranteed for those; for unseen named painters, style accuracy drops to 50.5% (CLIP) | retest on named artists only |
| 5 genres, 6 styles | far fewer than a museum collection; style is hard (CLIP 55.4%) | more genres and styles, more labelled paintings (about 20 per genre give 86%) |
| Free questions judged by eye | the 66 labelled questions measure genre, style and artist, not fine content | real visitor questions, labelled |
| Kiosk CPU not measured | latencies are from a laptop | test on the real kiosk |
| Notices and French translations | written by an AI | review by the team, ideally an art historian |

Closing line: *"The recommendation holds on this data: the gap between CLIP and the classical model is about 36 points over 10 seeds. Real photos are the next test."* (90.2% against 53.9%.)

Sources: `reports/clip_photo_robustness.json`, `reports/style_task.json#accuracy_by_artist_known.clip_probe.named_artist` (0.505), `reports/learning_curves.json` (see the README, "How many labelled paintings are needed?").

## If you can add one or two slides

The 10-minute format leaves little room (merging slides 4 and 5 frees one). In order of value: **calibration and "not sure"** (the ResNet is over-confident, 88.8% mean confidence for 74.6% accuracy; a temperature fitted on the validation set halves the calibration error, 0.151 to 0.077: `reports/calibration.json`), then **camera conditions** (a 15 degree rotation takes the ResNet from 72.3% to 51.7% and a blur to 54.6%: `reports/photo_robustness.json`).
