# Changelog

What was done, by theme. Each entry gives the pull request number and the title it was merged under; the numbers quoted come from the
[README](README.md) and the files in `reports/` (see [docs/PROVENANCE.md](docs/PROVENANCE.md)). There are no version tags yet.

## Foundations: direct commits by the three authors (27 October to 5 November 2026)

- Project structure, data script and packaging (Lou Brunet); artist-disjoint train/validation/test split (Mathis Meimoun); training loop with monitoring, evaluation and CPU latency helpers (Lou Brunet).
- Classical baseline, colour and edge features with logistic regression and random forest (Alexis Maeder).
- Feed-forward network and the diagnosis of its training (Lou Brunet); pretrained ResNet-18 fine-tuned and compared with training from scratch (Alexis Maeder).
- First results and recommendation in the README (Lou Brunet); natural-language search over the collection with CLIP (Mathis Meimoun).
- The 10-minute presentation, redesigned and shortened (Alexis Maeder, Mathis Meimoun), and corrections to the README claims on the split, latency and model size (Mathis Meimoun).

## Measuring how solid the results are

- #3 Robustness over 10 seeds, with confidence intervals and a shift test on the ResNet: the ResNet scores 74.0% ± 3.8.
- #8 Data audit: near-duplicates, split composition, and the accuracy a model would get from metadata alone.
- #6 Per-class metrics and an error analysis for the four models.
- #12 An opt-in option to keep frozen BatchNorm layers in evaluation mode (off by default, so no earlier number changes).
- #7 CLIP zero-shot and CLIP with a linear head against the ResNet: 90.2% against 74.0% on the genre task.
- #15 A 6-style task: CLIP with a linear head 55.4%, ResNet 40.6%.
- #21 Eight frozen image encoders compared on the same splits: the answer is to keep CLIP ViT-B/32.
- #31 Adapting CLIP (richer features, partial fine-tuning, LoRA): nothing beats the frozen linear head.
- #32 Learning curves and few-shot behaviour: how many labelled paintings the guide needs.

## The guide

- #2 A script that classifies photos of paintings.
- #9 Temperature calibration and an "unsure" answer.
- #16 Short notices on each genre and style, shown with `--story` (drafts written by an AI, to be reviewed).
- #18 CLIP in the guide: a saved calibrated head, `predict --model clip`, and a `guide` command that combines recognition, the story and search.
- #19 Refusing what is not a painting (`--reject-non-paintings`, off by default).
- #20 A small web demo.

## Search

- #11 A search command, a saved index, and a 66-question query set labelled from the metadata.
- #14 The same questions in French, with a multilingual text encoder (the multilingual encoder did not help; the loss came mostly from artists' spellings).
- #30 Using the catalogue fields named in a question (artist, style, genre), then CLIP.

## Cameras and kiosks

- #10 Robustness to camera-like perturbations for the ResNet, and a test of photo-style augmentation.
- #27 The same for the CLIP head: it holds up better than the ResNet.
- #13 The kiosk budget: weight size, memory and latency for the ResNet and CLIP, with int8 and fp16.
- #25 A harness and a protocol to evaluate the guide on real photos taken by the team (no real-photo result exists yet).

## The recommendation

- #17 The recommendation to the director rewritten with the measured results: a pretrained model, CLIP's image encoder with a linear head, the ResNet-18 as the fallback.

## The repository

- #1 Tests and a GitHub Actions workflow. #5 Lint, a Makefile and a lint job. #28 Type checks, coverage, and a test that starts every script.
- #4 Small fixes (install instructions, layout, data directory override).
- #22 A walkthrough notebook that reads the committed results. #34 A reorganised README with a contents list and a quick start.
- #23 A provenance map: where each number comes from, kept up to date by a test. #26 Known-good versions and a Dockerfile.
- #24 Contributing rules, a citation file, a model card, a data statement and repository templates.
- #29 WikiArt paintings are no longer published in figures: they are generated locally, and a test guards the tracked images. The three figures that showed paintings are still in the Git history.

## Open points

- No licence has been chosen; it is for the three authors to decide.
- The story notices and the French translations of the queries were written by an AI and need a human review.
- The presentation (`presentation/museum-guide.pptx`) still carries the first recommendation and shows some paintings on slides 1 and 9.
- No real-photo evaluation has been run yet.
