# Model card: the museum guide's recognition models

Every figure below is quoted from a file of `reports/` (named in the last column or in the text). Accuracy means test accuracy on an
**artist-disjoint split** (no named painter appears in two parts); "10 seeds" means ten different splits and training runs, mean ± standard deviation.
`tests/test_repo_files.py` checks the headline figures against the reports. How the data was built: [DATA_STATEMENT.md](DATA_STATEMENT.md).

## Model details

| | Proposed model | Fallback |
|---|---|---|
| What | CLIP ViT-B/32 image encoder (`openai/clip-vit-base-patch32`, frozen) + logistic-regression head | ResNet-18 pretrained on ImageNet, `layer4` and the new head fine-tuned |
| Task | genre of a painting (5 classes: cityscape, landscape, portrait, religious painting, still life); style (6 classes) is evaluated with a head trained for that task (`scripts/style_task.py`), not saved | same, genre |
| Size | image tower 87.8 M parameters, 351.5 MB in fp32 (the full CLIP checkpoint, with the text tower used for search, is 151.3 M parameters and 605.1 MB) | 11.2 M parameters, 44.8 MB |
| Input | a photo, resized and centre-cropped to 224 px by CLIP's processor; the vector is L2-normalised | resized and centre-cropped to 128 px |
| Saved in the repository | `reports/clip_probe.npz` (12.8 KB: head, temperature, "unsure" threshold) | nothing: `reports/resnet18.pt` is not committed (`*.pt` is ignored); `python scripts/train_resnet.py` writes it |
| Source | `scripts/train_clip_probe.py` | `scripts/train_resnet.py` |

The saved CLIP head was trained on the 972 images of the training part of the seed-0 split, with the regularisation `C = 1` chosen on the validation part (249 images)
(`reports/clip_probe.json`). It is not retrained on the validation or test images. Developed by the three authors of the project, November 2026; there is no version number.

## Intended use

- **For**: a demo of a museum guide for a fictional French fine-arts museum: recognise the genre (or style) of a painting from a photo, show a short notice on that
  tradition, find similar paintings, answer questions in natural language, say "not sure" or "not a painting" when appropriate. Course project and presentation.
- **Not for**: authenticating, attributing, dating or valuing artworks; naming the artist of a painting; any decision about people; any use that needs more than 5 genres or 6 styles.
  It has not been evaluated on real camera photos, so it should not be deployed to visitors without that check.

## Training and evaluation data

A subset of WikiArt (see the data statement): 1,500 images for the genre task (5 classes × 300) and 1,800 for the style task (6 classes × 300), 128 px.
Split by artist 70/15/15; each "Unknown Artist" image is its own group, so the guarantee covers named painters only.
The test set of the seed-0 split has 279 images (195 to 301 depending on the seed, `reports/data_audit.json`).

## Metrics

### Genre, 5 classes (chance 20%)

| Model | Accuracy, 10 seeds | Report |
|---|---|---|
| Logistic regression, colour/edge features | 49.0% ± 3.7 | `robustness.json` |
| Random forest, colour/edge features | 53.9% ± 5.2 | `robustness.json` |
| Feed-forward network from scratch | 40.9% ± 6.0 | `robustness.json` |
| ResNet-18, fine-tuned (fallback) | 74.0% ± 3.8 | `robustness.json` |
| CLIP zero-shot, prompt "a painting of a …" | 85.1% ± 2.4 | `clip_classifier.json` |
| **CLIP image encoder + linear head (proposed)** | **90.2% ± 2.5** | `clip_classifier.json` |

CLIP's head beats the ResNet on all 10 seeds; on seed 0 the paired gap is +15.1 points, 95% interval [+9.5, +21.0] (`clip_classifier.json`).
The saved head, on the seed-0 split: 88.0% on validation, 89.6% on the 279 test images (`clip_probe.json`).
Larger encoders were compared: CLIP B/16 91.2% ± 2.5 and L/14 92.0% ± 2.3 (`encoder_comparison.json`), at several times the cost; the proposal keeps B/32.

### Style, 6 classes (chance 16.7%, `style_task.json`)

CLIP head 55.4% ± 5.2; ResNet-18 40.6% ± 5.4; CLIP zero-shot 37.5% ± 3.5; logistic regression 32.1% ± 2.2; random forest 32.4% ± 2.2.
On paintings by named painters never seen in training: CLIP head 50.5%, ResNet-18 36.1%.

### Confidence and "unsure" (CLIP head, seed 0, `clip_probe.json`)

The raw head is under-confident: mean confidence 71.8% for 89.6% accuracy, calibration error 0.178. A temperature of 0.44 fitted on the validation set brings it to 0.033.
The saved rule answers only when the top probability is at least 0.83 (threshold chosen on validation data): on the test set it answers 78.5% of the photos with 96.3% accuracy.
(For the ResNet, which is over-confident, the temperature is 2.05 and the calibration error goes from 0.151 to 0.077, `calibration.json`.)

### Robustness to camera conditions

- **ResNet-18 (5 seeds, `photo_robustness.json`)**: 72.3% on clean images; 51.7% under a 15° rotation, 54.6% under blur, 63.6% under a perspective change.
  Photo-style training augmentation recovers part of it (rotation 62.7%, blur 61.0%) at a cost on clean images (70.4%).
  A circular shift of 12.5% of the width lowers its accuracy from 74.0% to 69.0% and changes 19.7% of its predictions (`robustness.json`).
- **CLIP head: not measured yet.** No rotation, blur, perspective or shift test exists for the proposed model.

### Photos that are not paintings (`non_painting.json`, `non_painting_rule.json`)

Without a filter, the guide answers confidently when pointed at a face: at the documented thresholds the CLIP head answers 90% of the face photos of the test draw
and 34.5% of all real non-painting photos (objects, faces, textures, scenes); the ResNet answers 30.7% of them.
An optional rule (`--reject-non-paintings`, off by default: CLIP "a painting in a museum" against "a photo of an object, a person or a wall", combined with the similarity to the collection)
reaches an AUROC of 0.991 on the validation draw; at its saved threshold on the test draw it accepts 97.8% of paintings, 3.25% of real non-painting photos and 82% of paintings framed on a wall,
and rejects heavily blurred paintings (0% accepted). Posters, drawings and engravings were not part of the non-painting sets.

### Cost on a kiosk (Apple M4, one thread, `kiosk_benchmark.json`)

| | Weights | Time per image (read, resize, forward) | Memory added |
|---|---|---|---|
| ResNet-18 fp32 | 44.8 MB | 7.2 ms | 120 MB |
| CLIP image encoder fp32 | 351.5 MB | 20.9 ms | 729 MB |
| CLIP int8 (dynamic) | 95.6 MB | 97.9 ms (slower on Apple silicon) | 314 MB |

CLIP's accuracy is 90.2% in fp32, 90.1% in fp16 and 89.5% in int8 (a head trained on fp32 vectors). A kiosk CPU will be slower than this laptop.

### Search (CLIP, `search_eval.json`, `search_french.json`)

On 66 queries built from the WikiArt metadata, precision@10 is 0.37 (0.03 by chance): 1.00 for the 5 genre queries, 0.27 for style × genre, 0.40 for artists.
French questions lose some accuracy on the English-trained encoder (average precision 0.327 to 0.263), mostly because of artists' spellings.

## Limitations and risks

- **Small label sets**: 5 genres and 6 styles, 300 images each, at 128 px. A museum collection is far larger and more ambiguous; WikiArt genres overlap (a religious scene in a landscape) and its style labels are debatable.
- **Unknown painters**: 48% of the genre images and 49.6% of the style images have no named artist (`data_audit.json`, `style_task.json`); the artist-disjoint split cannot separate them,
  and style is tied to the painter, so the style figures are optimistic for unseen painters (see the named-painter figures above).
- **Western-art bias**: WikiArt is dominated by Western painting, and the notices and styles follow its categories.
- **English-trained CLIP**: questions in French or other languages work less well; a multilingual encoder did not help (`search_french.json`).
- **No real photos yet**: all tests use scaled-down scans of paintings and synthetic perturbations; glare, frames, angles and reflections of a real gallery are not measured, and robustness is not measured for CLIP.
- **Texts awaiting review**: the short notices on each genre and style and the French translations of the search queries were written by an AI assistant and have not been read by an art historian or a native speaker.
- **Images**: the WikiArt images are "license: unknown". Some figures tracked in `reports/figures/` and the slides show paintings; see the data statement.
- **Misuse**: a confident answer is not a fact about the painting. The guide should show its uncertainty (the "unsure" rule) and must not be used to attribute works.

## Ethical considerations

The data are paintings, not people, but photos taken in a museum may contain visitors: the demo should not store uploaded photos, and the optional non-painting rule was tested on public photo sets, not on visitors.
The copyright status of the WikiArt images is unresolved (see the data statement); the repository should not redistribute them.

## How to update this card

Change a number only from a regenerated `reports/*.json` (the README's "Reproduce" section lists the scripts), then update the sentence here. Add a line to the limitations when a limit is lifted or a new one is found.
