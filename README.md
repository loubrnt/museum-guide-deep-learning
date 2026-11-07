# Museum guide: deep learning for artwork recognition

Final project for **Neural Networks & Deep Learning Foundations** (MSA-DATB09-91, Albert School, 2026-27).

**Group 5:** Lou Brunet, Mathis Meimoun, Alexis Maeder

## Brief

A mid-sized French fine-arts museum (fictional) wants an intelligent guide for visitors' phones and gallery kiosks:

1. **Recognise** the painting the visitor points the camera at: its style or genre, and tell the story of that tradition.
2. **Answer questions** in natural language ("Which painting shows a stormy sea?") with the best-matching artworks.

The director also wants a recommendation: kiosks run on modest CPUs, the team is small and the budget limited.
Is a neural network worth it in production, or would a simpler classical model do the job?

## Deliverables

- [x] A feed-forward network in PyTorch for artwork classification (`scripts/train_mlp.py`)
- [x] Training monitoring and failure diagnosis (`reports/figures/mlp_*.png`)
- [x] A pretrained CNN (ResNet) for image classification (`scripts/train_resnet.py`)
- [x] A classical baseline and a measured recommendation (`scripts/baseline_classical.py`)
- [x] Natural-language search over the collection (`scripts/search_clip.py`)
- [x] 10-minute presentation (`presentation/museum-guide.pptx`)

## Results

Task: 5 genres (cityscape, landscape, portrait, religious painting, still life), 300 images each.
The split is **by artist** (972/249/279 images, about 65/17/19, because `GroupShuffleSplit` splits groups, not images): no named painter appears in two parts.
48% of the images are "Unknown Artist": each is its own group, so a new painter cannot be guaranteed for those.
Chance level is 20% overall (classes are balanced in the full set, not inside each part).
Latency = median time for one image on **one thread** of a development machine, a stand-in for a modest kiosk. Classical times include image loading and features; network times cover the forward pass only, so compare orders of magnitude.

| Model | Test accuracy | CPU latency | Parameters |
|---|---|---|---|
| Logistic regression on colour/edge features | 50.2% | 1.5 ms | 126 features |
| Random forest on colour/edge features | 53.8% | 12.0 ms | 126 features |
| Feed-forward network (64x64 pixels, 512-128 hidden, dropout) | 41.9% | 0.5 ms | 6.4 M |
| ResNet-18 trained from scratch | 46.2% (val) | - | 11.2 M |
| **ResNet-18 pretrained on ImageNet, layer4 + head fine-tuned** | **74.6%** | **11.5 ms** | 11.2 M |

### Feed-forward network: monitoring and diagnosis

- **Sanity check:** it memorises 64 images (100% train accuracy, loss 0), so the code and gradients work.
- **Learning rate:** at 1e-1 the loss explodes in the first epoch, then stays stuck at ln(5) = 1.61 (best val 21.7%); 1e-2 is unstable; 1e-3 to 1e-5 all plateau around 40-41% val.
- **Overfitting:** without regularisation train accuracy hits 100% while validation stays at 42%. Dropout 0.5 + weight decay narrows the gap (71% train / 43% val); augmentation brings no extra gain.
- **Failure diagnosis:** an MLP ties every weight to one pixel position. A circular shift of the test images by 8 pixels changes **63%** of its predictions (accuracy 41.9% -> 30.8%). A painting framed differently could be misread. Convolutions share weights across positions, which should help; the same test on the ResNet is in [How solid are these numbers?](#how-solid-are-these-numbers) below.

### Per-class results and errors

`scripts/error_analysis.py` retrains the four models on the split of the table (seed 0; it reproduces 50.2 / 53.8 / 41.9 / 74.6%) and reports more than an accuracy (`reports/error_analysis.json`).
Macro-F1 is 0.491 (logistic regression), 0.529 (random forest), 0.411 (feed-forward network) and **0.746** (ResNet-18), so no model hides behind one easy class. ResNet-18 per genre on the 279 test images:

| Genre | Precision | Recall | F1 | Images |
|---|---|---|---|---|
| cityscape | 77.6% | 72.2% | 74.8% | 72 |
| landscape | 64.7% | 66.0% | 65.3% | 50 |
| portrait | 78.3% | 80.0% | 79.1% | 45 |
| religious painting | 78.6% | 69.8% | 73.9% | 63 |
| still life | 72.9% | 87.8% | 79.6% | 49 |

- **Mistakes are between neighbouring genres.** The 12 most confident mistakes (`reports/figures/error_confident_mistakes.png`) are all one of two confusions: landscape <-> cityscape (7 images: villages, churches and squares inside a landscape) and portrait <-> religious painting (5 images: a praying friar, a bearded man in a robe, ladies in ruffs, all Baroque except one Impressionist). Many of these labels are arguable, since a genre label is a single choice for a picture that can be both.
- **The ResNet is sure of itself when wrong.** Those 12 mistakes have 98-100% confidence; the mean confidence is 93% on correct predictions and 77% on mistakes.
- **Hesitant but correct** (`reports/figures/error_hesitant_correct.png`): the least confident correct predictions (37-65%) include several drawings and strongly stylised paintings.
- **No sign that "Unknown Artist" images are easier or harder**: ResNet accuracy is 75.5% on them (102 images) and 74.0% on named artists (177 images).
- **By style**, the ResNet is best on Realism (90%, 31 images) and Post-Impressionism (83%, 59) and weakest on Northern Renaissance (54%, 13) and Impressionism (65%, 46). With 10 to 59 images per style, differences of this size are within noise, so this is a hint to look at with more data, not a finding.

Limits: one split and one training run, and genre labels that overlap (a religious scene in a landscape).

### Pretrained CNN

Trained from scratch, ResNet-18 overfits 972 images (46.2% val). With ImageNet weights, training only the new head gives 68.7% val
with 2,565 trainable parameters; also fine-tuning `layer4` gives 73.5% val and 74.6% test.
The jump comes from pretraining: from scratch, the same network does not beat the random forest on validation (46.2% vs 47.4%).

### How solid are these numbers?

The table above comes from one split and one training run, on a 279-image test set. `scripts/robustness.py` repeats the
comparison for **10 seeds** (a seed changes the artist-disjoint split *and* the training randomness; the test set has 195 to 301 images)
and computes bootstrap intervals that resample **artists**, not single images (`reports/robustness.json`, `reports/figures/robustness.png`).
Seed 0 is the split of the table and reproduces it exactly.

| Model | Test accuracy, 10 seeds (mean ± std) | Seed 0, 95% interval |
|---|---|---|
| Logistic regression | 49.0% ± 3.7 | 50.2% [44.7, 55.6] |
| Random forest | 53.9% ± 5.2 | 53.8% [48.2, 59.3] |
| Feed-forward network | 40.9% ± 6.0 | 41.9% [35.5, 48.2] |
| **ResNet-18, layer4 + head fine-tuned** | **74.0% ± 3.8** | **74.6% [69.0, 80.3]** |

- **The ResNet gain is real.** It beats the random forest and the logistic regression on all 10 seeds. On seed 0 the paired gap
  to the random forest is +20.8 points, 95% interval [+14.5, +26.9].
- **Random forest vs logistic regression is a tie.** On seed 0 the gap is +3.6 points, interval [-2.1, +9.9], and the forest wins on 8 seeds out of 10.
  The README ranking between the two classical models should not be over-read.
- **The feed-forward network is worse than the forest on all 10 seeds** (seed 0: the forest is ahead by 11.8 points, interval [+5.9, +18.1]).
- **Shift test on the ResNet** (the circular shift of the MLP test, as a fraction of the image width so 64 and 128 px compare):

| Shift (fraction of the width) | MLP: accuracy, predictions changed | ResNet-18: accuracy, predictions changed |
|---|---|---|
| 6.25% (4 px / 8 px) | 40.9% -> 36.3%, 44.5% | 74.0% -> 70.9%, 16.3% |
| 12.5% (8 px / 16 px) | 40.9% -> 29.3%, 60.9% | 74.0% -> 69.0%, 19.7% |

  Convolutions help a lot (about 5 points lost instead of 12, and 20% of predictions changed instead of 61%), but the ResNet is
  **not** invariant to framing: one prediction in five changes under a 12.5% shift.

Limits: the seed-to-seed spread mixes split and training noise; the shift is circular (it creates a seam), not a real re-framing; there are still no real camera photos.

### CLIP as a recognition model

The CLIP model used for search below can also recognise the genre, with no network to train: `scripts/clip_classifier.py` compares it with
the fine-tuned ResNet-18 on the same 10 seeds and splits as `scripts/robustness.py` (`reports/clip_classifier.json`, `reports/figures/clip_classifier.png`).

- **Zero-shot**: the image vector is compared with one sentence per class, no training at all. Two wordings were fixed in advance and both are reported (none was tuned on the test set).
- **Linear probe**: a logistic regression on the frozen image vectors, trained on the training part; its regularisation is chosen on the validation part.

| Model | Test accuracy, 10 seeds (mean ± std) | Macro-F1 | Seed 0, 95% interval | Seeds won against the ResNet |
|---|---|---|---|---|
| CLIP zero-shot, "a *class*" | 76.0% ± 4.2 | 74.6% | 74.2% [66.3, 81.7] | 6 / 10 |
| CLIP zero-shot, "a painting of a *class*" | 85.1% ± 2.4 | 84.7% | 84.2% [79.4, 88.8] | 10 / 10 |
| **CLIP linear probe** | **90.2% ± 2.5** | **89.9%** | **89.6% [85.8, 93.5]** | **10 / 10** |
| ResNet-18, layer4 + head fine-tuned | 74.0% ± 3.8 | | 74.6% [69.0, 80.3] | |

On seed 0 the paired difference with the ResNet (artist-level bootstrap, ResNet retrained and reproducing the 74.6% above) is **+15.1 points for the linear probe,
interval [+9.5, +21.0]**, and +9.7 points for the "a painting of a" zero-shot, interval [+4.4, +14.9]. With the plainer wording the zero-shot is no better than the ResNet
(-0.4 points, interval [-7.4, +6.9]): the wording of the class sentence matters by 9 points.

Cost: the image encoder alone has 87.8 M parameters (351 MB in 32-bit floats; the full CLIP checkpoint, with the text encoder, is 605 MB), against 11.2 M parameters (45 MB) for the ResNet.
Its forward pass is several times slower on CPU (about 33-41 ms on one thread in two runs, both made while other jobs were running on the laptop: indicative only, a proper benchmark is still to do).
On the other hand one CLIP model serves recognition **and** natural-language search, and the 1,500 image vectors are computed once.

Limits: same 5 broad, visually distinct genres and 1,500 images; the seed-to-seed spread mixes split and training noise; CLIP was trained on web images that may include some of these paintings,
so its numbers could be flattering; and no real camera photos yet. The recommendation to the director below was written before this comparison and should be revisited.

## Natural-language search

A pretrained CLIP dual encoder (`openai/clip-vit-base-patch32`) maps paintings and questions into the same space.
The 1,500 image vectors are computed once; a question costs one text encoding plus a dot product: **14 ms on one CPU core**.

- **Genre questions** ("I would like to see portraits of people", "paintings showing saints, angels or scenes from the Bible"…):
  precision@10 = **1.00** for all five genres. This test is easy: the genres are broad and visually distinct.
- **Free questions** (`reports/figures/search_examples.png`): "portraits of children", "a snowy winter village",
  "the crucifixion of Christ" and "a vase of flowers" return 5/5 relevant paintings. "Which painting shows a stormy sea?"
  is the weak case: dramatic skies and boats, but only one or two real seascapes in the top 5, because the
  subset holds few marine paintings (no "marina" genre was downloaded).
- Limit: no hand-labelled question set, so free-question quality is judged by eye. A labelled set of ~50 museum
  questions would be the next step.

## Recommendation to the director

Use the **fine-tuned ResNet-18** for recognition and **CLIP** for search, both run locally. It is 21 points more accurate than the best classical model, at the same order of CPU cost
(about 12 ms for the forward pass on one thread, no GPU, no API fee). The weights are 45 MB for the ResNet and 605 MB for the full CLIP checkpoint, so kiosk memory must be checked.
Over 10 seeds the ResNet stays at 74.0% ± 3.8 and wins every time; its predictions are not invariant to framing (20% change under a 12.5% shift).
Still to measure: real camera photos.
The feed-forward network is not worth deploying: it is less accurate than the classical baseline and fragile to framing.

## Reproduce

From the repository root, with the [data](#data) built:

```bash
pip install -e .
python scripts/baseline_classical.py
python scripts/train_mlp.py
python scripts/train_resnet.py
python scripts/search_clip.py
python scripts/robustness.py --seeds 10    # about 10 minutes: seeds, confidence intervals, shift test
python scripts/clip_classifier.py --seeds 10    # CLIP zero-shot and linear probe vs the ResNet (needs reports/robustness.json)
```

Each script writes its numbers to `reports/<name>.json` and its figures to `reports/figures/`.

### Classify a photo

Once `train_resnet.py` has written `reports/resnet18.pt` (not versioned), any photo can be classified on CPU:

```bash
python scripts/predict.py photo.jpg                 # genre and confidence, top 3
python scripts/predict.py a.jpg b.jpg --top-k 2 --json
```

Photos are resized and centre-cropped to 128 px exactly as at evaluation time, and phone EXIF rotation is applied.
The code is in `src/museum_guide/predict.py` (`load_classifier`, `classify`) for use in a kiosk or app.

## Data

A subset of [WikiArt](https://huggingface.co/datasets/huggan/wikiart), built with the course script:

```bash
pip install -e .
python project/make_wikiart_subset.py --label genre \
    --classes portrait landscape cityscape still_life religious_painting \
    --per-class 300 --size 128 --out data/wikiart_genre5
```

`data/` is not versioned; the command above rebuilds it. The scripts look for it in `data/wikiart_genre5` relative to the working
directory; set `MUSEUM_GUIDE_DATA=/path/to/wikiart_genre5` to put it elsewhere.

### Sanity checks on the data

`python scripts/data_audit.py` (about 10 seconds, writes `reports/data_audit.json`) runs the checks below.

- **No near-duplicate images.** Perceptual hashes over the 1,500 images find no pair within 3 bits (of 64) and one pair within 6 bits
  (two different genres, never across a split for seed 0). At the loosest setting (9 bits, plain or mirrored) there are 7 and 10 pairs;
  I looked at all of them: different paintings with a similar layout or tone. A 16x16 hash finds none up to 38 bits of 256.
- **No named painter in two parts**, for each of the 10 seeds (checked by an assertion). The split sizes move with the seed:
  test 195 to 301 images, 36% to 56% "Unknown Artist" images in the test part (seed 0: 279 images, 37%; train 52%, val 44%).
  Class balance also varies: the gap between the most and least frequent class is 7 to 29 points of the test part.
- **Neighbouring unknown paintings look alike a little more than chance, not clearly more than their shard does.** The dataset is ordered in
  blocks of artists, so consecutive "Unknown Artist" rows might share a hand. They share a genre in 36% to 39% of the cases (any
  two unknown images of one shard: 30%; chance: 20%), and the share does not grow as the rows get closer. This is a small effect on a few
  hundred pairs: it cannot show that unknown paintings leak between train and test, nor that they do not.

How far can a model get without looking at a single pixel? A random forest on metadata only (test accuracy, mean ± std over 10 seeds):

| Information used | Genre accuracy |
|---|---|
| Chance (5 classes) | 20% |
| Most frequent training class | 15.4% ± 4.7 |
| Original shape of the painting (aspect ratio, size) | 30.1% ± 3.6 |
| Style label (23 styles) | 45.1% ± 6.5 |
| Style + shape | 46.5% ± 4.7 |
| Source block in WikiArt (shard) | 48.7% ± 5.4 |

The style and the shard are not available to a camera, so these are floors, not deployable baselines. They show that the style alone
predicts the genre of nearly half of the paintings: the feed-forward network (40.9%) and the logistic regression (49.0%) of the results above are at that
level, the random forest (53.9%) is a few points above it, and the ResNet (74.0%) is far above.

Limits: a perceptual hash only catches near-identical copies (not a crop, a different photograph of the same canvas, or a colour-only
change); the audit cannot tell whether two "Unknown Artist" paintings share a hand, and they are 48% of the data; the shard
is a trace of how the subset was built, not something a model sees.

## Repository layout

```
project/       course data script (make_wikiart_subset.py)
src/           reusable code (data and split, models, training loop, inference, robustness helpers)
scripts/       one script per experiment (baseline, MLP, ResNet, CLIP search, robustness) and predict.py
tests/         pytest suite on synthetic data
reports/       figures and results (JSON)
presentation/  the 10-minute slides
notebooks/     placeholder, empty for now
data/          the WikiArt subset (not versioned, rebuilt with the command above)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use synthetic data (no download, no GPU, a few seconds). They check the artist-disjoint split,
the models, and the training loop (best-epoch restoration, early stopping). GitHub Actions runs them on every pull request.

`make test` runs the tests and `make lint` runs [ruff](https://docs.astral.sh/ruff/) (errors, unused names, import order, likely bugs; configured in `pyproject.toml`).
GitHub Actions runs both. The `Makefile` also has `install`, `data` (rebuilds the dataset), `reproduce` (every experiment of this README, in order) and `clean` (caches only).
