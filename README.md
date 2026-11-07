# Museum guide: deep learning for artwork recognition

Final project for **Neural Networks & Deep Learning Foundations** (MSA-DATB09-91, Albert School, 2026-27).

**Group 5:** Lou Brunet, Mathis Meimoun, Alexis Maeder

**New here?** Start with [`notebooks/museum_guide_walkthrough.ipynb`](notebooks/museum_guide_walkthrough.ipynb): the whole story in charts, computed from the committed results (no data or model needed).

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

- **Mistakes are between neighbouring genres.** The 12 most confident mistakes (a gallery that `scripts/error_analysis.py` generates locally, see [Figures that show paintings](#figures-that-show-paintings)) are all one of two confusions: landscape <-> cityscape (7 images: villages, churches and squares inside a landscape) and portrait <-> religious painting (5 images: a praying friar, a bearded man in a robe, ladies in ruffs, all Baroque except one Impressionist). Many of these labels are arguable, since a genre label is a single choice for a picture that can be both.
- **The ResNet is sure of itself when wrong.** Those 12 mistakes have 98-100% confidence; the mean confidence is 93% on correct predictions and 77% on mistakes.
- **Hesitant but correct** (the second local gallery): the least confident correct predictions (37-65%) include several drawings and strongly stylised paintings.
- **No sign that "Unknown Artist" images are easier or harder**: ResNet accuracy is 75.5% on them (102 images) and 74.0% on named artists (177 images).
- **By style**, the ResNet is best on Realism (90%, 31 images) and Post-Impressionism (83%, 59) and weakest on Northern Renaissance (54%, 13) and Impressionism (65%, 46). With 10 to 59 images per style, differences of this size are within noise, so this is a hint to look at with more data, not a finding.

Limits: one split and one training run, and genre labels that overlap (a religious scene in a landscape).

### Pretrained CNN

Trained from scratch, ResNet-18 overfits 972 images (46.2% val). With ImageNet weights, training only the new head gives 68.7% val
with 2,565 trainable parameters; also fine-tuning `layer4` gives 73.5% val and 74.6% test.
The jump comes from pretraining: from scratch, the same network does not beat the random forest on validation (46.2% vs 47.4%).

### Frozen BatchNorm layers

`pretrained_resnet18` freezes the backbone weights, but the training loop puts the model in train mode, so the frozen BatchNorm
layers still update their running statistics and normalise each batch with its own statistics instead of the ImageNet ones.
`pretrained_resnet18(..., bn_eval=True)` keeps them in eval mode. The default is unchanged: **every number in this README was produced without it**.
`scripts/bn_study.py` compares both over 10 seeds (`reports/bn_study.json`, `reports/figures/bn_study.png`); the default at seed 0 reproduces 71.3% and 74.6%.

| Variant | Default (mean ± std) | `bn_eval=True` | Paired difference | Seeds improved |
|---|---|---|---|---|
| Head only | 69.6% ± 3.6 | 73.9% ± 3.5 | +4.3 points (s.e. 0.5) | 10/10 |
| `layer4` + head (the recommended model) | 74.0% ± 3.8 | 75.2% ± 4.0 | +1.2 points (s.e. 0.9) | 6/10 |

With the head only the gain is clear and consistent. For the recommended `layer4` model it is within the noise: the layers being trained
re-estimate their own statistics anyway. Making `bn_eval=True` the default would change the head-only numbers and require
regenerating the figures and tables above (a possible follow-up); for the recommended model it would be about +1 point on average.

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

### Recognising the style, not only the genre

The brief asks for a painting's "style or genre"; everything above is the 5-genre task. `scripts/style_task.py` runs the same comparison on **6 styles**
(Impressionism, Post-Impressionism, Realism, Romanticism, Expressionism, Symbolism: the six most frequent in WikiArt), 300 images each, with the same
artist-disjoint split and the same 10 seeds. The subset has 1800 images, 79 named painters and 49.6% "Unknown Artist" images;
the test parts have 199 to 313 images. It is built with `--strategy spread` (7 source files, about 3.3 GB), which mixes the sources more than the default
(see the last bullet):

```bash
python project/make_wikiart_subset.py --label style \
    --classes Impressionism Realism Romanticism Expressionism Post_Impressionism Symbolism \
    --per-class 300 --size 128 --strategy spread --out data/wikiart_style6
python scripts/style_task.py          # about 5 minutes; writes reports/style_task.json and two figures
```

| Model | Test accuracy, 10 seeds | Macro-F1 | Seed 0, 95% interval |
|---|---|---|---|
| Logistic regression, colour/edge features | 32.1 ± 2.2% | 31.5 | 34.2 [29.3, 39.3] |
| Random forest, colour/edge features | 32.4 ± 2.2% | 31.3 | 33.1 [28.2, 38.4] |
| ResNet-18 pretrained, `layer4` + head | 40.6 ± 5.4% | 39.7 | 40.5 [32.6, 51.4] |
| CLIP zero-shot, "a painting in the style of ..." | 37.5 ± 3.5% | 35.2 | 37.7 [32.4, 45.1] |
| CLIP zero-shot, "a ... painting" | 37.1 ± 5.2% | 31.8 | 42.0 [34.6, 48.6] |
| **CLIP + linear probe** | **55.4 ± 5.2%** | **54.2** | **52.5 [44.8, 62.7]** |

Chance is 16.7%; always answering the most common style scores 26.5% on average. Metadata floors (a random forest that sees no pixel): the painting's genre label
26.4 ± 4.7%, the source file 26.7 ± 5.9%, the original shape 21.3 ± 2.7%, genre and shape 28.3 ± 3.1%: about the majority class, so no shortcut explains the results.

- **Style is much harder than genre.** The best model reaches 55.4%, against 90.2% on the 5 genres.
- **CLIP + linear probe is the best model on all 10 seeds** (10/10 against the ResNet; on seed 0 the paired gap is +12.1 points, interval [+4.0, +20.4]).
  The ResNet beats the random forest on 10/10 seeds and the logistic regression on 9/10, but on seed 0 alone the gaps are not clear
  (+7.4 points, interval [-0.3, +18.1] and +6.2 points, interval [-2.1, +18.3]). CLIP with no training at all (37.5%) beats the ResNet on only 2 seeds out of 10.
- **The mistakes are neighbouring movements.** Pooled over the 10 seeds, the commonest are Post-Impressionism predicted as Impressionism (24% of its images for the ResNet,
  25% for the CLIP probe) or as Expressionism (21%, 22%), Realism and Romanticism mixed up in both directions
  (ResNet: 25% and 25%), and Expressionism and Symbolism (21% and 18% for the ResNet). On seed 0 the recall of Post-Impressionism is only
  22.5% for the ResNet and 39.4% for the CLIP probe (`reports/figures/style_task_confusion.png`).
- **Unknown Artist images are easier, not harder.** Pooled over the seeds (1331 unknown and 1367 named test images), the ResNet scores 44.9% on unknown artists and 36.1% on named ones; the CLIP probe
  59.7% and 50.5%. Named painters never cross the split, so their accuracy is the cleaner estimate of how the guide would do on a painter it has never seen:
  **50.5% for the CLIP probe and 36.1% for the ResNet.** The gap is consistent with unknown paintings sharing a hand with training images, but this does not prove it.
- **How the subset is built matters.** The default `--strategy fewest-shards` (3 source files, 1.4 GB, 74.3% unknown artists) gives different numbers (`reports/style_task_fewest.json`):
  ResNet 46.4 ± 2.6% instead of 40.6%, CLIP probe 55.9 ± 3.4% instead of 55.4%, and the source file alone predicts the style at 36.7 ± 5.3% instead of 26.7%.
  The ResNet gains about 6 points on the more confounded build, so use `spread` for style experiments.

Limits: 6 styles, 1,800 images of 128 px; the test parts are small and the seed-to-seed spread mixes split and training noise; WikiArt style labels are themselves debatable (Impressionism and
Post-Impressionism are neighbours, and one painter can belong to two movements); style is strongly tied to the painter, so generalising to a painter the model has never seen is the hard part and the
artist-disjoint split only protects named painters; CLIP was trained on web images that may include these paintings; no timing is given here.

### Other image encoders

Is another frozen encoder a better trade-off than CLIP ViT-B/32? `scripts/encoder_comparison.py` trains the same linear probe (logistic regression on
L2-normalised vectors, C chosen on validation among 0.1 to 1000, the same grid for every encoder) on the same 10 artist-disjoint splits, on the genre
task (1500 images) and the style task (1800 images, previous section), and measures the image tower's cost (`reports/encoder_comparison.json`).
Accuracy is test accuracy, mean ± std over the 10 seeds; "vs B/32" is the number of seeds won and the mean difference in points.

| Encoder | Genre probe | vs B/32 (seeds won, points) | Style probe | vs B/32 | Zero-shot genre / style | Parameters | Weights | One image, 1 thread |
|---|---|---|---|---|---|---|---|---|
| CLIP ViT-B/32 (current) | 90.2 ± 2.5 | - | 55.4 ± 5.2 | - | 85.1 / 37.5 | 87.8 M | 351 MB | 20.3 ms |
| CLIP ViT-B/16 | 91.2 ± 2.5 | 7/10, +1.0 | 54.9 ± 5.3 | 4/10, -0.5 | 86.8 / 40.6 | 86.2 M | 345 MB | 56.7 ms |
| CLIP ViT-L/14 | 92.0 ± 2.3 | 7/10, +1.8 | 55.4 ± 5.1 | 4/10, 0.0 | 89.9 / 43.8 | 304.0 M | 1216 MB | 236.1 ms |
| SigLIP base/16 | 90.8 ± 2.4 | 6/10, +0.6 | 54.1 ± 3.2 | 3/10, -1.3 | 89.0 / 36.1 | 92.9 M | 372 MB | 78.7 ms |
| DINOv2 small | 86.5 ± 1.9 | 1/10, -3.6 | 49.7 ± 5.5 | 0/10, -5.7 | - | 22.1 M | 88 MB | 25.1 ms |
| DINOv2 base | 89.5 ± 1.2 | 2/10, -0.7 | 50.6 ± 6.0 | 1/10, -4.8 | - | 86.6 M | 346 MB | 70.9 ms |
| ResNet-50, frozen | 81.2 ± 4.8 | 0/10, -8.9 | 43.0 ± 4.9 | 0/10, -12.3 | - | 23.5 M | 94 MB | 24.4 ms |
| ResNet-18, frozen | 75.6 ± 3.8 | 0/10, -14.5 | 40.5 ± 4.2 | 0/10, -14.9 | - | 11.2 M | 45 MB | 6.2 ms |

Cost: image tower only (no image reading or resizing), random weights of the right architecture, one image, one thread, Apple M4, median of 3 fresh processes
(the three agree within 5%; load average 2.6 when the run started). CLIP B/32 gives 20.3 ms here, against 20.9 ms with image reading and resizing in the
[kiosk budget](#kiosk-budget-measured). Weights are the fp32 file.

**The answer is: keep CLIP ViT-B/32.**

- **A bigger CLIP buys little.** On genre, B/16 gains 1.0 point (7 seeds out of 10, 2.8 times slower) and L/14 gains 1.8 points (7 out of 10, 11.6 times slower,
  3.5 times the weights: 1.2 GB). With the spread between seeds at about 2.5 points, 7 wins out of 10 is not distinguishable from chance (sign test, p about 0.34).
  On style there is no gain at all: L/14 scores 55.4%, the same as B/32.
- **SigLIP does not help the probe** (90.8% genre, 54.1% style) and is 3.9 times slower. Its zero-shot genre accuracy is higher (89.0% against 85.1%), but its zero-shot
  style accuracy is not (36.1% against 37.5%).
- **Zero-shot, the large CLIP comes close to the trained head on genre** (L/14: 89.9%, against 90.2% for the B/32 probe), without any labelled image, at 11.6 times the latency.
  For style zero-shot stays far below the probe (43.8% at best, against 55.4%).
- **DINOv2 is worse than CLIP here**, on genre (86.5% small, 89.5% base) and clearly on style (49.7% and 50.6%), and the base model is 3.5 times slower than CLIP B/32.
  The small one has a quarter of the weights (88 MB) but loses 3.6 points on genre and 5.7 on style.
- **Frozen ImageNet ResNets are far behind** (ResNet-50: 81.2% genre, 43.0% style). A frozen ResNet-18 with a linear probe (75.6 ± 3.8% on genre) is about as good as the fine-tuned one
  of the first results (74.0 ± 3.8%), without any training of the network.

Limits: 10 seeds on 1500 and 1800 images, so differences of about 2 points or less are within the noise. The images are 128 px and are upsampled to each encoder's input size
(224 px, 128 px for the ResNet-18): no encoder sees more detail than 128 px contain, which may be why larger models do not gain more; this was not tested. Linear probes only: no
fine-tuning of the encoders. One machine. The DINOv2 convention (CLS token concatenated with the mean of the patch tokens) was fixed before the runs, and a different one
could change its numbers. The regularisation grid here is wider than in the CLIP section; CLIP B/32 gives the same 90.2% and 55.4%. Running the script downloads about 3.7 GB of
weights; the SigLIP tokenizer needs `pip install sentencepiece protobuf`, without them SigLIP is reported as skipped.

```bash
python scripts/encoder_comparison.py                   # everything: embeddings, probes, timings
python scripts/encoder_comparison.py --stage latency   # only the timings (leave the machine idle)
```

## Natural-language search

A pretrained CLIP dual encoder (`openai/clip-vit-base-patch32`) maps paintings and questions into the same space.
The 1,500 image vectors are computed once; a question costs one text encoding plus a dot product: **14 ms on one CPU core**.

- **Genre questions** ("I would like to see portraits of people", "paintings showing saints, angels or scenes from the Bible"…):
  precision@10 = **1.00** for all five genres. This test is easy: the genres are broad and visually distinct.
- **Free questions** (the figure `search_examples.png`, generated locally by `scripts/search_clip.py`): "portraits of children", "a snowy winter village",
  "the crucifixion of Christ" and "a vase of flowers" return 5/5 relevant paintings. "Which painting shows a stormy sea?"
  is the weak case: dramatic skies and boats, but only one or two real seascapes in the top 5, because the
  subset holds few marine paintings (no "marina" genre was downloaded).
- Limit: no hand-labelled question set, so free-question quality is judged by eye. A labelled set of ~50 museum
  questions would be the next step.

### Searching, and a measurable query set

```bash
python scripts/search.py "a snowy winter village" --k 5 --grid result.png
```

The image vectors are saved once in `data/wikiart_genre5/search_index.npz` (not versioned); a question then costs only one text encoding.
`--rebuild` recomputes the index. The code is in `src/museum_guide/search.py` (`build_index`, `search`, and the metrics).

`scripts/search_eval.py` adds a query set **generated from the WikiArt metadata**, so "relevant" is objective and reproducible
(`reports/search_queries.json`, 66 queries, each with at least 10 relevant paintings): the 5 genre questions above,
39 style x genre questions ("a Cubist still life", "an Impressionist landscape"; relevant = the style group and the genre both match)
and 22 artist questions ("paintings by Claude Monet"; relevant = the artist field matches). Precision@10 (P@10) and average precision (AP)
are given next to a random ranking, which scores the share of relevant paintings:

| Query type (queries) | P@10 | AP | nDCG@10 | Random ranking: P@10, AP |
|---|---|---|---|---|
| Genre (5) | 1.00 | 0.87 | 1.00 | 0.19, 0.20 |
| Style x genre (39) | 0.27 | 0.23 | 0.28 | 0.02, 0.03 |
| Artist (22) | 0.40 | 0.38 | 0.45 | 0.01, 0.02 |
| **All (66)** | **0.37** | **0.33** | **0.39** | **0.03, 0.04** |

Writing "a painting: " before each question gives the same picture (all queries: P@10 0.38, AP 0.32), so the phrasing is not what limits the scores.

- Far above chance on every query type: AP is more than twice the random level on 36 of the 39 style x genre queries and on all 22 artist queries.
  Still, the exact top 10 is often wrong: P@10 is at least 0.5 on only 8 of the 39 style x genre queries and 8 of the 22 artist queries,
  and 7 style x genre queries and 1 artist query have no relevant painting in the top 10 ("a Symbolist landscape", "a Romantic religious painting",
  "paintings by Mstislav Dobuzhinsky").
- Style and artist are harder than genre: a style label is a historical category that looks like its neighbours (Impressionism and Post-Impressionism,
  the Renaissance labels), and the model must know a painter's name. A missed result is not always a visual error.

Limits: relevance comes from WikiArt metadata, so this measures knowledge of genre, style and artist, **not** fine content such as "a stormy sea",
which stays judged by eye (the local figure `search_examples.png`). Paintings by "Unknown Artist" can never be relevant to an artist question.
The collection has 1,500 paintings and the scores depend on it; the two phrasings were fixed before looking at the results.

### Questions in French

The museum is French but CLIP's text encoder was trained mostly on English. `scripts/search_french.py` asks the same 66 questions in French
(`museum_guide.french`: same ids, same relevant paintings, only the wording changes; `reports/search_queries_fr.json`) and compares two text encoders on
the same image index: CLIP's own, and `sentence-transformers/clip-ViT-B-32-multilingual-v1`, a multilingual student encoder whose image side is the same CLIP.
Each cell is P@10 / AP:

| Text encoder and language | All (66) | Genre (5) | Style x genre (39) | Artist (22) |
|---|---|---|---|---|
| CLIP text encoder, questions in English | 0.37 / 0.33 | 1.00 / 0.87 | 0.27 / 0.23 | 0.40 / 0.38 |
| CLIP text encoder, questions in French | 0.30 / 0.26 | 0.98 / 0.82 | 0.19 / 0.16 | 0.35 / 0.31 |
| CLIP, French without accents | 0.32 / 0.27 | 0.98 / 0.81 | 0.19 / 0.16 | 0.39 / 0.36 |
| CLIP, French, artists spelled as in the metadata | 0.33 / 0.29 | 0.98 / 0.82 | 0.19 / 0.16 | 0.42 / 0.39 |
| Multilingual encoder, English | 0.24 / 0.20 | 1.00 / 0.86 | 0.23 / 0.19 | 0.09 / 0.07 |
| Multilingual encoder, French | 0.22 / 0.18 | 1.00 / 0.84 | 0.18 / 0.15 | 0.10 / 0.08 |
| Random ranking | 0.03 / 0.04 | 0.19 / 0.20 | 0.02 / 0.03 | 0.01 / 0.02 |

- **French costs CLIP about 0.06 of average precision**, and it is a real loss, not noise (-0.064, 95% interval [-0.098, -0.036]; worse on 52 of 66 questions),
  but it stays far above a random ranking. Genre questions barely move (P@10 0.98); style x genre questions lose -0.062 (95% interval [-0.090, -0.036]) and artist questions -0.070 ([-0.155, -0.001]).
- **For artists, the cause is the spelling, not the language.** With the French spelling of the names ("Édouard Cortès", "Ilia Machkov", "Constantin Korovine") P@10 falls to
  0.35; with the spelling of the metadata ("Edouard Cortes", "Ilya Mashkov", "Konstantin Korovin") it is 0.42, as good as in English (0.40).
  For one painter the effect is huge: AP for Édouard Cortès is 0.80 in English, 0.01 with the French spelling and 0.75 once the accents are stripped. Stripping accents helps the artist questions
  (+0.043, 95% interval [+0.001, +0.113]) but not clearly overall (+0.011, 95% interval [-0.005, +0.038]).
  A French application should map the names people type to the spelling CLIP knows before encoding them.
- **The multilingual encoder does not help here.** On the French questions it is worse than CLIP's own encoder (-0.086, 95% interval [-0.135, -0.042]). The whole loss is on artist names
  (-0.234, 95% interval [-0.340, -0.134]; 12 of the 22 artist questions have no relevant painting in the top 10, against 2 for CLIP).
  On genre (+0.027, 95% interval [-0.019, +0.068], 5 questions: too few to conclude) and style x genre (-0.016, 95% interval [-0.051, +0.013]) there is no clear difference.
  It does keep the answers more stable between languages: for the four free questions that have an answer in the collection the top 10 shares 9/10, 7/10, 9/10, 10/10 of its paintings between English and French, against 7/10, 6/10, 6/10, 8/10 for CLIP.
- **The free questions** ("Quel tableau montre une mer agitée ?", "des portraits d'enfants", "un village enneigé en hiver", "la crucifixion du Christ", "un vase de fleurs") are judged by eye, as in English: with CLIP's encoder
  the share of the expected genre in the top 10 stays between 0.9 and 1.0 in French. The stormy-sea question, already the weak case in English, shares only 1 of 10 paintings between the two languages with either encoder.

So for French, **keep CLIP's own text encoder** and normalise artist names; the multilingual encoder stays an optional extra
(`pip install -e ".[multilingual]"`, then `python scripts/search.py "un village enneigé en hiver" --encoder multilingual`).
A translation step from French to English was not tested; the English row is what a perfect one could reach at best.

Limits: **the French texts were written by an AI assistant and must be read by a French speaker** (`reports/search_queries_fr.json`; the French spellings of Russian names are the least certain).
Relevance still comes from the metadata, so this measures genre, style and artist, not fine content. Only one multilingual encoder was tested. With 5 genre questions the by-type intervals for genre say little.

## Recommendation to the director

**Is a neural network worth it? Yes, provided it is a pretrained one.** A network trained from scratch on a few hundred images is not
(the feed-forward network is worse than the classical baseline), but a pretrained model gives the guide a clear lead, with no GPU and no API fee.
All figures below are test accuracy on the artist-disjoint split, mean over 10 seeds, unless stated otherwise (sections above give the details).

| Approach | Genre (5 classes) | Style (6 classes) | Cost on the kiosk |
|---|---|---|---|
| Logistic regression, colour/edge features | 49.0% ± 3.7 | 32.1% ± 2.2 | about 1.5 ms (first table) |
| Random forest, colour/edge features | 53.9% ± 5.2 | 32.4% ± 2.2 | about 12 ms (first table) |
| Feed-forward network from scratch | 40.9% ± 6.0 | not run | 0.5 ms (first table) |
| ResNet-18 pretrained, fine-tuned | 74.0% ± 3.8 | 40.6% ± 5.4 | 45 MB, 7.2 ms, +120 MB memory |
| **CLIP image encoder + linear head** | **90.2% ± 2.5** | **55.4% ± 5.2** | 351 MB, 20.9 ms, +729 MB memory |

Chance is 20% for the genre task and 16.7% for the style task. Kiosk costs are from the [kiosk budget](#kiosk-budget-measured) (Apple M4, one thread, image
read and resize included); the classical and feed-forward latencies come from the first results table, measured earlier on another machine, so compare orders of magnitude only.

**What I would deploy**

1. **Recognition: CLIP's image encoder with a linear head**, if the kiosk has about 2 GB of memory and a CPU not far below a recent laptop. It beats the fine-tuned
   ResNet on all 10 seeds for genre (paired gap on seed 0: +15.1 points, interval [+9.5, +21.0]) and for style, at about 3 times the time and 6 times the added memory,
   which is still a fraction of a second. It also serves the search below, so the kiosk loads one model, not two. If the CPU or memory is very constrained,
   the **fine-tuned ResNet-18** is the fallback (74.0%, 45 MB).
2. **Search: CLIP, run locally.** Genre questions are solved (precision@10 = 1.00); style and artist questions are far above chance but often imprecise (precision@10 of 0.27
   for style x genre and 0.40 for artists, against 0.02 and 0.01 by chance). It was trained mostly on English: French questions lose some accuracy (average precision 0.327 to 0.263 on
   66 queries), mostly through artists' spellings; a multilingual encoder did not help.
3. **Do not hide the uncertainty.** The ResNet is over-confident (mean confidence 88.8% for 74.6% accuracy). A temperature fitted on validation data halves the calibration error
   (0.151 to 0.077), and answering only when the top probability is at least 0.71 gives 86.0% accuracy on 64% of the photos (threshold chosen on validation data, measured on test);
   below it the kiosk can say "not sure". These figures are for the ResNet and one split; the same should be done for the CLIP head before deployment.
4. **Show the story, with human review.** The short notices on each genre and style are drafts written by an AI: they must be read and corrected by the team before visitors see them.

**What the classical model can and cannot do.** It is cheap, but at about 50% genre accuracy it is little better than guessing from the painting's *style label* alone (45%,
see the [sanity checks](#sanity-checks-on-the-data)). It would only make sense as a fallback when no pretrained model can run.

**Not deployed.** The feed-forward network: less accurate than the classical baseline and fragile to framing (a circular shift changes 61% of its predictions).

**Still open before a real deployment**
- **Real camera photos.** Nothing here uses them. A 15 degree rotation or a blur costs the ResNet about 20 points (72.3% to 51.7% and 54.6%, 5 seeds); photo-style augmentation recovers part
  of it (rotation back to 62.7%) at a cost of about 2 points on clean images, see [robustness to camera conditions](#robustness-to-camera-conditions). The CLIP head loses much less: 89.0% on clean images, 82.5% averaged over the same nine perturbations, but still 78.5% when tilted and 72.7% when blurred ([details](#clip-under-camera-like-conditions)).
- **New painters.** Style is tied to the painter, and the split only guarantees unseen *named* painters. On those, the style accuracy drops to 50.5% for CLIP and 36.1% for the ResNet.
- **The real kiosk CPU.** The latencies are from a laptop. int8 quantisation made CLIP slower on Apple silicon (97.9 ms against 20.9 ms); whether it helps on an x86 kiosk was not measured.
  Decoding a 12-megapixel phone photo costs 29 ms (15 ms with draft decoding), more than the ResNet itself.
- **Label quality.** WikiArt genres overlap (a religious scene in a landscape) and its style labels are debatable; 5 genres and 6 styles are far fewer than a museum collection.

## Reproduce

From the repository root, with the [data](#data) built:

```bash
pip install -e .
python scripts/baseline_classical.py
python scripts/train_mlp.py
python scripts/train_resnet.py
python scripts/search_clip.py
python scripts/robustness.py --seeds 10    # about 10 minutes: seeds, confidence intervals, shift test
python scripts/benchmark_kiosk.py                # about 15 minutes: leave the machine idle, it measures time and memory
python scripts/clip_classifier.py --seeds 10    # CLIP zero-shot and linear probe vs the ResNet (needs reports/robustness.json)
python scripts/train_clip_probe.py              # refit and calibrate the CLIP head saved in reports/clip_probe.npz (seconds)
```

Each script writes its numbers to `reports/<name>.json` and its figures to `reports/figures/`.

### Where each number comes from

[docs/PROVENANCE.md](docs/PROVENANCE.md) lists, for each of the 23 scripts, its exact command, the data it needs, the reports and figures it writes,
the README claim it supports (as a `file.json#key` you can look up), its run time when known, and which scripts must run before it.
`make reproduce-all` rebuilds every report in dependency order (`make install-reproduce` first; more than 10 GB of downloads; total time not measured),
and `make reproduce-genre`, `reproduce-robustness`, `reproduce-search`, `reproduce-clip`, `reproduce-ood`, `reproduce-style`, `reproduce-encoders` and
`reproduce-kiosk` run one group at a time. The older `make reproduce` is unchanged. `make check-provenance` (also part of the tests) fails when the map and the repository disagree.

### Classify a photo

Once `train_resnet.py` has written `reports/resnet18.pt` (not versioned), any photo can be classified on CPU:

```bash
python scripts/predict.py photo.jpg                 # genre and confidence, top 3
python scripts/predict.py a.jpg b.jpg --top-k 2 --json
```

Photos are resized and centre-cropped to 128 px exactly as at evaluation time, and phone EXIF rotation is applied.
The code is in `src/museum_guide/predict.py` (`load_classifier`, `classify`) for use in a kiosk or app.

**Calibrated confidence and "I am not sure".** The raw network is over-confident: on the test set it claims 88.8% confidence on average for 74.6% accuracy
(expected calibration error 0.151). `python scripts/calibrate_resnet.py` fits one temperature T on the **validation** set (T = 2.05, 249 images) and divides the logits by it.
The predicted class does not change; on the test set the calibration error falls to 0.077, the log-loss from 0.925 to 0.697, and the mean confidence to 76.2%
(`reports/calibration.json`, `reports/figures/calibration.png`). `predict.py` reads `reports/calibration.json` automatically, and `--min-confidence` turns low confidence into an answer of "unsure":

```bash
python scripts/predict.py photo.jpg --min-confidence 0.7
# photo.jpg: unsure (best guess: cityscape 55%)
```

Answering only the photos the network is surest about raises the accuracy of the answers (test set, calibrated confidence):

| Share of photos answered | 100% | 90% | 80% | 70% | 50% |
|---|---|---|---|---|---|
| Accuracy of the answers | 74.6% | 79.0% | 83.0% | 85.7% | 92.1% |

A threshold chosen on the validation set carries over to the test set: 0.71 (aimed at 85%) answers 64% of the test photos with 86.0% accuracy, and 0.77 (aimed at 90%) answers 54% with 90.7%.
These figures come from one split (seed 0) and 279 test photos, so they are rough: the shape (fewer answers, more accurate ones) is the finding, not the second decimal.
With `--json`, `--min-confidence` changes each entry to `{"unsure": bool, "predictions": [...]}`; without it the output is unchanged.

### The story of the tradition

The brief asks the guide to recognise a painting "and tell the story of that tradition". `src/museum_guide/notices.py` holds a short notice
(2 to 4 sentences, 40 to 57 words) for each of the 5 genres and for the 19 styles that appear at least 15 times in the subset, in French (the museum's language) and in English.
`--story` adds the notice of the predicted genre after the usual answer (French by default, `--lang en` for English); without `--story` nothing changes:

```bash
python scripts/predict.py data/wikiart_genre5/portrait/s33_r0513.jpg --story
# data/wikiart_genre5/portrait/s33_r0513.jpg: portrait 86%, religious_painting 7%, ...   (the usual line, a test-set image)
#   Le portrait représente une personne précise, pour garder son souvenir ou affirmer son rang. À la Renaissance, en Italie comme en Flandre, [...]
python scripts/predict.py photo.jpg --story --lang en                      # "A portrait shows a particular person, to preserve their memory or to assert their rank. [...]"
python scripts/search.py "a Cubist still life" --story    # the notice of the style of the top result
```

With `--json`, `--story` changes each entry to `{"predictions": [...], "story": {"label", "lang", "title", "text"}}`
(plus `"unsure"` when `--min-confidence` is given); a guess the guide is "unsure" about gets `"story": null` and no text, since it should not tell the story of a tradition it is not sure about.
From Python: `notice_for("Post-Impressionism", "en")`, `title_for(...)`, `story_entry(...)`, `normalise_label(...)` (WikiArt spellings such as `Post_Impressionism` and human ones such as `Religious painting` both work).

**These texts are drafts written by an AI assistant.** They stick to widely known general facts (where and when, what the artists were after, what to look for, a few famous names),
but nobody has checked them against the literature. They must be read by the team, and ideally by someone trained in art history, before visitors see them.

Covered labels: the genres `cityscape`, `landscape`, `portrait`, `religious_painting`, `still_life`; the styles Impressionism, Post-Impressionism, Realism, Expressionism, Romanticism, Baroque,
Northern Renaissance, Cubism, Synthetic Cubism, Early Renaissance, High Renaissance, Mannerism (Late Renaissance), Rococo, Symbolism, Art Nouveau, Naive Art (Primitivism), Fauvism, Pop Art and Ukiyo-e.
Four rarer styles of the subset have no notice yet (Pointillism, Contemporary Realism, New Realism, Analytical Cubism: 26 images out of 1,500); `search.py --story` says so on the error output instead of guessing.
The notices tell the story of a tradition, not of the painting in the photo: the guide does not identify the individual work or its painter.


### One model for recognition and search

CLIP's image encoder already serves the [search](#natural-language-search), and a linear head on its image vectors recognises the genre better than the
ResNet (90.2% against 74.0% over 10 seeds, see [CLIP as a recognition model](#clip-as-a-recognition-model)). So the guide can load **one** model for both.
`scripts/train_clip_probe.py` fits the head on the README split (seed 0) and saves it with its calibration in `reports/clip_probe.npz` (12.8 KB, versioned, so nothing needs refitting);
`scripts/predict.py --model clip` and `scripts/guide.py` use it.

```bash
python scripts/predict.py photo.jpg --model clip --min-confidence 0.83 --story   # same flags as the ResNet
python scripts/guide.py photo.jpg --ask "a snowy winter village" --k 3 --lang en
```

`guide.py` is the whole guide for one photo: the genre (top 3, "unsure" below the saved confidence threshold), the story of that tradition (French by default, `--lang en`), the paintings of the
collection closest to the photo (never the photo itself) and, with `--ask`, the paintings that best answer a question. CLIP is loaded once; the 1,500 collection vectors are computed once and saved
(`data/search_index.npz`), so a photo costs one image encoding and a few dot products. A real run on a test-set painting (a Pissarro cityscape), and on one the guide is not sure about:

```
$ python scripts/guide.py data/wikiart_genre5/cityscape/s11_r0265.jpg --ask "a snowy winter village" --k 3 --lang en
Genre: cityscape 86% (then landscape 14%, still_life 0%)
  The cityscape shows streets, squares and monuments. In the 18th century Canaletto made views of Venice famous; [...]
Similar paintings:
   1  0.844  cityscape/s11_r0638.jpg  [cityscape, Post_Impressionism, pyotr-konchalovsky]
   2  0.840  cityscape/s07_r0053.jpg  [cityscape, Post_Impressionism, vincent-van-gogh]
   3  0.830  cityscape/s11_r0678.jpg  [cityscape, Art_Nouveau, boris-kustodiev]
Paintings for "a snowy winter village":
   1  0.318  cityscape/s38_r0237.jpg  [cityscape, Expressionism, mstislav-dobuzhinsky]
   [...]

$ python scripts/guide.py data/wikiart_genre5/cityscape/s11_r0669.jpg --k 3 --lang en
Genre: unsure (best guess: cityscape 56%)          # no story for a guess the guide doubts
Similar paintings:
   1  0.844  religious_painting/s33_r0912.jpg  [religious_painting, Early_Renaissance, Unknown Artist]
   [...]
```

**Calibration of the CLIP head** (fitted on 972 training images, C = 1 chosen on 249 validation images; 89.6% on the 279 test images, the seed-0 value in `reports/clip_classifier.json`).
The temperature is fitted on the validation set only, with the method of the ResNet calibration above, and the test set is used once. The result is the **opposite** of the ResNet:
the head is *under*-confident (mean confidence 71.8% for 89.6% accuracy), so the temperature is below 1 (T = 0.44; the ResNet needed 2.05).

| Test set (279 images) | Raw probabilities | After temperature scaling |
|---|---|---|
| Mean confidence | 71.8% | 90.3% |
| Calibration error (ECE) | 0.178 | 0.033 |
| Negative log-likelihood | 0.443 | 0.301 |

Answering only the most confident photos keeps accuracy high: 94.0% on the 90% most confident, 96.0% on 80%, 96.4% on 70%, 98.6% on 50% (89.6% on all).
The threshold saved in the file, 0.83, was chosen on the validation set as the smallest confidence that reaches 95% accuracy there (95.1% on 73.5% of the validation photos). On the test set it
answers 219 of the 279 photos (78.5%) with 96.3% accuracy (8 errors). The 60 photos it declines are 39 correct and 21 wrong guesses: the rule removes 21 of the 29 errors, at the price of
dropping 39 correct answers. It is not a guarantee: 5 test photos are wrong with 95% confidence or more (for instance a religious scene by Murillo read as a portrait at 99.7%).

Limits: the head is trained on 972 images and calibrated on 249, both from one split, with no confidence intervals on these figures; the calibration was measured on the same painting collection
as the training data, not on camera photos; the first call downloads CLIP's weights (about 600 MB); "similar paintings" are neighbours in CLIP's space (they look or read alike), not curatorial links; and
the cost of the CLIP encoder on a kiosk is in the [kiosk budget](#kiosk-budget-measured).

### What if it is not a painting?

A visitor may point the phone at a wall, a face or an object. The classifiers only know five genres, so they answer anyway. `scripts/non_painting.py`
measures this and compares ways to refuse such photos (about 400 MB of ordinary photos are downloaded by `scripts/fetch_non_paintings.py`, kept in `data/ood/`, never committed):
Caltech-101 objects and faces, Imagenette (objects, animals, a church), DTD (textures of surfaces), and synthetic frames (a plain wall, plaster, a black frame, noise). Each source has 300
photos for the test and 150 others for validation; they are resized and cropped to 128 x 128 JPEG like the paintings, so that only the content differs. The paintings are the seed-0
test (279) and validation (249) paintings. Everything is in `reports/non_painting.json`.

**What the guide does today.** Share of each group that is still *answered* (not "unsure") at the documented thresholds (ResNet 0.71, CLIP head 0.83):

| Photo | ResNet (0.71) | CLIP head (0.83) |
|---|---|---|
| Paintings (test) | 64.2% | 78.5% |
| Caltech objects / faces | 37.0% / 35.3% | 12.0% / **90.0%** |
| Imagenette / textures (DTD) | 34.0% / 16.3% | 15.3% / 20.7% |
| Plain wall / plaster / black frame / noise | 0% / 0% / 0% / 0% | 10.0% / 0% / 0% / 0% |

The confidence thresholds were chosen to keep accuracy high among paintings, not to detect non-paintings: about a third of real non-painting photos still get an answer (30.7% for the ResNet, 34.5% for the CLIP head),
and the CLIP head answers for 9 faces out of 10 and calls every one of those 270 a portrait (the objects it answers are mostly called still lifes: 27 of 36). The top probability separates paintings from other photos poorly (AUROC 0.725 for the ResNet, 0.825 for the CLIP head, against 0.5 for chance).

**Ways to tell a painting from a non-painting.** AUROC (chance of ranking a test painting above a non-painting; 1 is perfect) over the 1,200 real test photos, and the share of those photos accepted
when the threshold lets 95% of the test paintings through:

| Score | AUROC | Real photos accepted at 95% of paintings |
|---|---|---|
| ResNet top probability | 0.725 | 88.8% |
| CLIP head top probability | 0.825 | 71.8% |
| CLIP zero-shot "a painting" vs "a photograph" | 0.984 | 8.3% |
| CLIP zero-shot "a painting in a museum" vs "a photo of an object, a person or a wall" | 0.986 | 2.0% |
| CLIP zero-shot "an oil painting"... vs "a photograph"... | 0.981 | 9.1% |
| CLIP zero-shot "a work of art on canvas" vs "a photo taken with a phone camera" | 0.917 | 37.9% |
| Similarity to the 5 closest paintings of the collection | 0.991 | 4.7% |
| **Zero-shot "museum" pair + similarity to the collection (fitted on validation)** | **0.996** | **0.8%** |

The four sentence pairs were written before looking at any result on the photos, and all four are shown. The zero-shot sentence pair is the best choice on the validation draw, and the combination of the
two signals (two weights and a bias, fitted by logistic regression on the validation photos only) is the best method there too, so it is the one in the guide. Everything above uses the
test photos only for reporting.

**The rule in the guide** (`reports/non_painting_rule.json`): score = 76.8 x (cos(photo, "a painting in a museum") - cos(photo, "a photo of an object, a person or a wall")) + 39.7 x (mean cosine to the
5 closest paintings of the collection) - 30.1; a photo is accepted when the score is at least 0.07. The threshold is the one that accepts 95% of the *validation* paintings: the 249 cropped ones and 100 more
framed on a wall (see the limits). On the test photos, with that threshold:

| Test photos | Accepted as paintings |
|---|---|
| Paintings, cropped to the canvas | 97.8% |
| Paintings framed on a wall (a proxy for a visitor's photo) | 82.0% |
| Caltech objects / faces | 2.7% / 0.0% |
| Imagenette / textures (DTD) | 1.0% / 9.3% |
| Plain wall / plaster / black frame / noise | 1.7% / 0% / 0% / 0% |
| All 1,200 real photos / all 240 synthetic frames | 3.3% (39 photos) / 0.4% (1 frame) |
| Heavily blurred paintings (a bad photo of a painting) | 0.0% |

The cut can be moved. Thresholds fitted on validation for other shares of paintings accepted (`python scripts/non_painting.py --tpr 0.98` saves one):

| Share of validation paintings accepted | 90% | 95% (default) | 98% | 99% |
|---|---|---|---|---|
| Test paintings accepted (cropped) | 97.1% | 97.8% | 98.2% | 99.6% |
| Test paintings framed on a wall accepted | 75% | 82% | 96% | 100% |
| Real non-painting photos accepted | 1.3% | 3.3% | 7.8% | 15.8% |
| Synthetic frames accepted | 0% | 0.4% | 13.3% | 58.8% |

**Using it.** The check is off by default and changes nothing unless asked:

```bash
python scripts/predict.py photo.jpg --model clip --reject-non-paintings
# photo.jpg: this does not look like a painting (score -9.07, threshold 0.07)
python scripts/guide.py photo.jpg --reject-non-paintings --ask "a stormy sea"   # no genre, but the question is still answered
```

It needs CLIP even with `--model resnet` (about 600 MB on first use) and the saved collection index (`data/search_index.npz`, built by the first `scripts/search.py`). With `--json`, each photo gets
`"not_a_painting"` (true or false) and `"painting_score"`; a rejected photo has no story.

**What I saw in the failures.** The non-paintings that get accepted are mostly textures that really look like paint (brush strokes, polka dots, an illuminated initial), an illustrated animal, a framed
lamp and a church under a painterly sky: arguably not errors. The paintings that get rejected are mostly drawings, etchings and black-and-white sketches (WikiArt files them under the same genres), plus one dark painted portrait: the rule treats
drawings and prints as non-paintings, which a museum would not want. I looked at the 12 highest-scoring non-paintings and the 12 lowest-scoring paintings; the grid is written to `data/ood/` and not committed.

Limits:
- The photos are ordinary datasets (objects, faces, textures, plain frames), resized to 128 px like the paintings. A real camera photo is larger and shows its surroundings; a gallery wall with several frames, a
  glare, a visitor's hand, a label or a poster of a painting are **not** covered. The "framed on a wall" images are crude compositions (a thin frame, a small rotation, a plaster background), not photos.
- A photo of a painting on a screen or a postcard is accepted by design: it is still a painting. Drawings and prints are often rejected (see above).
- One draw of 300 photos per source and one painting split; no confidence intervals. The validation and test photos come from the same four datasets, so another kind of non-painting (signs, objects in
  a showcase, a sculpture) may behave differently. Caltech-101 also holds a few drawn or sculpted items (cartoon characters, a statue of a Buddha) that are in the pool.
- The rule has three numbers fitted on 249 paintings, 100 framed compositions and 720 other photos. CLIP was trained on web images that may include some of these paintings.

### Try it in a browser

A small web page over the guide, for the presentation and for trying the kiosk idea (`scripts/demo.py`, built with Gradio):

```bash
pip install -e ".[demo]"
python scripts/demo.py                 # then open http://127.0.0.1:7860
python scripts/demo.py --lang en --port 8000
python scripts/demo.py --offline       # once CLIP's weights are cached: no network connection at all
```

Upload a photo (the widget also offers the webcam) and, if you like, type a question. The page shows a **verdict banner** (green: a painting; amber: not sure,
with the best guess; red: this does not look like a painting; grey: no photo), the **top 3 genres**, the **story** of the tradition (an AI-written draft, flagged as such on the page),
the **paintings of the collection closest to the photo** and, for a question, the **paintings that answer it**. A question is answered even when the photo is not a painting,
or when there is no photo. French is the default (`--lang en` or the switch on the page for English); switching language re-renders the last answer without computing it again.
The logic is in `src/museum_guide/demo_logic.py` and is tested without Gradio; one CLIP model is loaded at start-up and shared by all requests (one request at a time).

I checked it in a browser with the real model: a test-set portrait gives "C'est un tableau : Le portrait (94%)"; a test-set cityscape that the head is only 82% sure of (threshold 0.83) gives the amber banner and no story; a face from Caltech and a plain grey wall give the red banner;
removing the photo gives the grey one; the French question "un village enneigé en hiver" returns snowy townscapes; the language switch and `--lang en` work. The webcam input was **not** tested.

Limits:

- **A local demonstration, not a kiosk application.** The server listens on `127.0.0.1` only, Gradio's analytics are off and the page only talks to the local server, but loading CLIP contacts the Hugging Face hub
  (to download the weights, then to check the cached files) unless you pass `--offline`. `--share` (a public Gradio link) exists but is off by default and should stay off: see the next point.
- **The galleries show WikiArt paintings read from your local `data/` folder, which must not be published**: do not share the page or screenshots of it.
- The first start loads CLIP (about 600 MB downloaded the first time) and builds the collection index if `data/search_index.npz` does not exist (about 30 s).
- The story texts are drafts written by an AI assistant, pending review by the team.
- The rule that refuses non-paintings is not perfect (see the previous section): a genuine test-set landscape was refused in my checks.
- **The page cannot say "this is the same painting".** A lightly recompressed copy of a painting of the collection comes back as its own nearest neighbour (similarity close to 1), but a real photo of it generally would not: in a quick check
  (not saved in the repository), stronger compression, rescaling or lighting changes lowered the similarity to its own painting below that of the closest *different* painting, so no threshold separates the two.

### Reproducible environment

`pyproject.toml` only gives minimum versions, so a fresh install gets the latest of everything. `constraints/py311.txt` lists the versions the test suite is known to pass with
(Python 3.11, Linux, CPU-only PyTorch; `torch` and `torchvision` are minor-version ranges because their wheels differ between platforms):

```bash
pip install -c constraints/py311.txt torch torchvision --index-url https://download.pytorch.org/whl/cpu   # Linux; on macOS: pip install -c constraints/py311.txt torch torchvision
pip install -c constraints/py311.txt -e ".[dev]"
```

Or in a container (the data and the model downloads are not baked in; mount them):

```bash
make docker-build                  # docker build -t museum-guide .
make docker-test                   # runs the test suite in the image
docker run --rm -v "$PWD/data:/app/data" -v "$PWD/reports:/app/reports" \
    -v "$HOME/.cache/huggingface:/home/app/.cache/huggingface" museum-guide python scripts/baseline_classical.py
```

`make constraints` (or `bash constraints/regenerate.sh`) regenerates the file: it installs the project in a clean `python:3.11-slim` container, runs the tests there, and writes the versions only if they pass.

What was checked, on an Apple silicon Mac with Docker Desktop: the file was generated that way and the suite passed in that container (363 passed, 2 skipped: the Gradio tests, since the `demo` extra is not installed);
the image builds in about 2 minutes and is 2.5 GB (PyTorch and the dependencies are most of it); the tests pass inside it as a non-root user in about 2 minutes; with `data/` and `reports/` mounted, the 1500 images
load and the split is the usual 972/249/279. CI also runs both on Linux x86_64, and both passed on the pull request that added them: the pinned install followed by the tests (`pinned versions` workflow, when the constraints, `pyproject.toml`, `src/` or `tests/` change), and the image build followed by the tests in the container (`docker` workflow, when the recipe changes). Not checked: macOS with the pinned file, and the `multilingual` and `demo` extras (not pinned). The image is not published anywhere.

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

### Figures that show paintings

Two scripts draw galleries of WikiArt paintings: `scripts/search_clip.py` (`search_examples.png`, the answers to five free questions) and
`scripts/error_analysis.py` (`error_confident_mistakes.png` and `error_hesitant_correct.png`, 12 test paintings each). The WikiArt dataset card says
`license: unknown` and the course script asks not to publish the images, so these figures are **not committed**: the scripts write them to
`data/figures_local/` (ignored by Git), and `reports/figures/` holds charts only. The findings read from them are kept as text above and in `reports/*.json`.
To look at them, run the two scripts (see Reproduce) and open `data/figures_local/`. Set `MUSEUM_GUIDE_FIGURES_LOCAL=/some/dir` to write them elsewhere.
`tests/test_tracked_figures.py` fails when an image that is not on its allow-list (`tests/tracked_images.json`) is committed.

Earlier versions of these three figures are still in the Git history; removing them from it is a decision for the repository owner (see the pull request that introduced this rule).

## Repository layout

```
project/       course data script (make_wikiart_subset.py)
src/           reusable code (data and split, models, training loop, inference, robustness helpers)
scripts/       one script per experiment (baseline, MLP, ResNet, CLIP search, robustness) and predict.py
tests/         pytest suite on synthetic data
reports/       figures and results (JSON)
presentation/  the 10-minute slides
notebooks/     walkthrough notebook: reads reports/*.json, runs in seconds, no painting shown
data/          the WikiArt subset (not versioned, rebuilt with the command above)
```

## Robustness to camera conditions

The paintings are clean WikiArt scans, but the guide will see phone photos. There are no labelled photos yet, so `scripts/photo_robustness.py`
measures a **proxy**: nine deterministic perturbations of the test images (`src/museum_guide/perturb.py`), one effect at a time, on the
fine-tuned ResNet-18 (`layer4` + head, as in `train_resnet.py`). It also tests whether a photo-style training augmentation
(`photo_train_transform`: crop 50-100%, flip, rotation ±20°, perspective, colour jitter, blur) helps, against the augmentation of
`train_resnet.py` (crop 70-100% + flip). Same splits, seeds and hyper-parameters; **5 seeds** (a seed changes the split and the training),
results in `reports/photo_robustness.json` and `reports/figures/photo_robustness.png`.

| Condition | Standard augmentation | Photo augmentation | Difference (points) | Seeds where photo is better |
|---|---|---|---|---|
| None (clean test set) | 72.3% ± 4.0 | 70.4% ± 3.8 | -1.8 | 2/5 |
| Tilt: rotation by 15° | 51.7% ± 5.7 | 62.7% ± 3.0 | +11.0 | 4/5 |
| Oblique view (perspective) | 63.6% ± 5.3 | 66.6% ± 4.0 | +3.0 | 2/5 |
| Out of focus (Gaussian blur) | 54.6% ± 4.7 | 61.0% ± 6.6 | +6.4 | 4/5 |
| Close-up (central 60%) | 67.9% ± 4.7 | 63.5% ± 3.0 | -4.4 | 0/5 |
| Dim room (brightness ×0.5) | 70.8% ± 2.8 | 69.4% ± 4.8 | -1.5 | 2/5 |
| Spot light (brightness ×1.6) | 67.0% ± 6.0 | 66.6% ± 4.2 | -0.4 | 2/5 |
| Glare or haze (contrast ×0.5) | 70.2% ± 2.1 | 68.6% ± 5.0 | -1.6 | 2/5 |
| Heavy JPEG (quality 15) | 68.4% ± 3.8 | 65.8% ± 5.4 | -2.6 | 1/5 |
| Black-and-white | 64.8% ± 5.4 | 62.7% ± 5.8 | -2.1 | 3/5 |
| **Mean of the 9 perturbations** | 64.3% | 65.2% | +0.9 | 2/5 |

- **The ResNet is fragile to tilt and blur**: with the standard augmentation it loses 21 points under a 15° rotation and 18 points when the image is blurred, far more than under any lighting change.
- **Photo augmentation helps where it should, but is not a free win.** It recovers 11 points on the rotation (4/5 seeds) and 6 on the blur (4/5), but it costs 1.8 points on clean images, 4.4 on the close-up (0/5 seeds better) and a little on JPEG and grayscale.
- **Averaged over the nine perturbations the difference is +0.9 points, photo better on 2/5 seeds**: within the seed-to-seed noise (std 2-7 points). Only the rotation gain clearly stands out. Photo augmentation is worth it only if tilted shots matter more than clean-image accuracy; the same 20 epochs may also be too few for the harder augmentation.

Limits: these are synthetic proxies, not phone photos (which also have glare, frames, walls, hands and combined effects); each perturbation is applied alone; 5 seeds only; the augmentation strength was not tuned.

### CLIP under camera-like conditions

The recommended model is CLIP's image encoder with a linear head (see the [recommendation](#recommendation-to-the-director)), so the same nine perturbations were applied to it
(`scripts/clip_photo_robustness.py`, `reports/clip_photo_robustness.json`): same 5 seeds, same splits, same images as the ResNet study. Every image is perturbed at its 128 px size, then embedded
by the frozen encoder; the head is trained on clean vectors, as deployed. The "photo-augmented" head is trained on clean vectors plus 2 photo-augmented copies of each training image
(the encoder stays frozen). Seed 0 reproduces the saved head's test accuracy exactly (89.6%).

| Condition | ResNet-18 | ResNet-18, photo augmentation | CLIP head | CLIP head, photo-augmented |
|---|---|---|---|---|
| Clean | 72.3% ± 4.0 | 70.4% ± 3.8 | 89.0% ± 2.5 | 88.3% ± 3.5 |
| Tilt (rotation 15°) | 51.7% ± 5.7 | 62.7% ± 3.0 | 78.5% ± 1.4 | 80.8% ± 2.5 |
| Oblique view | 63.6% ± 5.3 | 66.6% ± 4.0 | 86.2% ± 1.5 | 85.5% ± 3.0 |
| Out of focus (blur) | 54.6% ± 4.7 | 61.0% ± 6.6 | 72.7% ± 2.0 | 78.5% ± 1.7 |
| Close-up (central 60%) | 67.9% ± 4.7 | 63.5% ± 3.0 | 81.5% ± 2.9 | 83.2% ± 2.2 |
| Dim room (brightness ×0.5) | 70.8% ± 2.8 | 69.4% ± 4.8 | 89.0% ± 2.1 | 88.7% ± 3.1 |
| Spot light (brightness ×1.6) | 67.0% ± 6.0 | 66.6% ± 4.2 | 84.8% ± 1.1 | 86.2% ± 1.9 |
| Glare or haze (contrast ×0.5) | 70.2% ± 2.1 | 68.6% ± 5.0 | 88.3% ± 2.5 | 87.7% ± 3.1 |
| Heavy JPEG (quality 15) | 68.4% ± 3.8 | 65.8% ± 5.4 | 78.1% ± 2.2 | 80.4% ± 2.2 |
| Black-and-white | 64.8% ± 5.4 | 62.7% ± 5.8 | 83.7% ± 2.8 | 83.9% ± 2.3 |
| **Mean of the 9 perturbations** | 64.3% ± 2.9 | 65.2% ± 3.4 | 82.5% ± 1.1 | 83.9% ± 1.8 |

- **CLIP is better on every condition, on every seed.** Over the nine perturbations the CLIP head averages 82.5% against 64.3% ± 2.9 for the ResNet
  (+18.2 points, better on 5/5 seeds); the smallest gap is on heavy JPEG (+9.6 points).
- **It has the same weak spots, much less pronounced.** From 89.0% on clean images it loses 16.3 points when blurred, 10.9 with heavy JPEG, 10.5 when tilted by 15° and 7.5 on the close-up,
  but at most 5.3 points for any lighting or colour change (the ResNet loses 18 to 21 points on tilt and blur).
- **Photo augmentation of the head helps a little, not much.** Averaged over the perturbations it gains +1.3 points (better on 4/5 seeds): +5.8 on blur (5/5 seeds),
  +2.4 on heavy JPEG (5/5), +2.3 on tilt (4/5), at -0.7 points on clean images (2/5 seeds better). The encoder is frozen, so only the head sees the augmented vectors; the gain is small compared with the seed-to-seed spread.
- **Shift test.** With the circular shift of the first sections (applied to CLIP's 224 px input), accuracy goes from 89.0% to 86.3% (6.25% of the width) and 86.5% (12.5%); 9.6% and 8.6% of the predictions change,
  against 16.3% and 19.7% for the ResNet and 44.5% and 60.9% for the feed-forward network (those two over 10 seeds; the shifts are the same fractions of the width). Better, but not invariant to framing.
- **The "unsure" answer reacts to bad photos, but does not fully protect.** With each seed's answer threshold (chosen on clean validation images for 95% accuracy), the head answers 85.7% of the clean test photos with 93.8% accuracy,
  but only 75.7% of the tilted ones (86.7% accurate when it answers) and 63.2% of the blurred ones (85.1%): it becomes less confident, so it declines more, yet on these two conditions its answers are 7 to 9 points less reliable than on clean images, short of the 95% it was set for.

Limits: synthetic proxies, not phone photos (glare, frames, walls, hands and combined effects are missing); each perturbation is applied alone; 5 seeds; one CLIP model; the augmentation strength and the number of augmented copies were not tuned;
the answer threshold and temperature are fitted on clean validation images only.

### Testing on real photos

**No result on real photos exists yet**: everything above uses clean dataset scans or synthetic perturbations of them. This repository now has what the team needs to
produce one: [`docs/REAL_PHOTO_PROTOCOL.md`](docs/REAL_PHOTO_PROTOCOL.md) says what to photograph (at least 10 paintings and 30 photos per genre, 30 non-paintings,
varied distance, angle and light), how to name the files, what the privacy rules are, and how to read the numbers; `scripts/evaluate_photos.py` turns a folder of photos
(one sub-folder per label) into the guide's own verdicts (answer, "unsure", "not a painting"), the accuracy with a 95% interval that resamples paintings rather than photos,
per-class scores, a confusion matrix and a per-photo CSV, written under `data/photo_eval/` (not versioned).

```bash
python scripts/evaluate_photos.py --demo                              # try it: perturbed dataset images, NOT real photos
python scripts/evaluate_photos.py my_photos/ --reject-non-paintings   # the real thing, with the guide's rules
```

## Kiosk budget (measured)

The recommendation below says kiosk memory "must be checked". `scripts/benchmark_kiosk.py` measures it for the ResNet-18 and for
CLIP's image encoder (the part that runs on a photo), in fp32, fp16 and int8. Each model is built and saved once, then served by
fresh processes (3 per setting, 100 different images each) so memory figures do not leak between models. Machine: Apple M4, 10 cores,
PyTorch 2.15 (`reports/kiosk_benchmark.json`). A modest kiosk CPU will be slower: trust the ratios more than the milliseconds.

| Model | Weight file | Latency per image, 1 thread (p95) | 4 threads | Memory added by the model |
|---|---|---|---|---|
| ResNet-18 fp32 | 44.8 MB | 7.2 ms (9.2) | 5.2 ms | 120 MB |
| CLIP image encoder fp32 | 351.5 MB | 20.9 ms (22.7) | 22.1 ms | 729 MB |
| CLIP fp16 | 175.8 MB | 202.5 ms (207.2) | 112.1 ms | 360 MB |
| CLIP int8 (dynamic) | 95.6 MB | 97.9 ms (103.6) | 41.4 ms | 314 MB |

Latency covers reading the JPEG, resizing and the forward pass (the forward pass is 93% of it for the ResNet and 96% for CLIP fp32). The whole process, with PyTorch,
transformers and the image libraries loaded, peaks at about 590 MB for the ResNet and 1.2 GB for CLIP fp32.

- **CLIP fp32 costs about 3 times the ResNet in time and 6 times in added memory** on this machine, for 90.2% instead of 74.0% accuracy (10 seeds).
  Both fit easily on a kiosk with 2 GB of memory; the real question is whether the CPU is fast enough.
- **int8 shrinks CLIP's file by 3.7 times and costs little accuracy, but is slower here.** On the 10 splits, a linear probe trained on fp32 vectors
  scores 90.16% ± 2.45 on fp32 vectors, 89.46% ± 2.77 on int8 vectors (-0.70 points on average, between -2.66 and +1.08) and 90.12% ± 2.48
  on fp16 vectors (-0.03). PyTorch's int8 kernels for Apple silicon (qnnpack) are 4.7 times slower than fp32 on one thread; fp16 is about 10 times slower
  because the CPU has no fast fp16 path in PyTorch. Whether int8 is faster on a kiosk's x86 CPU was **not** measured.
- **Reading a phone photo can cost more than the ResNet.** A 12-megapixel JPEG (4032x3024, 2.2 MB, synthetic) takes 29 ms to decode in full and 15 ms with
  PIL's draft decoding (`Image.draft`, decodes at a reduced scale). The project images are 128 px, so this cost never showed in the earlier latencies.

Limits: random weights (shapes, size, memory and speed do not depend on the values); one machine; the three repeats agree within about 8% (max minus min),
except CLIP fp16 on 4 threads (112 to 135 ms); the machine was not completely idle, a load average of about 6 at the start; the 12-megapixel image is a
proxy (smooth content plus noise), real photos may decode slower; the int8 parameter count in the JSON (2.4 M) excludes packed int8 weights.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use synthetic data (no download, no GPU, a few seconds). They check the artist-disjoint split,
the models, and the training loop (best-epoch restoration, early stopping). GitHub Actions runs them on every pull request.

`make test` runs the tests and `make lint` runs [ruff](https://docs.astral.sh/ruff/) (errors, unused names, import order, likely bugs; configured in `pyproject.toml`).
GitHub Actions runs both. The `Makefile` also has `install`, `data` (rebuilds the dataset), `reproduce` (every experiment of this README, in order) and `clean` (caches only).
`make coverage` reports the coverage of `src/museum_guide` and `make typecheck` runs [mypy](https://mypy.readthedocs.io/) on it (`make check` runs everything CI does); CI fails below 95% coverage, and a test starts every script with `--help`.

How to contribute: [CONTRIBUTING.md](CONTRIBUTING.md). What the models are and where they should not be used: [MODEL_CARD.md](MODEL_CARD.md). Where the data comes from and the open copyright point: [DATA_STATEMENT.md](DATA_STATEMENT.md).
