# Data statement

What the data is, where it comes from, how the subsets were built, what is known to be wrong with them, and what this repository does and does not redistribute.
Numbers come from `reports/data_audit.json` and `reports/style_task.json`, or are counted in the `metadata.csv` written when the subsets were built (marked below).

## Provenance

- **Source**: the `huggan/wikiart` dataset on the Hugging Face Hub (81,444 paintings with 27 style, 11 genre and 129 artist labels, including the catch-all "Unknown Artist"),
  at the pinned revision `d559852d2b232e0fcf195e775866964f0564f2b5` (a commit of 2023-03-22, set in `project/make_wikiart_subset.py`), so that everybody gets the same data.
- **Licence**: the dataset card says `license: unknown`. WikiArt images remain the property of their authors or rights holders. The course script that builds the subsets says to use them for coursework only,
  not to commit them to Git and not to publish them.
- **Labels**: the genre, style and artist of each painting are WikiArt's own, not checked by the authors.

## How the subsets were built

Both subsets were built with the course script `project/make_wikiart_subset.py`, which streams the dataset, picks 300 images per class with a fixed seed (0), caps the number of images per named painter
at 50 (`--max-per-artist`, default), resizes the shorter side to 128 px, centre-crops to 128 × 128 and saves JPEGs with a `metadata.csv` and a `subset_info.json` (command, revision, counts, shards used).

| Subset | Command | Strategy | Source shards |
|---|---|---|---|
| `data/wikiart_genre5` (genre task) | `python project/make_wikiart_subset.py --label genre --classes portrait landscape cityscape still_life religious_painting --per-class 300 --size 128 --out data/wikiart_genre5` | `fewest-shards` (default) | 5 |
| `data/wikiart_style6` (style task) | `python project/make_wikiart_subset.py --label style --classes Impressionism Realism Romanticism Expressionism Post_Impressionism Symbolism --per-class 300 --size 128 --strategy spread --out data/wikiart_style6` | `spread` | 7 |

The style subset uses `--strategy spread` (7 source shards) because the default `fewest-shards` build (3 source shards) is more confounded: the source shard alone predicts the style at 36.7% instead of 26.7%,
and the ResNet gains about 6 points on it (`reports/style_task_fewest.json` keeps that first version for comparison). Even in the `spread` build all the Impressionism images come from one shard
(largest shard share 1.0 for that class in `reports/style_task.json`). The shard counts come from each subset's `subset_info.json`.

## Composition

| | Genre subset | Style subset |
|---|---|---|
| Images | 1,500 (5 classes × 300) | 1,800 (6 classes × 300) |
| Share of "Unknown Artist" | 48% (`data_audit.json`) | 49.6% (`style_task.json`) |
| Named painters | 120 (counted in `metadata.csv`) | 79 (`style_task.json`) |

Split, for the models: `GroupShuffleSplit` by artist, 70/15/15, one split per seed. Each "Unknown Artist" image is its own group, so the artist-disjoint guarantee covers **named painters only**:
nothing in the data says whether two unknown paintings share a hand. On the genre subset the seed-0 split has 972 / 249 / 279 images (train / validation / test); across the seeds 0 to 9 the test part has 195 to 301 images,
and the share of Unknown Artist images in a part ranges from 36% to 68% (`data_audit.json`).

## Known issues

- **Near-duplicates (genre subset)**: a perceptual-hash audit (8×8 difference hash, plus a 16×16 variant and mirrored copies) found no pair within 3 bits among the 1,500 images and one pair within 6 bits (two different genres);
  its two images fall in different parts of the split for seeds 4 and 5. A perceptual hash only catches near-identical copies: a crop, another photograph of the same canvas or a colour change can slip through (`data_audit.json`).
- **Order of the source dataset**: the images come in blocks of artists. Among Unknown Artist images, consecutive rows share a genre 36 to 39% of the time against 30% for any two images of the same shard and 20% if independent;
  a model cannot see the shard, but it shows how the subsets were assembled (`data_audit.json`).
- **Shortcuts**: the style label alone predicts the genre with 45.1% ± 6.5 accuracy, the image format 30.1% ± 3.6 and the source shard 48.7% ± 5.4 (chance 20%); on the style task the genre label alone gives 26.4% ± 4.7 (chance 16.7%).
  Models that score near these figures have learnt little more than a shortcut (`data_audit.json`, `style_task.json`).
- **Resolution**: images are 128 px; models that expect 224 px (CLIP) see them upsampled. Original sizes are in `metadata.csv` (`orig_width`, `orig_height`).
- **Label noise and coverage**: WikiArt genres overlap, its style labels are debatable, and the collection is mostly Western painting. Five genres and six styles are far fewer than a museum has.
- **Not paintings**: the optional "not a painting" rule was tested on public photo sets (Caltech-101, DTD, Imagenette) fetched by `scripts/fetch_non_paintings.py` into `data/ood/`; they are not redistributed here either.

## What this repository does and does not redistribute

- **Not committed**: `data/` (the subsets, the CLIP vector caches, the search index and the non-painting photos), model weights (`*.pt`), and any screenshot or notebook output showing paintings.
- **Committed but derived from the data**: the reports (`reports/*.json`) and most figures are numbers and charts only.
- **Open point for the team — images of paintings that are tracked in the repository.** The repository is public and the dataset card says `license: unknown`.
  - `reports/figures/search_examples.png`: thumbnails of the paintings returned for five questions;
  - `reports/figures/error_confident_mistakes.png` and `reports/figures/error_hesitant_correct.png`: thumbnails of test paintings with the model's answers;
  - `presentation/museum-guide.pptx`: at least slide 1 (the title slide) and slide 9 (search examples) show paintings; slides 5 and 11 are tables and charts, and the other slides were not inspected one by one.
    The deck stores each slide as a picture.

  Removing them from the current files would not remove them from the Git history, which would need a history rewrite. This pull request does not touch them: it is a decision for the three authors.
