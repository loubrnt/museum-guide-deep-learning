# Testing the guide on real photos: a protocol for the team

**Status: no real photo has been tested yet.** Every accuracy in the README comes from clean, cropped WikiArt scans (128 px), or
from perturbed copies of them. A visitor's phone photo has a frame, a wall, glare, a slant and a hand shake. This page tells the
three of us how to build a small set of real photos and how to turn it into numbers with `scripts/evaluate_photos.py`.

## 1. What we want to learn

1. Does the guide still get the genre right on real photos? (accuracy, and how much the confidence threshold costs in coverage)
2. Does it say "not sure" when it should, and does it refuse a wall, a door or a bench? (`--reject-non-paintings`)
3. Which conditions hurt most: distance, angle, light, glare, frames, screens?

## 2. How many photos

| What | At least | Why |
|---|---|---|
| Different paintings per genre | **10** | Shots of one painting are not independent; the intervals count paintings, not photos |
| Photos per genre (5 genres) | **30** (so 3 to 5 shots per painting) | About 150 photos of 50 paintings: enough to see a large gap (20 points or more, like the 21-point loss of the ResNet under a 15 degree tilt), not a small one |
| Non-paintings | **30** | To measure how often the guide answers a wall or an object |

With 50 paintings the 95% interval on an accuracy near 90% is still about 18 points wide (80% to 98% in a simulation of 50 paintings with 3 shots each; 22 points at 75%), and wider per genre: say so when reporting.

## 3. What to photograph

**Paintings**: originals in a museum (check the photo policy, section 6), printed reproductions (art books, posters, postcards) and
paintings shown on a screen. Cover the five genres: cityscape, landscape, portrait, religious painting, still life. Decide each
painting's genre *before* shooting, from its subject, and write it in a shared sheet (see section 9). If two of us disagree, put it in a folder
called `ambiguous/`: the script ignores folders that are not labels, and says so.

**Shots of the same painting** (3 to 5, vary one thing at a time):

| Shot | What changes |
|---|---|
| Frontal, painting fills the frame | the "easy" case, close to the dataset |
| Seen from 2 to 3 metres, frame and wall visible | what a kiosk camera or a visitor really takes |
| Oblique, about 30 degrees | perspective |
| Close-up on the centre | cropped composition |
| Dim light, or with glare / a spotlight | lighting (not every painting offers it) |

Also vary the phone orientation (portrait / landscape) and keep one slightly blurred shot from time to time: it happens.

**Non-paintings** (`not_a_painting/`): walls, doors, floors, benches, ceilings, information panels, sculptures, objects, a gallery
without a painting in view, a photo of a hand or of the lens cover. **No faces of visitors.** Add a few hard ones: a poster with
text, a black-and-white photograph. Note that the current rule also refuses drawings and etchings, which a museum would not want
(see "What if it is not a painting?" in the README): if you photograph some, put them in `ambiguous/` and report them separately.

## 4. Folders and file names

```
my_photos/
  cityscape/         city01__frontal.jpg  city01__far.jpg  city01__oblique.jpg  city02__frontal.jpg ...
  landscape/
  portrait/
  religious_painting/
  still_life/
  not_a_painting/    wall01__a.jpg  door__a.jpg ...
  ambiguous/         (ignored by the script)
```

- The folder name is the true label (`Religious painting` and `religious-painting` are accepted too).
- **Name each file `<paintingid>__<shot>.jpg`**. Photos with the same id are one painting in the confidence intervals. Without this convention every
  photo counts as a different painting and the intervals come out too narrow.
- Any phone format works (JPEG, PNG; iPhone HEIC needs `pip install pillow-heif`). Keep the original orientation: the EXIF rotation is applied.
- Do not resize or edit the photos. Large ones are read at reduced size (`--max-side 1024`), which is faster and changes nothing for the models.

## 5. Running it

```bash
python scripts/evaluate_photos.py --demo                                        # try it first: perturbed dataset images, NOT real photos
python scripts/evaluate_photos.py my_photos/                                    # CLIP head, with its saved confidence threshold
python scripts/evaluate_photos.py my_photos/ --reject-non-paintings             # also refuse walls and objects
python scripts/evaluate_photos.py my_photos/ --model resnet                     # the fine-tuned ResNet-18 (needs reports/resnet18.pt)
python scripts/evaluate_photos.py my_photos/ --always-answer                    # no "unsure": the best guess for every photo
```

It writes, into `data/photo_eval/` (not versioned): `predictions.csv` (one row per photo: file, true label, guess, probability, status),
`summary.json` and `confusion_matrix.png`. File paths in these files are relative to `my_photos/`.
`--reject-non-paintings` needs CLIP and the collection index (`python scripts/search.py "a snowy village"` builds it once). The first CLIP run downloads about 600 MB.

## 6. Privacy and rights

- **Never commit photos** (the script writes under `data/`, which Git ignores; do not copy them anywhere else in the repository). Do not publish them.
- Check the museum's photography policy before shooting: most allow personal photos without flash, few allow sharing. Some paintings are still under copyright.
- No identifiable visitors in the frame. Delete such shots rather than hiding the face.

## 7. Reading the numbers

The reference to compare with is the saved CLIP head on the clean test set of the dataset (seed 0, 279 paintings, `reports/clip_probe.json`):
**89.6%** when it answers every photo; with the saved threshold (0.83) it answers **78.5%** of the photos and is right on **96.3%** of those.

| Number in the output | Read it as | Compare with |
|---|---|---|
| accuracy of the best guess on every photo | what the genre guess is worth when forced to answer | 89.6% |
| answered / unsure | how often the guide dares to answer | 78.5% answered |
| accuracy on the photos it answers | how reliable an answer is when given | 96.3% |
| macro-F1, per-class recall, confusion matrix | which genres fail, and into which | look at the figure |
| wrongly refused (with `--reject-non-paintings`) | paintings refused as "not a painting" | should be small |
| false-accept rate of non-paintings | walls and objects the guide would still answer | should be small |

Then **open `predictions.csv`, sort by confidence and look at the photos behind the confident mistakes**: a photo can be mislabelled, or genuinely ambiguous (a religious scene in a landscape).
A drop of a few points is expected; a drop of 20 points or more means the guide needs work on real conditions (augmentation, a different threshold) before it goes near a kiosk.
Report the interval, not only the point: with this many paintings it is wide.

## 8. Reporting the results

Add a small table to the README (section "Robustness to camera conditions"), with: number of photos and of paintings per genre, who shot them, the phone(s), where
(museum / reproductions / screens), the date, the command used, and the numbers above with their 95% intervals. Copy them from `summary.json`; do not retype them.
Commit `summary.json` only if it contains nothing you would not show publicly (it records no photo path, but check). Never commit `predictions.csv` if file names reveal a person or a place you should not name.

## 9. Who does what

| Task | Who | Done |
|---|---|---|
| Agree the genre of each painting, fill the shared sheet (id, genre, where, medium) | | [ ] |
| Photograph paintings: cityscape and landscape (3 to 5 shots each, at least 10 paintings per genre) | | [ ] |
| Photograph paintings: portrait, religious painting, still life | | [ ] |
| Photograph 30 or more non-paintings (no faces) | | [ ] |
| Name and sort the files as in section 4 | | [ ] |
| Run `--demo`, then the real set, with and without `--reject-non-paintings` | | [ ] |
| Look at the confident mistakes in `predictions.csv` and the confusion matrix | | [ ] |
| Write the results table in the README | | [ ] |
| Decide what to change if the numbers are bad (threshold, augmentation, rule) | all | [ ] |
