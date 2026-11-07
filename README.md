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
- **Failure diagnosis:** an MLP ties every weight to one pixel position. A circular shift of the test images by 8 pixels changes **63%** of its predictions (accuracy 41.9% -> 30.8%). A painting framed differently could be misread. Convolutions share weights across positions, which should help, but the same test has not been run on the ResNet.

### Pretrained CNN

Trained from scratch, ResNet-18 overfits 972 images (46.2% val). With ImageNet weights, training only the new head gives 68.7% val
with 2,565 trainable parameters; also fine-tuning `layer4` gives 73.5% val and 74.6% test.
The jump comes from pretraining: from scratch, the same network does not beat the random forest on validation (46.2% vs 47.4%).

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
Still to measure: real camera photos, several seeds, and the shift test on the ResNet.
The feed-forward network is not worth deploying: it is less accurate than the classical baseline and fragile to framing.

## Reproduce

```bash
python scripts/baseline_classical.py
python scripts/train_mlp.py
python scripts/train_resnet.py
pip install transformers && python scripts/search_clip.py
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

`data/` is not versioned; the command above rebuilds it.

## Repository layout

```
project/    course data script (make_wikiart_subset.py)
src/        reusable code (models, training loop, evaluation)
notebooks/  experiments
reports/    figures and results
```
