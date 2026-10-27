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

- [ ] A feed-forward network in PyTorch for artwork classification
- [ ] Training monitoring and failure diagnosis
- [ ] A pretrained CNN (ResNet) for image classification
- [ ] A classical baseline and a measured recommendation (accuracy, CPU latency, cost)
- [ ] 10-minute presentation

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
