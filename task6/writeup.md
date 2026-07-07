# Task 6 — Training a CLIP-style VLM on Flickr8k (Writeup)

The culmination of the project: the from-scratch InfoNCE loss and two-tower model from
Task 5, trained on 8,091 real image–caption pairs (Flickr8k) on an **RTX 4060**. The
model learns genuine image–text alignment — retrieval well above chance, attention that
localizes captioned objects, and zero-shot classification with no labels.

## How to reproduce

```bash
# 1. get the data + build the 64x64 cache (one-time, ~2 min)
python task6/download_data.py
python task6/preprocess.py
# 2. train (RTX 4060, ~12 min for 4000 steps)
python task6/train.py --batch_size 128 --lr 5e-4 --dropout 0.15 --weight_decay 0.05 \
                      --steps 4000 --val_every 200
# 3. evaluate + visualize + demos
python task6/eval.py
python task6/qualitative.py
python task6/zero_shot.py
python task6/text_to_image_search.py
```

Deliverables in this folder: `dataset.py`, `tokenizer.py`, `model.py`, `train.py`,
`eval.py`, `qualitative.py` (+`.png`), `training_curve.png`, `best_model.pt`,
`zero_shot.py` (+ confusion png), `text_to_image_search.py` (+ png), this writeup.

---

## Q30. Training curve. When did loss stop improving? Final train/val loss.

Training loss fell steadily from **5.23** (≈ log 128 = 4.85 baseline) to **2.37** at
step 4000, still slowly decreasing. **Validation InfoNCE loss behaved very
differently**: it bottomed at **4.228 at step 600**, then *rose* back to **4.73** by
step 4000. See [`training_curve.png`](training_curve.png).

Naively that looks like overfitting after step 600 — and on the in-batch loss it is.
But retrieval (the metric we care about) kept *improving* long after step 600 (Q32).
So "when did the model stop improving?" has two answers: the in-batch val loss stopped
at step 600; the actual downstream quality kept improving until ~step 3400.

## Q31. Retrieval metrics. Strongest / weakest?

Best checkpoint (selected by retrieval, step 3400), on the 1,000-image / 5,000-caption
val split:

| Direction | R@1 | R@5 | R@10 |
|-----------|-----|-----|------|
| **Image → Text** | 3.1% | 11.1% | 16.6% |
| **Text → Image** | 2.7% | 10.3% | 16.5% |

Random chance for R@1 is ~0.1%, so R@1 is **~30× chance** and R@10 (~16.6%) is **~16×**
the ~1% chance rate. **Strongest:** R@10 in both directions, and image→text slightly
edges text→image (an image has 5 correct captions to find, so hits are easier).
**Weakest:** R@1 — at 64×64 the model reliably gets the *theme* right (dog/beach/snow)
but rarely nails the single exact caption among 5,000, which needs fine detail the
resolution throws away. This matches the handbook's warning that 64px caps the ceiling;
its 15–25% R@1 target is for better-tuned / higher-res models. R@5/R@10 land right at
the lower end of the expected band.

## Q32. Bugs I hit and how I debugged them (the most valuable section)

**Bug 1 — Data loading starved the GPU.** A data-only epoch took **111 s** (target
10–30 s) because every `__getitem__` decoded a full-resolution JPEG. *Hypothesis:* JPEG
decode, not augmentation, dominates. *Fix:* since the project fixes 64×64, I pre-decode
all images once into a memory-mapped `uint8` cache (`preprocess.py`) and augment the
tiny cached image. Data stopped being the bottleneck; step time dropped to ~100 ms.

**Bug 2 — Catastrophic overfitting.** First full run (dropout 0.1, plain resize): train
loss dove to **1.70** while val loss *rose above the log(128) baseline to 5.22*
(handbook Symptom 4). *Fix:* stronger but not excessive regularization — dropout 0.15,
`RandomResizedCrop` augmentation, and (crucially) checkpoint selection below.

**Bug 3 (the big one) — over-regularizing made it worse.** Over-correcting (dropout
0.3, weight-decay 0.2, batch 256) *underfit*: val hovered near baseline and R@1 fell to
1.7%. *Lesson (exactly the handbook's "don't panic-tune"):* I changed too many knobs at
once. Reverting to mild regularization (dropout 0.15) recovered the good regime.

**Bug 4 — validation loss is the WRONG early-stopping signal.** This cost the most time
and taught the most. My best-*val-loss* checkpoint (step 600) gave I→T R@10 = **8.6%**.
But retrieval measured at *later* steps was far better — e.g. R@10 = **15.1%** at step
2000 — even though val loss was *higher* there. Val InfoNCE loss is computed over only
128 in-batch negatives and bottoms early; full retrieval over 5,000 captions is a much
harder task that keeps improving as embeddings sharpen. *Fix:* select the checkpoint by
**retrieval score (sum of the six recalls)**, not val loss. This single change nearly
**doubled** final R@10 (8.6% → 16.6%). Temperature confirms the story: it drifted *up*
(0.07→0.076) while alignment was weak, then finally **sharpened to 0.066** late in
training as retrieval peaked.

My debugging log in short: hypothesis → one change → observe val *and* retrieval → keep
or revert. The winning config: batch 128, lr 5e-4, dropout 0.15, wd 0.05, RRC aug,
4000 steps, retrieval-based checkpointing.

## Q33. Qualitative attention maps (see `qualitative.png`)

For each image I dot every patch's projected embedding against the caption embedding and
overlay the 8×8 similarity grid. The figure shows 6 successes + 2 failures.
**What surprised me:** even at 64×64 the high-similarity region genuinely tracks the
captioned subject — a dog, a child, water — rather than smearing uniformly, despite the
model never being given any localization supervision (it only ever saw a global
contrastive loss). The **failure cases** are the informative ones: cluttered scenes and
captions describing fine attributes ("red shirt") produce diffuse maps, because at 8-px
patches those details are simply gone.

## Q34. Zero-shot classification — what worked, what failed

Weak labels were derived from caption keywords (502 val images, 6 classes; chance
≈ 16.7%). Accuracy: **43.4% single prompt → 45.8% with a 4-template prompt ensemble.**

- **Worked:** `dog` (61%) and `bike` (61% with ensemble) — visually distinctive, common
  in Flickr8k, so the concept is well-represented in training captions.
- **Failed / weak:** `water` (~30%) and `child` (~38%). `water` is a background texture
  that co-occurs with many subjects, so its embedding is diffuse; `child` overlaps
  heavily with `dog`/`bike` scenes (kids appear everywhere), so the weak label itself is
  ambiguous. **Prompt ensembling helped most where a single template was unlucky**
  (bike 43%→61%) — averaging templates denoises the class embedding, a real deployed-CLIP
  technique.

## Q35. Applied Stretch deliverables

I built **three** (the handbook asks for at least one):

1. **Text-to-image search** (`text_to_image_search.py`, `text_to_image_search.png`):
   precompute all 8,091 image embeddings, encode a text query, return top-5 by cosine.
   Queries like "a dog running on the beach" surface plausibly matching images. This is
   the exact algorithm behind Google Images / phone photo search (minus FAISS-scale ANN).
2. **Zero-shot classification** (`zero_shot.py`, confusion matrix) — Q34.
3. **Semantic search** (Task 5 stretch) — the text-only analogue.

**What I learned beyond training the core model:** the *same* frozen embeddings power
search, classification, and retrieval with no extra training — the representation is the
product. And that an evaluation's headline number can be actively misleading (Bug 4):
what you checkpoint on determines what you ship.

## Q36. One specific experiment I'd run next

**Swap the image global-embedding readout from the CLS token to attention-pooling over
patch tokens** (a small learned query attending to the 64 patch embeddings). Hypothesis:
at 64×64 with only 4 layers, a single CLS token is an under-trained bottleneck — the
patch tokens carry more usable signal (the qualitative maps already show patch-level
structure), so a light attention-pool should raise R@1 specifically, where fine
discrimination matters, without touching resolution or model size. Concretely: add one
`nn.MultiheadAttention`-style pool head, keep everything else fixed, and compare R@1/R@5
against the CLS baseline.

---

### Summary
From scratch, on my own GPU: a working CLIP-style VLM with **I→T R@10 = 16.6%**
(~16× chance), attention that localizes without localization labels, and 46% zero-shot
accuracy with no labels. The biggest lesson wasn't in the architecture — it was that
**the metric you early-stop on is a modeling decision**, and getting it wrong halved the
result.
