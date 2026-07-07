# Task 5 — Contrastive Learning and InfoNCE Loss (Writeup)

This task builds and verifies the loss function that makes vision–language alignment
happen: **InfoNCE**, the contrastive objective behind CLIP. All code is from scratch
(`loss.py`, `encoders.py`, `clip_model.py`) and every claim below is backed by a run
in this folder. The full math derivation is in [`derivation.pdf`](derivation.pdf).

## Deliverables in this folder

| File | What it is |
|------|-----------|
| `loss.py` | `InfoNCELoss` — symmetric InfoNCE with a learnable, clamped log-temperature. |
| `test_loss.py` | The four handbook sanity tests (all pass). |
| `encoders.py` | `ImageEncoder` (ViT) and `TextEncoder`, each exposing `.encode()`. |
| `clip_model.py` | `CLIPStyleModel` — two towers meeting only at the loss (no cross-attention). |
| `toy_alignment.py` | Isolated 500-step alignment experiment + plots. |
| `derivation.md` → `derivation.pdf` | InfoNCE derived from first principles. |
| `semantic_search.py` | Applied Stretch — a tiny semantic search engine. |

Reproduce everything: `python test_loss.py && python clip_model.py && python toy_alignment.py`.

## Verified results

**Sanity tests** (`python test_loss.py`):

```
[1] identical embeds       loss = 0.00014   (perfectly aligned -> ~0)
[2] random embeds (tau=1)  loss = 5.54871   (~log(256) = 5.54518)
[3] half-aligned mixture   loss = 2.78699   (between the two references)
[4] gradients flow         grads present & finite on log_inv_tau and both inputs
```

**CLIP model on dummy data** (`python clip_model.py`): 3.76M params; image/text
embeddings `(16, 128)`; init loss `3.02 ≈ log(16)=2.77`; gradients reach all 101
parameter tensors.

**Toy alignment** (`python toy_alignment.py`): loss fell from `log(32)=3.47`
(step-0 measured 4.25, higher because the sharp 0.07 temperature scales up the random
logits) to **0.0001** by step 100. Final learned similarity matrix: **avg diagonal
0.867**, **avg off-diagonal −0.027** — a clean diagonal, see
[`toy_similarity.png`](toy_similarity.png) and [`toy_loss_curve.png`](toy_loss_curve.png).

---

## Answers to the handbook questions

### 1. Explain InfoNCE in your own words. Why is it different from classification?
InfoNCE turns "find the matching caption" into a softmax classification, but with a
crucial twist: **the classes are the other examples in the batch, not a fixed label
set.** For each image we score its cosine similarity to all N captions, treat those N
scores as logits, and use cross-entropy with the target being the image's own caption
(the diagonal). Standard classification has a fixed output head with one weight vector
per class; here there is no head and no fixed vocabulary of classes — the "correct
answer" is defined *relationally* ("this pair matches, everything else in the batch
does not"). The model is asked to arrange embeddings on a sphere so each image is
nearest its own caption. Because the negatives are just whatever else is in the batch,
the same loss works for any two paired modalities.

### 2. Why is temperature important? Too high / too low? Why learn it?
Temperature `tau` controls how sharply the softmax separates the positive from the
negatives (logits are `similarity / tau`).
- **Too high (~1.0):** logits are flat, the softmax is near-uniform, gradients are
  weak. In the limit `tau → ∞` the loss collapses to the constant `log N` with **zero
  gradient** — training stalls (derived in §2 of `derivation.pdf`).
- **Too low (~0.01):** logits are extreme; the loss becomes an all-or-nothing
  max-margin constraint dominated by the single hardest negative, and the `1/tau`
  factor blows up gradient magnitudes, causing instability / collapse.
CLIP **learns** `tau` because the right sharpness changes during training, and it
parameterises it as `log(1/tau)` (clamped to `[log 1, log 100]`, i.e. `tau ∈
[0.01, 1]`) because optimising in log-space keeps `tau` positive and its gradient
well-scaled. My `InfoNCELoss` does exactly this.

### 3. Why must embeddings be L2-normalized?
Normalization makes similarity **cosine** (direction only). Without it, the dot
product `I·T` scales with `‖I‖‖T‖`, so the model can lower the loss by simply
**growing or shrinking embedding magnitudes** instead of improving alignment — the
positive's logit can be inflated by making its norm large, bypassing the objective
entirely. Projecting onto the unit sphere removes that degree of freedom, so the only
way to reduce the loss is to change *directions* — i.e. to actually align matching
pairs. It also bounds the logits to `[-1/tau, 1/tau]`, which is what makes the
temperature interpretable and the training numerically stable.

### 4. Toy alignment: smooth or jumpy? Final diagonal vs off-diagonal?
The loss dropped **smoothly and very fast** — a steep exponential-looking decay from
4.25 at step 0 to ~0.01 by step ~60 and 0.0001 by step 100, then a flat floor (see
`toy_loss_curve.png`). No jumps or plateaus, because with only 32 fixed vectors and
two free linear maps the problem is convex-like and easily separable. Final similarity
matrix: **average diagonal = 0.867**, **average off-diagonal = −0.027** (worst
diagonal 0.843, best off-diagonal 0.169). The diagonal is ~0.9 higher than the
off-diagonal — unambiguous alignment.

### 5. Why does bigger batch size help contrastive learning? The catch?
The batch size **is** the number of negatives: with batch N, each positive is
contrasted against N−1 in-batch negatives for free. More negatives make the softmax
denominator a harder, more informative discrimination task — the model must push the
positive above many more competitors, which yields a tighter estimate of the InfoNCE
mutual-information bound and better representations. **The catch:** (a) memory/compute
for the N×N similarity matrix grows as O(N²), and encoding N images+texts per step
grows linearly, so large N needs a big GPU (CLIP used 32k) or tricks like gathering
negatives across GPUs / a memory bank (MoCo); (b) beyond a point the extra negatives
are mostly easy and give diminishing returns; (c) very large batches can need LR
warmup/tuning to stay stable. At our 64×64 Flickr8k scale, batch 128 is the sweet spot.

---

## Applied Stretch — semantic search
`semantic_search.py` embeds a 40-sentence corpus with `all-MiniLM-L6-v2` and returns
the top-5 by cosine similarity — the exact mechanism behind vector databases. It
retrieves *semantically* (e.g. "a pet playing indoors" surfaces the kitten/puppy
sentences, not keyword matches). **To scale to a million documents:** replace the
exact O(N) dot-product scan with an approximate-nearest-neighbour index (FAISS
HNSW/IVF or a managed vector DB) for sub-linear lookup, quantise vectors (int8/PQ) to
shrink memory, precompute/persist embeddings, and shard + GPU-batch the query encoder.
