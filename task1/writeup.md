# Task 1 — Conceptual Writeup

## 1. Why do we divide attention scores by √d_k?

The attention score for a query–key pair is their dot product, `q · k = Σ_{i=1}^{d_k} q_i k_i`.
If the components of `q` and `k` are roughly independent with mean 0 and variance 1, then each
product term `q_i k_i` has variance ~1, and summing `d_k` of them gives a dot product with
variance ~`d_k`. So as the head dimension grows, the *magnitude* (standard deviation `√d_k`) of
the raw scores grows too.

This matters because of the **shape of the softmax**. Softmax is exponential, so when the input
scores are large in magnitude, the exponentials are wildly different sizes and the distribution
becomes extremely **peaked** — almost all the weight lands on a single position and the rest are
nearly zero. A near-one-hot softmax has tiny gradients (the Jacobian `p_i(δ_ij − p_j)` vanishes
when `p` is one-hot), so learning stalls.

Dividing by `√d_k` rescales the scores so their variance is ~1 regardless of head dimension. The
softmax then stays in a well-behaved, non-saturated regime, and gradients flow. (Task 2 makes the
variance argument rigorous.)

## 2. Masking before vs after softmax

The causal mask sets future-position scores to `−∞` **before** the softmax. After exponentiation,
`exp(−∞) = 0`, so future positions get exactly zero weight, and — crucially — the softmax
**renormalizes over only the visible positions**, so the remaining weights still sum to 1.

If instead you applied the mask **after** softmax (zeroing out future entries), the softmax would
already have included the future positions in its denominator. Zeroing them afterward leaves the
surviving weights summing to **less than 1**, so the attention output is a shrunken, improperly
normalized average — and worse, information about the future tokens has already leaked into the
denominator, contaminating the visible weights. The before-softmax approach is correct because it
removes the future from the computation *entirely* before normalization happens.

## 3. Q, K, V in my own words

**Analogy.** Think of a library lookup. The **query** is the question I'm asking ("I need
information relevant to position *t*"). Each token also publishes a **key**, which is like the
label on a book spine advertising what it's about. I compare my query against every key to decide
how relevant each token is. The **value** is the actual content I take away from a token once I've
decided it's relevant. The output is a relevance-weighted blend of values.

**Linear-algebra view.** Q, K, V are three different learned linear projections of the same input
`x` (each is `x W` with a learned weight matrix, no bias). `W_q` maps each token into a "what am I
looking for" subspace; `W_k` maps it into a "what do I match against" subspace (Q and K live in the
same space so their dot product is meaningful); `W_v` maps it into the "what content do I
contribute" subspace. The model learns all three projections so that the dot-product similarities
in QK-space pick out useful tokens, and the V-projection carries the useful information.

## 4. Why does single-head attention barely beat the bigram?

My bigram converged to val loss ≈ 2.49 and the single-head attention model to ≈ 2.41 — a real but
small improvement. The bottleneck is **not** the attention mechanism itself; it is mostly
**capacity and depth**, with context length a secondary factor:

- The model is a *single* head feeding *directly* into the output projection — there is no MLP, no
  stacking of blocks, no nonlinearity-rich processing. One linear value-mixing step followed by a
  linear readout is a very shallow function.
- `block_size = 8` means only 8 characters of context, which limits how much attention can even
  help — but on Shakespeare at the character level, even modest context is useful, so this is not
  the main limit.

The fix is Task 2: multiple heads, an MLP per block, and several stacked blocks give the depth and
capacity needed for the loss to drop substantially (to ~1.5).

## 5. Qualitative comparison of generated text (200 chars each)

**Bigram:**
```
THAr toud en: ty s come

Pat s, ore wilkigulllo isowanornon ws alinthifofr
Cuker:
Thinertor wien IUCUng l t.
And! he s thaswsantld was w,
Aue y s Ind benteecofo thime me,
```

**Single-head attention:**
```
ORKHIO:
AOROLENToatigiste baroth we win thyes t be bers, thard, tung:
Here ed fornow foue me, louk men hesy pad?

LAkensind tay mens borden.
Lo utfousorienotes ngst cunt howoute ty dyot san yobacleran
```

**Difference.** Both are gibberish, as expected. The bigram only knows pairwise letter statistics,
so it produces plausible *letter* transitions but no structure beyond two characters. The attention
model, with 8 characters of context and learned position embeddings, produces slightly more
word-like chunks and noticeably better newline/colon usage that imitates Shakespeare's
speaker-name layout (e.g. `ORKHIO:`, `LAkensind`). The gains are modest — exactly as the handbook
warns — because one head with tiny context and no depth can't do much more than refine local
statistics.
