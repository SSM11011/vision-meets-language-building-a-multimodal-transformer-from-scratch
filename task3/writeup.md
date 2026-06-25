# Task 3 — Conceptual Writeup

> Empirical numbers below come from CPU runs at reduced epochs (CNN: 10 epochs;
> ViT: 6 epochs) so the whole task completes in a reasonable time. The handbook's
> headline numbers (ViT 30 epochs → 65–72%) assume a GPU/Colab; the code keeps
> those defaults. The *qualitative* comparison — CNN beating ViT on small CIFAR-10
> — holds regardless and is the actual learning point.

## 1. CNN vs ViT on CIFAR-10 — which won, and why?

My results (see `comparison_plot.png`):

| Model | epochs | final validation accuracy |
|---|---|---|
| TinyCNN | 10 | **76.2%** |
| ViT | 6 | 64.4% (still climbing) |

The **CNN reached higher validation accuracy than the ViT**, which is exactly what the
handbook predicts for a small dataset like CIFAR-10. (The ViT was still improving steeply
at epoch 6 — 41.8% → 64.4% — and would close much of the gap with the full 30-epoch
schedule plus heavier augmentation, but on this little data it does not overtake the CNN.)

The reason is **inductive bias**. A convolution hard-codes two priors that are true
of natural images: *locality* (a pixel is most related to its neighbours, so a small
sliding kernel is the right primitive) and *translation equivariance* (a cat is a cat
wherever it appears, and weight-sharing across spatial positions bakes this in). These
priors mean the CNN does not have to *learn* them from data — it starts already knowing
useful structure, so it generalizes well from only 50k images.

A ViT has almost none of this built in. To a ViT, patch (0,0) and patch (7,7) are just
two tokens in a set; it must *learn* from data that nearby patches are related and that
the same content matters regardless of position. That learning needs a lot of data
(ImageNet-21k, JFT-300M) — far more than CIFAR-10 provides. So on a small dataset the
ViT works against itself and underperforms the CNN. Its advantage (fewer assumptions →
can discover patterns a CNN's bias would hide) only pays off at large scale.

## 2. Why is patching necessary for ViT? Why not feed pixels directly?

Two reasons.

**Cost.** Self-attention is O(T²) in sequence length. A 32×32×3 image has 3072 pixels;
treating each as a token gives ~3072² ≈ 9.4 million attention scores per layer, and a
224×224 image would need T≈50k tokens and ~2.5 *billion* scores — completely
impractical. Cutting the image into, say, 4×4 patches gives only 64 tokens (an 8×8 grid),
reducing the sequence length by a factor of patch_size² and the score count
quadratically.

**Representation.** A single pixel carries almost no information — its value in isolation
is meaningless. A patch is a small chunk of structure (an edge, a texture, a colour blob)
that is a meaningful unit to attend over. Patches are to images what subword tokens are
to text: small, composable pieces the model can reason about.

## 3. Role of the CLS token — why read from it instead of averaging patches?

The `[CLS]` token is a single extra learnable vector prepended to the patch sequence. It
carries no image content of its own; instead, through the attention layers it acts as a
**learned aggregation slot** that can attend over all patch tokens and pull together
whatever information is useful for classification. After the final block, the classifier
reads only the CLS token's state.

Why not just average the patch tokens? Averaging is a *fixed, unweighted* pooling — every
patch contributes equally, including background patches that are irrelevant to the class.
The CLS token instead learns, via attention, *which* patches to weight and *how* to
combine them — a content-dependent, learnable summary rather than a uniform mean. (Global
average pooling is a valid alternative and the gap is small, but the CLS token is in the
original paper and is what I implemented.)

## 4. Why a causal mask in Task 2 but not in ViT?

A causal (lower-triangular) mask forces position *t* to attend only to positions ≤ *t*.
This is essential for **language modeling**, where the model is trained to predict the
next token: if a position could see future tokens, it would simply read off the answer,
and the model would never learn to predict — it would cheat at training and fail at
generation time (where the future genuinely doesn't exist yet).

An image has **no temporal ordering**. Patch (5,3) is not "before" or "after" patch (2,6);
they coexist. For classification we want every patch to freely exchange information with
every other patch (bidirectional attention), because the class depends on the whole image
at once. Masking would be artificial and harmful here.

**If you accidentally kept the causal mask in ViT:** each patch could only attend to
"earlier" patches in the flattening order (e.g., top-left can't see bottom-right). The
CLS token, if placed first, would be especially crippled — it could attend to *nothing*
but itself (every patch comes after it), so it would receive no image information and the
classifier would be reading a vector that never saw the image. Accuracy would collapse to
near chance. The handbook flags this as the single most common bug at this stage.

## 5. What do position embeddings encode for image patches, and why are they needed?

Attention is permutation-invariant: it treats its input as an unordered set, so by itself
it cannot tell where a patch came from. The flattening step destroys the 2D grid layout —
patch index 9 and patch index 17 are just numbers. The learned position embedding (one
vector added to each patch token, including a slot for CLS) restores that information: it
encodes **where in the image grid the patch sits** ("top-left" vs "centre" vs
"bottom-right"). The model needs this because spatial arrangement is meaningful — an eye
above a nose above a mouth is a face; the same patches shuffled are not. For text,
position embeddings encode token *order*; for images, they encode 2D *spatial location*.

## 6. What was hardest? What clicked?

The hardest part was keeping the patch tensor reshapes straight — verifying that the
`Conv2d` patch-embedding trick (`flatten(2).transpose(1, 2)`) produces exactly the same
patch ordering as the manual `unfold` method, and that the position embedding (length
`num_patches + 1`) lines up with the CLS-prepended sequence. Off-by-one on the `+1` for
CLS is an easy silent bug.

What clicked unexpectedly: that ViT is *literally* the Task 2 transformer with the mask
removed and a different tokenizer (patches instead of characters). Once the patch
embedding is in place, the entire transformer stack is reused unchanged. "An image is
worth 16×16 words" stopped being a slogan and became a precise statement about the code.
