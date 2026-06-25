# Task 4 — Conceptual Writeup

## 1. Exact difference between self-attention and cross-attention

**Mechanically:** in self-attention the queries, keys, and values are all linear
projections of the *same* input sequence `x`. In cross-attention the **queries come from
one sequence** (`x`) and the **keys and values come from a different sequence**
(`context`): `q = W_q x`, `k = W_k context`, `v = W_v context`. That is the entire change
— one line.

**Semantically:** self-attention lets a sequence mix information *within itself* (each
token refines its representation using the other tokens of the same sequence).
Cross-attention lets one sequence *pull in* information from another, possibly of a
different modality — e.g., text tokens querying image patch features. It is the bridge
that lets two separately-encoded streams communicate.

## 2. Causal masking in the three decoder sub-layers

My decoder block is: (1) causal self-attention, (2) cross-attention (no mask), (3) MLP.

- **Self-attention is causally masked.** The decoder is autoregressive — it generates the
  caption one token at a time and is trained to predict the next token. Token *t* must not
  see tokens > *t*, or it would cheat by reading the answer, and generation (where the
  future doesn't exist) would break.
- **Cross-attention is NOT masked.** It attends from text queries onto the *image*
  features. The whole image is available at once — there is no temporal order among image
  patches and nothing to hide. Every text position may freely look at every patch. (This
  mirrors the original encoder–decoder Transformer: decoder self-attention is masked,
  cross-attention onto the encoder is not.)
- **The MLP is position-wise** and has no attention at all, so masking does not apply — it
  transforms each position independently.

**If you swapped which were causal:** masking the cross-attention would stop later caption
tokens from seeing "later" image patches — meaningless and harmful, since patch order is
arbitrary; the model would be denied parts of the image for no reason. Removing the mask
from self-attention would let each position see future caption tokens during training, so
the model would learn to copy the next token rather than predict it — training loss would
collapse to ~0 but generation would produce garbage, because at inference the future
tokens it learned to rely on are not there.

## 3. Why does output length equal query length, not context length?

Walk the shapes. With `x: (B, T_x, C)` and `context: (B, T_c, C)`:
- `q = W_q x` → `(B, T_x, head)`, `k = W_k context` → `(B, T_c, head)`,
  `v = W_v context` → `(B, T_c, head)`.
- scores `wei = q @ kᵀ` → `(B, T_x, T_c)` — one row per query position, one column per
  context position.
- softmax is over the last dim (`T_c`), so each query row becomes a distribution over the
  `T_c` context positions.
- output `wei @ v` → `(B, T_x, T_c) @ (B, T_c, head)` = `(B, T_x, head)`.

The context length `T_c` is summed away by the matmul; what survives is `T_x`. Intuitively,
cross-attention **produces one output vector per query** — it transforms the query
sequence using information retrieved from the context, so it keeps the query's length.

## 4. If the vision encoder's n_embd differed from the text decoder's — two fixes

1. **A linear projection adapter.** Add `nn.Linear(n_embd_img, n_embd_text)` and apply it
   to the image features before they enter cross-attention as keys/values. This is what
   real systems (e.g. BLIP-2's Q-Former, LLaVA's projection layer) do.
2. **Make the dimensions equal by construction.** Set the vision encoder's `n_embd` equal
   to the text decoder's from the start (what I did here — both 128), so no adapter is
   needed. (A third option: use separate `W_k`/`W_v` in the cross-attention head that map
   directly from `n_embd_img` to `head_size`, absorbing the mismatch into the existing
   projections.)

## 5. What the cross-attention visualization showed

Training drove the loss from ~log(23)=3.14 down to **0.065** (well below the 0.5 target),
and the model produces grammatical "this is a \<class\>" captions for every test image —
confirming gradients flow through the whole vision→cross-attention→text stack. On the 10
held-out test images it got the class exactly right **4/10** of the time (e.g. cat, dog,
frog, truck correct; see `samples.txt`). That modest test accuracy is not a cross-attention
failure — it reflects the **vision encoder**: a from-scratch ViT trained for only ~2
epochs' worth of steps via the captioning objective is itself only a ~40%-accurate CIFAR
classifier (consistent with the Task 3 ViT's accuracy that early in training). The
cross-modal *mechanism* works; the bottleneck is how well the tiny ViT classifies.

I plotted the head-averaged cross-attention of the last decoder block (`attention_viz.png`):
rows are caption tokens (queries), columns are the image context positions (CLS + 64
patches). Given how **tiny and contrived** the synthetic task is — 10 fixed templates all
sharing the prefix "this is a " — I did not expect a crisp, semantically meaningful patch
map, and the visualization reflects that. The shared-prefix tokens carry no class
information, so their attention has little reason to localize on any particular patch; the
informative signal concentrates when the model emits the *class word*. The honest read: the
routing is real (the model clearly conditions on the image) but the *spatial* pattern is
weak and not strongly interpretable at this scale — exactly the outcome the handbook calls
out as valid to report.

## 6. Connection to a real multimodal model

What I built is closest to **Flamingo / BLIP-style cross-attention fusion**. In Flamingo, a
frozen vision encoder produces image features, and the language model's blocks are
augmented with **gated cross-attention layers** that let text tokens attend to those image
features — structurally the same "text queries, image keys/values" mechanism in my
`DecoderBlock`.

**What is the same:** the core idea — a vision encoder turns an image into a sequence of
feature vectors, and cross-attention injects those features into a text decoder so
generation is conditioned on the image.

**What is different:** scale and training. Flamingo uses a huge pretrained LLM and a
powerful pretrained vision encoder, trains on billions of image–text pairs, adds gating so
the cross-attention can be smoothly introduced without destabilizing the language model,
and produces open-ended natural language. Mine is a tiny from-scratch model trained on 10
synthetic templates as an engineering check. CLIP, by contrast, is a *different* design —
it uses contrastive alignment of separate image/text embeddings (no cross-attention
generation), which is the subject of Task 5.
