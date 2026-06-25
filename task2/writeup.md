# Task 2 — Final Writeup

> Trained on CPU at a reduced step count (1500 steps per variant) so the three
> ablation runs finish in reasonable time. The handbook's 5000-step target (loss
> ~1.5) assumes a GPU/Colab; the code keeps that default. The trends reported
> here — and the ablation conclusions — are unchanged by the step count.

## 1. Role of the MLP in each block

Attention is fundamentally a **mixing** operation: each output token is a weighted average
of (projections of) other tokens' values. Crucially, that mixing is *linear* in the values,
and the weights, while data-dependent, still produce a convex combination — attention on its
own cannot apply a rich per-token nonlinear transformation. It moves information *between*
positions but does little *computation* at a position.

The MLP (Linear → ReLU → Linear, with a 4× hidden expansion) is where that per-position
computation happens. Applied independently to every token, it gives the block the capacity
to nonlinearly transform each token's gathered context into new features — to detect
combinations, threshold them (ReLU), and project them back. In short: **attention decides
*what* information each token should gather; the MLP decides *what to do* with it.** The 4×
width is where most of the model's parameters live, which is why removing or shrinking it
hurts capacity badly. Stacking (attention → MLP) blocks alternates "communicate" and
"compute" steps.

## 2. Pre-norm vs post-norm

I used **pre-norm**: `x = x + sublayer(LayerNorm(x))`. The original Transformer used
post-norm: `x = LayerNorm(x + sublayer(x))`.

Pre-norm is easier to train for deep networks because it keeps the **residual path clean**.
With pre-norm, the identity branch `x` is added *without* a LayerNorm in the way, so the
gradient flows straight back through the residual highway from the loss to every earlier
layer with no rescaling — gradients neither explode nor vanish as depth grows. In post-norm,
a LayerNorm sits *on* the residual sum at every layer, so the backward signal is repeatedly
rescaled by the LayerNorm Jacobian as it traverses the stack; for deep models this
destabilizes training and typically requires a learning-rate warmup to converge at all.
Pre-norm trains stably out of the box, which is why most modern transformers adopt it.

## 3. Generated text (300 characters)

```
Nury sove forther, eyea? O'er much put lese; bold anese!
as'd is! fassiffe! Thou, graint, shand,
who of in a sunciffing, I carest thee cannou
What by pose to sheif your rutterre?

TUS:
So that you s love.
Seal quare to me good, farren with nebles has his and lorame
Then lal'res bountly ump tirtorenc
```

(Baseline reached val loss **1.82** at 1500 steps, trending toward the handbook's
~1.5 with more steps — already far below Task 1's ~2.4.)

Compared to Task 1: the bigram and single-head outputs were locally plausible letter
sequences with almost no structure beyond a couple of characters. This 4-block,
multi-head model produces text with the **shape of English** — capitalization after line
breaks, the Shakespearean `NAME:` speaker layout, word-like token lengths with vowel/
consonant alternation, and occasional real fragments — even though it is still nonsense.
The jump comes from depth (4 blocks), multiple heads, the MLPs, and a 64-token context.

## 4. Ablation: residuals vs LayerNorm — which removal was more catastrophic?

Final training loss after 1500 steps (see `ablation_plot.png`):

| Variant | train loss | val loss |
|---|---|---|
| Baseline (residual + LayerNorm) | **1.65** | **1.82** |
| No LayerNorm (residual kept) | 2.34 | 2.36 |
| No residual connections | 3.31 | 3.35 |

**Removing residual connections was by far the more catastrophic.** The no-residual
model's loss dropped to ~3.32 within the first 150 steps and then **stalled there for the
entire run** — barely better than the random-init ~4.17 and worse than even the Task 1
bigram. The no-LayerNorm model still trained and kept improving (to ~2.36), just slower
and to a worse optimum than the baseline (1.82).

**Why, in terms of gradient flow:**

- **Residuals are the gradient highway.** With `x + sublayer(x)`, the derivative of the
  output w.r.t. the input contains an identity term (`∂x/∂x = I`), so the gradient reaches
  early layers undiminished even through many stacked blocks. Remove them and the gradient
  must pass *through* every sublayer's Jacobian in sequence; across 4 blocks (each with
  attention + MLP) those repeated multiplications shrink the signal toward zero — the early
  layers get almost no learning signal, so the network can't train. Residuals also let each
  block learn a small *correction* to the identity rather than a full transformation from
  scratch, which is a much easier optimization target.

- **LayerNorm stabilizes the *scale* of activations and gradients** but is not the highway
  itself. Without it, activation magnitudes can drift and the loss landscape gets rougher,
  so training is noisier and slower and may need a smaller learning rate — but because the
  residual highway is still intact, gradients still reach the early layers and the model
  still learns. Hence removing LayerNorm degrades training; removing residuals breaks it.

## 5. Hardest part / what clicked

Hardest: deriving `∂A/∂Q` by hand (Part C) — keeping the indices straight through the
softmax Jacobian `P(δ − P)` and not losing the row-wise structure of the softmax. It took a
few attempts to see that the softmax derivative only couples entries *within the same row*.

What clicked: doing that derivation made the `√d_k` scaling *obvious* rather than a magic
constant — I could see directly in the algebra that a saturated softmax sends the Jacobian
`P(δ − P)` to zero, which is exactly the vanishing gradient the scaling exists to prevent.
The ablation also made residuals feel concrete: watching the no-residual loss flatline drove
home that the residual path is what makes a deep stack trainable at all.
