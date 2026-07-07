====================================================================
 Task 5 — InfoNCE Loss: Derivation from First Principles
 VLM from Scratch · Seasons of Code · IIT Bombay
====================================================================

Notation
--------
  N            number of image-caption pairs in the batch
  I_i, T_j     L2-normalized image / text embeddings (unit vectors in R^D)
  s_{ij}       cosine similarity  = I_i . T_j   (dot product on the unit sphere)
               so  s_{ij} in [-1, 1],  and  s_{ii}  is the correct (positive) pair
  tau          temperature > 0.  We use logits  z_{ij} = s_{ij} / tau.

Row i of the similarity matrix holds the similarities of image i to ALL N texts.
Pair i means image i should match text i, so the target for row i is the index i
(the diagonal).


--------------------------------------------------------------------
 1. InfoNCE for one direction (image -> text) as an explicit sum
--------------------------------------------------------------------
Principle: "maximize the log-probability of the correct pair under a softmax over
all candidate texts." For image i we define a probability distribution over the N
candidate texts using the scaled similarities as logits:

        exp(s_{ii} / tau)
  p_i = ---------------------------------           (prob. that i's match is text i)
        sum_{k=1..N} exp(s_{ik} / tau)

We want p_i to be large for every i. Maximizing the average log-probability is the
same as minimizing its negative — this negative-log-likelihood IS the loss:

  L_{i2t}
      = - (1/N) * sum_{i=1..N} log p_i

      = - (1/N) * sum_{i=1..N} log [ exp(s_{ii}/tau)
                                     / sum_{k=1..N} exp(s_{ik}/tau) ]

      = - (1/N) * sum_{i=1..N} [  s_{ii}/tau
                                  - log sum_{k=1..N} exp(s_{ik}/tau) ]  .   (*)

This is exactly cross-entropy applied row-wise to the matrix z = s/tau with target
labels (1,2,...,N) on the diagonal. The full symmetric CLIP loss adds the text->
image direction (softmax down each COLUMN) and averages:

  L = (1/2) ( L_{i2t} + L_{t2i} ).


--------------------------------------------------------------------
 2. Limiting behavior in the temperature tau
--------------------------------------------------------------------
Look at one term of (*):   L_i = -s_{ii}/tau + log sum_k exp(s_{ik}/tau).

(a) tau -> 0  (sharp).  Factor out 1/tau. The log-sum-exp becomes a hard maximum:

      log sum_k exp(s_{ik}/tau)  ->  (1/tau) * max_k s_{ik}      as tau -> 0,

  because the largest term dominates the sum exponentially. Hence

      L_i  ->  ( max_k s_{ik} - s_{ii} ) / tau .

  - If s_{ii} is the strict maximum of row i, the bracket is 0 and L_i -> 0.
  - If ANY off-diagonal s_{ij} > s_{ii}, the bracket is a positive constant divided
    by tau -> 0+, so L_i -> +infinity.

  So as tau -> 0 the loss becomes an all-or-nothing constraint: "the correct pair's
  similarity must be strictly greater than every negative's." Even a tiny violation
  is punished without bound. This is max-margin / hard-negative behavior — the model
  becomes obsessed with the single hardest negative in each row.

(b) tau -> infinity  (flat).  Now z_{ik} = s_{ik}/tau -> 0 for all k, so every
  exp(...) -> 1 and the softmax becomes uniform: p_i -> 1/N. Therefore

      L_i  ->  -log(1/N) = log N ,   a constant independent of the embeddings.

  Since L no longer depends on s_{ij}, its gradient w.r.t. every similarity -> 0.
  Training stalls: there is no signal telling embeddings to move. This is the
  "logits are flat, gradients vanish" failure mode. The loss has collapsed to the
  fixed value log N — the same baseline we see for random embeddings.

  Intuition: temperature sets how sharply the model distinguishes the positive from
  the negatives. Too sharp -> unstable, dominated by one hard negative. Too flat ->
  no distinction, no learning. CLIP therefore LEARNS tau (as log(1/tau), clamped).


--------------------------------------------------------------------
 3. Gradients w.r.t. the similarities, and their signs
--------------------------------------------------------------------
Take one row's term L_i = -log p_i with logits z_{ik} = s_{ik}/tau. This is a plain
softmax cross-entropy with target index i, so its gradient w.r.t. the logits is the
standard "softmax minus one-hot":

      dL_i / dz_{ij} = p_{ij} - [j == i]

  where p_{ij} = softmax_k(z_{ik})_j = exp(s_{ij}/tau) / sum_k exp(s_{ik}/tau),
  and [j == i] is 1 on the diagonal, 0 otherwise. Chain rule through z = s/tau
  (dz_{ij}/ds_{ij} = 1/tau) and the overall 1/N average give:

  Correct pair (j = i):
      dL_{i2t} / ds_{ii} = (1 / (N * tau)) * ( p_{ii} - 1 )      <= 0
                                                                 (since p_{ii} <= 1)

  Incorrect pair (j != i):
      dL_{i2t} / ds_{ij} = (1 / (N * tau)) * p_{ij}              >= 0
                                                                 (since p_{ij} >= 0)

Interpretation of the signs (gradient DESCENT moves opposite the gradient):

  - ds_{ii} gradient is negative  =>  descent INCREASES s_{ii}. The correct
    image-caption pair is PULLED TOGETHER. The pull is strongest when p_{ii} is
    small, i.e. exactly when the model is currently wrong — the update is largest
    where it is most needed, and fades to zero once p_{ii} -> 1.

  - ds_{ij} gradient is positive  =>  descent DECREASES s_{ij}. Every incorrect
    (image i, text j) pair is PUSHED APART. The push on a negative is proportional
    to p_{ij}: hard negatives (those the model wrongly finds similar, large p_{ij})
    are repelled hardest; easy negatives are barely touched.

  - Conservation check: sum_j dL_i/dz_{ij} = (sum_j p_{ij}) - 1 = 1 - 1 = 0. The
    total pull on the positive exactly balances the total push on the negatives, so
    the embeddings are rearranged on the sphere rather than all collapsing or
    exploding. The 1/tau prefactor shows temperature also scales gradient MAGNITUDE:
    smaller tau -> larger gradients (another reason tiny tau is unstable).

The text->image direction gives the symmetric statements down each column; averaging
the two directions yields the final CLIP gradient.

====================================================================
 Summary: InfoNCE = row/column softmax cross-entropy on cosine
 similarities scaled by 1/tau, targeting the diagonal. It pulls
 positives together, pushes negatives apart with force proportional
 to how wrongly-similar they are, and needs a well-chosen (learned,
 clamped) temperature to avoid the tau->0 instability and the
 tau->infinity collapse to the constant log N.
====================================================================
