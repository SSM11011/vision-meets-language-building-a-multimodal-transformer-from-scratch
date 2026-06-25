# Task 0 — Paper Response: *Attention Is All You Need* (Vaswani et al., 2017)

Sections read: 1 through 3.2.

## 1. What problem were the authors solving, and what were they replacing?

Before this paper, the dominant models for sequence-to-sequence tasks (machine translation,
especially) were recurrent neural networks — LSTMs and GRUs — usually arranged as an
encoder-decoder, sometimes with an attention mechanism bolted on top of the recurrence. The
core problem with recurrence is that it is **inherently sequential**: to compute the hidden
state at position *t* you must first have computed the state at *t-1*. This forbids
parallelization across the time dimension within a single example, which makes training on
long sequences slow, and it makes it hard for information to flow between distant positions
(the signal has to survive many sequential steps). Convolutional approaches (e.g.
ByteNet, ConvS2S) parallelize better but still need many layers to connect far-apart
positions, so the path length between two tokens grows with distance.

The Transformer replaces recurrence and convolution **entirely** with attention. Because
attention computes interactions between all positions directly, the path length between any
two tokens is constant (O(1)), and the whole sequence can be processed in parallel. The
authors' claim, captured in the title, is that attention alone — with no recurrence — is
sufficient, and in fact better and far faster to train.

## 2. What is self-attention computing? (in my own words)

Imagine each token in the sequence is a person at a meeting, and each person is holding a
vector that summarizes what they currently know. Self-attention is the step where everyone
updates what they know by listening to everyone else — but selectively.

Each token produces three vectors from its current representation by three separate learned
linear maps: a **query** ("what am I looking for?"), a **key** ("what do I offer / what am I
about?"), and a **value** ("here is the actual content I'll hand over"). To update a given
token, we take its query and compare it (via dot product) against the keys of every token in
the sequence. A large dot product means "this other token is relevant to me." We turn those
similarity scores into weights with a softmax (so they're non-negative and sum to one), and
then the token's new representation is the weighted average of everyone's *value* vectors,
weighted by relevance.

In pure linear-algebra terms: with queries `Q`, keys `K`, values `V` (each a matrix whose
rows are the per-token vectors), self-attention computes
`softmax(Q Kᵀ / √d_k) V`. `Q Kᵀ` is a matrix of all pairwise query-key similarities; the
softmax normalizes each row into a probability distribution over positions; multiplying by `V`
mixes the value vectors according to those distributions. The output has the same number of
rows as the input — one updated, context-aware vector per token. "Self" attention just means
Q, K, and V all come from the *same* sequence.

## 3. Three things I did not fully understand

1. **The exact reason for scaling by √d_k.** I follow that without scaling the dot products
   get large in magnitude and softmax becomes too peaked, but I don't yet have the variance
   argument crisp in my head — *why* the variance of the dot product grows linearly with `d_k`
   and how dividing by `√d_k` restores it. (The handbook says I'll prove this in Task 2, so I
   expect this to resolve.)

2. **Multi-head attention's concatenation step.** I understand running several attention
   "heads" in parallel, but I'm fuzzy on the dimensional bookkeeping: each head projects down
   to `d_k = d_model / h`, the heads are concatenated back to `d_model`, and then there's a
   final output projection `W^O`. I'm not certain I understand what that final projection buys
   us beyond just gluing the heads together, or why splitting into lower-dimensional heads
   isn't strictly worse than one big head.

3. **Positional encoding interacting with attention.** The sinusoidal positional encodings are
   *added* to the token embeddings. I don't intuitively see why adding a position signal to the
   content signal doesn't corrupt the content, or how the dot-product attention later manages to
   separate "these tokens are similar in meaning" from "these tokens are close in position."
