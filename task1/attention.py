"""Task 1, Part B — A single self-attention head, written from scratch.

This extends the bigram model: token embeddings + position embeddings are
summed, passed through ONE causal self-attention head, then projected to logits.
Everything in the Head is hand-written (no nn.MultiheadAttention).

Run:
    python task1/attention.py            # full run: 5000 steps, lr 1e-3
    python task1/attention.py --smoke    # fast sanity run

Expected (full run): loss drops to ~2.4, marginally below the bigram baseline.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "input.txt")

# ----- Handbook hyperparameters (Task 1 Part B) -----
BATCH_SIZE = 32
BLOCK_SIZE = 8
N_EMBD = 32
HEAD_SIZE = 32
LEARNING_RATE = 1e-3
MAX_STEPS = 5000
EVAL_ITERS = 200
SEED = 1337


def load_data() -> str:
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"Dataset not found at {DATA_PATH}.\n"
            f"Run `python {os.path.join('task1', 'download_data.py')}` first."
        )
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return f.read()


def build_tokenizer(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]                # noqa: E731
    decode = lambda ids: "".join(itos[i] for i in ids)     # noqa: E731
    return len(chars), encode, decode


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


class Head(nn.Module):
    """One head of causal self-attention."""

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # tril is not a parameter -> register as a buffer so it moves with .to(device)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        wei = q @ k.transpose(-2, -1)          # (B, T, T)
        wei = wei * (k.size(-1) ** -0.5)       # scale by 1/sqrt(d_k)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # causal mask
        wei = F.softmax(wei, dim=-1)           # (B, T, T)
        out = wei @ v                          # (B, T, head_size)
        return out


class AttentionLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, n_embd: int, head_size: int, block_size: int):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd, head_size, block_size)
        self.lm_head = nn.Linear(head_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                                  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = tok_emb + pos_emb                                                       # (B, T, n_embd)
        x = self.sa_head(x)                                                         # (B, T, head_size)
        logits = self.lm_head(x)                                                    # (B, T, vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]   # crop to block_size for position emb
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


@torch.no_grad()
def estimate_loss(model, train_data, val_data, eval_iters, block_size, batch_size, device):
    out = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train_attention(smoke: bool = False, steps: int | None = None):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_data()
    vocab_size, encode, decode = build_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    max_steps = steps if steps is not None else (300 if smoke else MAX_STEPS)
    eval_interval = max(1, max_steps // 10)
    eval_iters = 20 if smoke else EVAL_ITERS

    model = AttentionLanguageModel(vocab_size, N_EMBD, HEAD_SIZE, BLOCK_SIZE).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"device={device}  steps={max_steps}  params={sum(p.numel() for p in model.parameters())}")

    for step in range(max_steps + 1):
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, train_data, val_data, eval_iters,
                                   BLOCK_SIZE, BATCH_SIZE, device)
            print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f}")
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, max_new_tokens=200)[0].tolist())
    print("\n----- 200-char sample (1-head attention) -----\n" + sample)
    return model, decode, sample


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Single self-attention head LM on Tiny Shakespeare.")
    p.add_argument("--smoke", action="store_true", help="fast sanity run (300 steps)")
    p.add_argument("--steps", type=int, default=None, help="override number of training steps")
    args = p.parse_args()
    train_attention(smoke=args.smoke, steps=args.steps)
