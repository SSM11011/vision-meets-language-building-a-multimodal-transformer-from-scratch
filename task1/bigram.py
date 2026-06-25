"""Task 1, Part A — Character-level bigram language model on Tiny Shakespeare.

A bigram model predicts the next character from *only* the current character.
The model is literally a lookup table of shape (vocab_size, vocab_size),
implemented with nn.Embedding(vocab_size, vocab_size).

Run:
    python task1/download_data.py        # once, to fetch input.txt
    python task1/bigram.py               # full run: 3000 steps, lr 1e-2
    python task1/bigram.py --smoke       # fast sanity run

Expected (full run): loss starts ~4.17 (= log 65) and drops to ~2.5.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "input.txt")

# ----- Handbook hyperparameters (Task 1 Part A) -----
BATCH_SIZE = 32       # how many independent sequences per batch (B)
BLOCK_SIZE = 8        # maximum context length (T)
LEARNING_RATE = 1e-2
MAX_STEPS = 3000
EVAL_INTERVAL = 300
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
    """Sorted unique characters -> stoi / itos dicts and encode/decode fns."""
    chars = sorted(set(text))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]                     # noqa: E731
    decode = lambda ids: "".join(itos[i] for i in ids)          # noqa: E731
    return vocab_size, stoi, itos, encode, decode


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str):
    """Sample batch_size random chunks; y is x shifted by one position."""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    """Each token directly indexes a row of logits over the next token."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.token_embedding_table(idx)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(idx)             # (B, T, vocab_size)
            logits = logits[:, -1, :]         # bigram: only last step matters
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


def train_bigram(smoke: bool = False, steps: int | None = None):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    text = load_data()
    vocab_size, _, _, encode, decode = build_tokenizer(text)
    print(f"vocab_size = {vocab_size}  (expected ~65 for Tiny Shakespeare)")

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    max_steps = steps if steps is not None else (200 if smoke else MAX_STEPS)
    eval_interval = max(1, max_steps // 10)
    eval_iters = 20 if smoke else EVAL_ITERS

    model = BigramLanguageModel(vocab_size).to(device)
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

    # Generate a sample from a single newline-ish start token (id 0).
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(context, max_new_tokens=200)[0].tolist())
    print("\n----- 200-char sample (bigram) -----\n" + sample)
    return model, decode, sample


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bigram language model on Tiny Shakespeare.")
    p.add_argument("--smoke", action="store_true", help="fast sanity run (200 steps)")
    p.add_argument("--steps", type=int, default=None, help="override number of training steps")
    args = p.parse_args()
    train_bigram(smoke=args.smoke, steps=args.steps)
