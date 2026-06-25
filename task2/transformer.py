"""Task 2 — Multi-head attention and a full decoder-only transformer.

Builds, from scratch:
  * Head                  - one causal self-attention head (from Task 1)
  * MultiHeadAttention    - several heads in parallel + output projection + dropout
  * FeedForward           - 4x MLP with ReLU + dropout
  * Block                 - pre-norm transformer block (attn + ffn + 2 LayerNorms + residuals)
  * GPTLanguageModel      - token+pos embeddings -> N blocks -> final LayerNorm -> lm head

The Block supports ablations (Task 2 Part D): toggling residual connections and
LayerNorm so we can measure how catastrophic their removal is.

Run:
    python task2/transformer.py                 # full: train baseline + 3 ablations, write artifacts
    python task2/transformer.py --smoke         # fast end-to-end sanity run
    python task2/transformer.py --steps 1500    # custom step count

Artifacts written to task2/: ablation_plot.png, samples.txt
Expected (full run): baseline val loss ~1.5.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
# Reuse the Tiny Shakespeare file downloaded for Task 1.
DATA_PATH = os.path.join(HERE, os.pardir, "task1", "input.txt")

# ----- Handbook hyperparameters (Task 2) -----
BATCH_SIZE = 64
BLOCK_SIZE = 64
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2
LEARNING_RATE = 3e-4
MAX_STEPS = 5000
EVAL_ITERS = 200
SEED = 1337


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_data() -> str:
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            f"Dataset not found at {os.path.normpath(DATA_PATH)}.\n"
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


# --------------------------------------------------------------------------- #
# Model components
# --------------------------------------------------------------------------- #
class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)        # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v                                              # (B, T, head_size)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """Position-wise MLP with a 4x hidden expansion."""

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Pre-norm transformer block. `use_residual` / `use_layernorm` enable ablations."""

    def __init__(self, n_embd, n_head, block_size, dropout,
                 use_residual=True, use_layernorm=True):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        # nn.Identity keeps the forward pass uniform when LayerNorm is ablated.
        self.ln1 = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()
        self.ln2 = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()

    def forward(self, x):
        if self.use_residual:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.attn(self.ln1(x))
            x = self.ffn(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout,
                 use_residual=True, use_layernorm=True):
        super().__init__()
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, block_size, dropout, use_residual, use_layernorm)
             for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.token_embedding_table(idx)
        pos = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok + pos
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
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


def train_one(variant, train_data, val_data, vocab_size, device,
              max_steps, eval_iters, log=True):  # noqa: D401
    """Train a single model variant. Returns (model, step_list, train_loss_list)."""
    torch.manual_seed(SEED)
    use_residual = variant != "no_residual"
    use_layernorm = variant != "no_layernorm"
    model = GPTLanguageModel(vocab_size, N_EMBD, N_HEAD, N_LAYER, BLOCK_SIZE, DROPOUT,
                             use_residual=use_residual, use_layernorm=use_layernorm).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    eval_interval = max(1, max_steps // 10)

    steps_hist, loss_hist = [], []
    for step in range(max_steps + 1):
        if step % eval_interval == 0 or step == max_steps:
            losses = estimate_loss(model, train_data, val_data, eval_iters,
                                   BLOCK_SIZE, BATCH_SIZE, device)
            steps_hist.append(step)
            loss_hist.append(losses["train"])
            if log:
                print(f"[{variant:13s}] step {step:5d} | train {losses['train']:.4f} | "
                      f"val {losses['val']:.4f}")
        xb, yb = get_batch(train_data, BLOCK_SIZE, BATCH_SIZE, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, steps_hist, loss_hist


def save_ablation_plot(histories, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    labels = {
        "baseline": "Baseline (residual + LayerNorm)",
        "no_residual": "No residual connections",
        "no_layernorm": "No LayerNorm (residual kept)",
    }
    for variant, (steps, losses) in histories.items():
        plt.plot(steps, losses, marker="o", markersize=3, label=labels.get(variant, variant))
    plt.xlabel("training step")
    plt.ylabel("training loss")
    plt.title("Task 2 Part D — Ablation: residuals & LayerNorm")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved {path}")


def main(smoke=False, steps=None, ablations=True, eval_iters=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = load_data()
    vocab_size, encode, decode = build_tokenizer(text)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    max_steps = steps if steps is not None else (200 if smoke else MAX_STEPS)
    if eval_iters is None:
        eval_iters = 20 if smoke else EVAL_ITERS
    print(f"device={device}  vocab={vocab_size}  steps={max_steps}")

    histories = {}
    variants = ["baseline", "no_residual", "no_layernorm"] if ablations else ["baseline"]
    baseline_model = None
    for variant in variants:
        model, steps_hist, loss_hist = train_one(
            variant, train_data, val_data, vocab_size, device, max_steps, eval_iters)
        histories[variant] = (steps_hist, loss_hist)
        if variant == "baseline":
            baseline_model = model

    # 300-char sample from the baseline model.
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(baseline_model.generate(context, max_new_tokens=300)[0].tolist())
    samples_path = os.path.join(HERE, "samples.txt")
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write("===== Task 2 baseline transformer — 300-char sample =====\n")
        f.write(sample + "\n")
    print(f"Saved {samples_path}")
    print("\n----- 300-char sample (baseline) -----\n" + sample)

    if ablations:
        save_ablation_plot(histories, os.path.join(HERE, "ablation_plot.png"))
    return histories


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Decoder-only transformer on Tiny Shakespeare.")
    p.add_argument("--smoke", action="store_true", help="fast sanity run (200 steps)")
    p.add_argument("--steps", type=int, default=None, help="override training steps")
    p.add_argument("--no-ablations", action="store_true", help="train only the baseline")
    p.add_argument("--eval-iters", type=int, default=None, help="batches per loss estimate")
    args = p.parse_args()
    main(smoke=args.smoke, steps=args.steps, ablations=not args.no_ablations,
         eval_iters=args.eval_iters)
