"""Task 4, Part C/D — A multimodal model: a text decoder cross-attends to a ViT.

Architecture (all from scratch):
  * ViTEncoder        - the Task 3 ViT without its classification head; returns
                        (B, 1+num_patches, n_embd) patch-token features.
  * MultiHeadAttention      - causal self-attention (Task 2).
  * MultiHeadCrossAttention - queries from text, keys/values from image features (no mask).
  * DecoderBlock      - three sub-layers: causal self-attn -> cross-attn -> MLP.
  * MultimodalModel   - token+pos embeddings -> N decoder blocks -> LM head.

Training task (synthetic, for engineering verification only): pair each CIFAR-10
image with the caption "this is a <class>" and train the decoder to produce that
caption conditioned on the image. Loss should drop from ~log(vocab) toward ~0.

Run:
    python task4/multimodal.py             # full: ~2000 steps
    python task4/multimodal.py --smoke     # fast end-to-end sanity run
    python task4/multimodal.py --steps 1500

Artifacts -> task4/: training_curve.png, attention_viz.png, samples.txt
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, os.pardir, "task3", "data")

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

# ----- Hyperparameters (handbook: same n_embd for vision and text) -----
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
PATCH_SIZE = 4
IMG_SIZE = 32
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_STEPS = 2000
BATCH_SIZE = 64
SEED = 1337


# --------------------------------------------------------------------------- #
# Text tokenizer over the (tiny) synthetic caption vocabulary
# --------------------------------------------------------------------------- #
class CharTokenizer:
    BOS, EOS, PAD = "^", "$", "#"

    def __init__(self):
        captions = [f"{self.BOS}this is a {c}{self.EOS}" for c in CIFAR10_CLASSES]
        chars = sorted(set("".join(captions)) | {self.PAD})
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.block_size = max(len(c) for c in captions)  # full caption incl BOS/EOS

    def caption(self, class_idx: int) -> str:
        return f"{self.BOS}this is a {CIFAR10_CLASSES[class_idx]}{self.EOS}"

    def encode_padded(self, class_idx: int):
        s = self.caption(class_idx)
        ids = [self.stoi[c] for c in s]
        ids += [self.pad_id] * (self.block_size - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# --------------------------------------------------------------------------- #
# Vision encoder (ViT without the classification head)
# --------------------------------------------------------------------------- #
class _BiHead(nn.Module):
    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class _BiMHA(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([_BiHead(n_embd, head_size, dropout) for _ in range(n_head)])
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class _ViTBlock(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = _BiMHA(n_embd, n_head, head_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ViTEncoder(nn.Module):
    """ViT without classification head; returns all tokens (CLS + patches)."""

    def __init__(self, img_size=32, patch_size=4, in_chans=3,
                 n_embd=128, n_head=4, n_layer=4, dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, n_embd, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, n_embd))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([_ViTBlock(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)   # (B, num_patches, n_embd)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)                        # (B, 1+num_patches, n_embd)
        x = self.dropout(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# --------------------------------------------------------------------------- #
# Text decoder with cross-attention
# --------------------------------------------------------------------------- #
class CausalHead(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention."""

    def __init__(self, n_embd, n_head, head_size, block_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalHead(n_embd, head_size, block_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class CrossAttentionHead(nn.Module):
    """Query from x, key/value from context. No causal mask."""

    def __init__(self, n_embd, head_size, dropout=0.1):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.last_attn = None  # (B, T_x, T_c), stored for visualization

    def forward(self, x, context):
        q = self.query(x)        # (B, T_x, head_size)
        k = self.key(context)    # (B, T_c, head_size)
        v = self.value(context)  # (B, T_c, head_size)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)   # (B, T_x, T_c)
        wei = F.softmax(wei, dim=-1)
        self.last_attn = wei.detach()
        wei = self.dropout(wei)
        return wei @ v           # (B, T_x, head_size)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList(
            [CrossAttentionHead(n_embd, head_size, dropout) for _ in range(n_head)]
        )
        self.proj = nn.Linear(n_head * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context):
        out = torch.cat([h(x, context) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

    def attention_maps(self):
        """Stack per-head cross-attention weights: (B, n_head, T_x, T_c)."""
        return torch.stack([h.last_attn for h in self.heads], dim=1)


class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attn = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)  # causal
        self.cross_attn = MultiHeadCrossAttention(n_embd, n_head, head_size, dropout)          # not causal
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

    def forward(self, x, image_features):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), image_features)
        x = x + self.ffn(self.ln3(x))
        return x


class MultimodalModel(nn.Module):
    def __init__(self, vocab_size, n_embd=128, n_head=4, n_layer=4,
                 block_size=32, img_size=32, patch_size=4, dropout=0.1, pad_id=0):
        super().__init__()
        self.block_size = block_size
        self.pad_id = pad_id
        self.vision_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size,
                                         n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                                         dropout=dropout)
        self.token_embed = nn.Embedding(vocab_size, n_embd)
        self.pos_embed = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList(
            [DecoderBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, image, text_ids, targets=None):
        image_feats = self.vision_encoder(image)           # (B, 1+num_patches, n_embd)
        B, T = text_ids.shape
        tok = self.token_embed(text_ids)
        pos = self.pos_embed(torch.arange(T, device=text_ids.device))
        x = tok + pos
        for blk in self.blocks:
            x = blk(x, image_feats)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                ignore_index=self.pad_id,   # do not train on padding positions
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, image, bos_id, eos_id, max_new_tokens=None):
        """Greedy decode a caption for a single image (batch size 1)."""
        max_new_tokens = max_new_tokens or self.block_size
        idx = torch.tensor([[bos_id]], device=image.device)
        for _ in range(max_new_tokens - 1):
            logits, _ = self(image, idx[:, -self.block_size:])
            next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, next_id], dim=1)
            if next_id.item() == eos_id:
                break
        return idx[0].tolist()


# --------------------------------------------------------------------------- #
# Data: CIFAR-10 images paired with synthetic captions
# --------------------------------------------------------------------------- #
def get_loader(batch_size, smoke):
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tf)
    if smoke:
        train_set = Subset(train_set, range(512))
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)


def build_batch(images, labels, tok, device):
    """Return (images, x_text, y_text). x is caption[:-1], y is caption[1:]."""
    full = torch.stack([tok.encode_padded(int(l)) for l in labels])  # (B, block_size)
    x = full[:, :-1].to(device)
    y = full[:, 1:].to(device)
    return images.to(device), x, y


# --------------------------------------------------------------------------- #
# Train / visualize / sample
# --------------------------------------------------------------------------- #
def train(smoke=False, steps=None):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = CharTokenizer()
    max_steps = steps if steps is not None else (200 if smoke else MAX_STEPS)

    loader = get_loader(BATCH_SIZE, smoke)
    model = MultimodalModel(
        vocab_size=tok.vocab_size, n_embd=N_EMBD, n_head=N_HEAD, n_layer=N_LAYER,
        block_size=tok.block_size - 1,  # x has length block_size-1
        img_size=IMG_SIZE, patch_size=PATCH_SIZE, dropout=DROPOUT, pad_id=tok.pad_id,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"device={device}  vocab={tok.vocab_size}  block_size={tok.block_size}  "
          f"steps={max_steps}  params={sum(p.numel() for p in model.parameters())}")
    print(f"baseline loss ~ log(vocab) = {torch.log(torch.tensor(float(tok.vocab_size))):.3f}")

    step_hist, loss_hist = [], []
    data_iter = iter(loader)
    model.train()
    for step in range(max_steps + 1):
        try:
            images, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, labels = next(data_iter)
        images, x, y = build_batch(images, labels, tok, device)
        _, loss = model(images, x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % max(1, max_steps // 20) == 0 or step == max_steps:
            step_hist.append(step)
            loss_hist.append(loss.item())
            print(f"step {step:5d} | loss {loss.item():.4f}")

    save_training_curve(step_hist, loss_hist, os.path.join(HERE, "training_curve.png"))
    evaluate_and_sample(model, tok, device, smoke)
    return model, tok


def save_training_curve(steps, losses, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", markersize=3)
    plt.xlabel("step"); plt.ylabel("cross-entropy loss")
    plt.title("Task 4 — Multimodal caption training loss")
    plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


@torch.no_grad()
def evaluate_and_sample(model, tok, device, smoke):
    """Generate captions for one image per class, write samples.txt, save attention viz."""
    import torchvision
    import torchvision.transforms as T

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)

    # One representative image per class.
    per_class = {}
    for img, label in test_set:
        if label not in per_class:
            per_class[label] = img
        if len(per_class) == len(CIFAR10_CLASSES):
            break

    model.eval()
    lines, correct = [], 0
    viz_img = viz_caption_ids = None
    for cls in range(len(CIFAR10_CLASSES)):
        img = per_class[cls].unsqueeze(0).to(device)
        out_ids = model.generate(img, tok.bos_id, tok.eos_id)
        text = tok.decode(out_ids).replace(tok.BOS, "").replace(tok.EOS, "")
        target = f"this is a {CIFAR10_CLASSES[cls]}"
        ok = text.strip() == target
        correct += int(ok)
        lines.append(f"[{CIFAR10_CLASSES[cls]:11s}] -> '{text}'   ({'OK' if ok else 'x'})")
        if viz_img is None:
            viz_img, viz_caption_ids = img, out_ids

    acc = 100.0 * correct / len(CIFAR10_CLASSES)
    header = f"Task 4 — generated captions ({correct}/{len(CIFAR10_CLASSES)} = {acc:.0f}% exact match)\n"
    with open(os.path.join(HERE, "samples.txt"), "w", encoding="utf-8") as f:
        f.write(header + "=" * 60 + "\n" + "\n".join(lines) + "\n")
    print("\n" + header + "\n".join(lines))

    visualize_cross_attention(model, tok, viz_img, viz_caption_ids,
                              os.path.join(HERE, "attention_viz.png"))


@torch.no_grad()
def visualize_cross_attention(model, tok, image, caption_ids, path):
    """Heatmap: rows = generated caption tokens, cols = image context positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = torch.tensor([caption_ids], device=image.device)[:, :model.block_size]
    model(image, x)  # populates cross_attn.last_attn in each block

    # Average heads in the last decoder block: (T_x, T_c)
    maps = model.blocks[-1].cross_attn.attention_maps()  # (B, n_head, T_x, T_c)
    attn = maps[0].mean(0).cpu().numpy()                  # (T_x, T_c)
    token_labels = [tok.itos[i] for i in x[0].tolist()]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(attn, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(token_labels)))
    ax.set_yticklabels(token_labels)
    ax.set_xlabel("image context position (0 = CLS, 1..N = patches)")
    ax.set_ylabel("caption token (query)")
    ax.set_title("Task 4 — cross-attention (last decoder block, head-averaged)")
    fig.colorbar(im, ax=ax, fraction=0.025)
    plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Multimodal model: text decoder cross-attends to ViT.")
    p.add_argument("--smoke", action="store_true", help="fast run (200 steps, data subset)")
    p.add_argument("--steps", type=int, default=None, help="override training steps")
    args = p.parse_args()
    train(smoke=args.smoke, steps=args.steps)
