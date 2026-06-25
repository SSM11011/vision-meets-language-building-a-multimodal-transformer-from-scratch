"""Task 3, Part E — Vision Transformer (ViT) from scratch on CIFAR-10.

Reuses the Task 2 transformer block, with ONE change: no causal mask. Every patch
attends to every other patch (bidirectional attention). Pipeline:

    image -> Conv2d patch embedding -> prepend CLS token -> + position embedding
          -> N bidirectional transformer blocks -> LayerNorm -> classify from CLS

Run:
    python task3/vit.py             # full: 30 epochs
    python task3/vit.py --smoke     # fast: 1 epoch on a data subset
    python task3/vit.py --epochs 10

Expected (full run): ~65-72% validation accuracy on CIFAR-10.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")

# ----- Handbook hyperparameters (Task 3 Part E) -----
PATCH_SIZE = 4
N_EMBD = 192
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.1
LEARNING_RATE = 3e-4
EPOCHS = 30
BATCH_SIZE = 128
SEED = 1337


class Head(nn.Module):
    """One head of *bidirectional* self-attention (no causal mask)."""

    def __init__(self, n_embd, head_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)   # (B, T, T)
        wei = F.softmax(wei, dim=-1)                           # NO mask: every patch sees all
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, dropout) for _ in range(n_head)])
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
            nn.GELU(),                      # GELU is conventional for ViT
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BlockBidirectional(nn.Module):
    """Pre-norm transformer block with bidirectional (unmasked) attention."""

    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.attn = MultiHeadAttention(n_embd, n_head, head_size, dropout)
        self.ffn = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, num_classes=10,
                 n_embd=192, n_head=6, n_layer=6, dropout=0.1):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # Conv2d with kernel=stride=patch_size is a non-overlapping patch projector.
        self.patch_embed = nn.Conv2d(in_chans, n_embd, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, n_embd))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, n_embd))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [BlockBidirectional(n_embd, n_head, dropout) for _ in range(n_layer)]
        )
        self.norm = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)                 # (B, n_embd, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)        # (B, num_patches, n_embd)
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, n_embd)
        x = torch.cat([cls, x], dim=1)          # (B, 1+num_patches, n_embd)
        x = x + self.pos_embed
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_final = x[:, 0]                      # read from CLS token only
        return self.head(cls_final)


def get_loaders(batch_size: int, smoke: bool):
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=test_tf)
    if smoke:
        train_set = Subset(train_set, range(512))
        test_set = Subset(test_set, range(512))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    model.train()
    return 100.0 * correct / total


def train_vit(smoke: bool = False, epochs: int | None = None):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = epochs if epochs is not None else (1 if smoke else EPOCHS)

    train_loader, test_loader = get_loaders(BATCH_SIZE, smoke)
    model = ViT(patch_size=PATCH_SIZE, n_embd=N_EMBD, n_head=N_HEAD,
                n_layer=N_LAYER, dropout=DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"device={device}  epochs={epochs}  params={sum(p.numel() for p in model.parameters())}")

    train_acc_hist, val_acc_hist = [], []
    for epoch in range(1, epochs + 1):
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item()
        train_acc = evaluate(model, train_loader, device, max_batches=20)  # subset for speed
        val_acc = evaluate(model, test_loader, device)
        train_acc_hist.append(train_acc)
        val_acc_hist.append(val_acc)
        print(f"epoch {epoch:2d} | loss {running / len(train_loader):.4f} | "
              f"train acc {train_acc:.2f}% | val acc {val_acc:.2f}%")

    save_curves(train_acc_hist, val_acc_hist, os.path.join(HERE, "vit_curves.png"))
    return train_acc_hist, val_acc_hist


def save_curves(train_acc, val_acc, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(train_acc) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_acc, marker="o", label="train acc")
    plt.plot(epochs, val_acc, marker="s", label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy (%)")
    plt.title("ViT on CIFAR-10")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Vision Transformer on CIFAR-10.")
    p.add_argument("--smoke", action="store_true", help="fast run (1 epoch, 512-image subset)")
    p.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    args = p.parse_args()
    train_vit(smoke=args.smoke, epochs=args.epochs)
