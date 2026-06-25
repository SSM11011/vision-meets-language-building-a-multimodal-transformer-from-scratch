"""Task 3, Part B — TinyCNN vision baseline on CIFAR-10.

Builds the exact CNN from the handbook and trains it on CIFAR-10 to establish a
baseline that ViT will be compared against. Saves training/validation accuracy
curves to task3/cnn_curves.png.

Run:
    python task3/cnn_baseline.py             # full: 10 epochs
    python task3/cnn_baseline.py --smoke     # fast: 1 epoch on a data subset
    python task3/cnn_baseline.py --epochs 5

Expected (full run): ~65-70% validation accuracy.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(HERE, "data")

# ----- Handbook hyperparameters (Task 3 Part B) -----
LEARNING_RATE = 1e-3
EPOCHS = 10
BATCH_SIZE = 64
SEED = 1337


class TinyCNN(nn.Module):
    """Three conv blocks (each followed by 2x2 max-pool) + two FC layers."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 32x32 -> 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 16x16 -> 8x8
        x = self.pool(F.relu(self.conv3(x)))   # 8x8   -> 4x4
        x = x.flatten(1)                       # (B, 128*4*4)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def get_loaders(batch_size: int, smoke: bool):
    import torchvision
    import torchvision.transforms as T
    from torch.utils.data import DataLoader, Subset

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),  # CIFAR-10 stats
    ])
    train_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True, transform=tf)
    test_set = torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)
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


def train_cnn(smoke: bool = False, epochs: int | None = None):
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = epochs if epochs is not None else (1 if smoke else EPOCHS)

    train_loader, test_loader = get_loaders(BATCH_SIZE, smoke)
    model = TinyCNN().to(device)
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

    save_curves(train_acc_hist, val_acc_hist, os.path.join(HERE, "cnn_curves.png"))
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
    plt.title("TinyCNN on CIFAR-10")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="TinyCNN on CIFAR-10.")
    p.add_argument("--smoke", action="store_true", help="fast run (1 epoch, 512-image subset)")
    p.add_argument("--epochs", type=int, default=None, help="override number of epochs")
    args = p.parse_args()
    train_cnn(smoke=args.smoke, epochs=args.epochs)
