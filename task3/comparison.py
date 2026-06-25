"""Task 3 — Train both the CNN baseline and the ViT, then plot their validation
accuracy on the same axes (task3/comparison_plot.png).

Run:
    python task3/comparison.py             # full runs (slow on CPU)
    python task3/comparison.py --smoke      # fast end-to-end sanity run
    python task3/comparison.py --cnn-epochs 10 --vit-epochs 30
"""
from __future__ import annotations

import argparse
import os

from cnn_baseline import train_cnn
from vit import train_vit

HERE = os.path.dirname(os.path.abspath(__file__))


def save_comparison(cnn_val, vit_val, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cnn_val) + 1), cnn_val, marker="o", label="TinyCNN val acc")
    plt.plot(range(1, len(vit_val) + 1), vit_val, marker="s", label="ViT val acc")
    plt.xlabel("epoch"); plt.ylabel("validation accuracy (%)")
    plt.title("CIFAR-10: TinyCNN vs ViT")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


def main(smoke=False, cnn_epochs=None, vit_epochs=None):
    print("===== Training TinyCNN =====")
    _, cnn_val = train_cnn(smoke=smoke, epochs=cnn_epochs)
    print("\n===== Training ViT =====")
    _, vit_val = train_vit(smoke=smoke, epochs=vit_epochs)
    save_comparison(cnn_val, vit_val, os.path.join(HERE, "comparison_plot.png"))
    print(f"\nFinal val acc — CNN: {cnn_val[-1]:.2f}%  ViT: {vit_val[-1]:.2f}%")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compare CNN and ViT on CIFAR-10.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--cnn-epochs", type=int, default=None)
    p.add_argument("--vit-epochs", type=int, default=None)
    args = p.parse_args()
    main(smoke=args.smoke, cnn_epochs=args.cnn_epochs, vit_epochs=args.vit_epochs)
