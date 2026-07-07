"""Task 5 — toy alignment experiment: does InfoNCE actually cause alignment?

This is the cheapest possible "is my training pipeline correct?" check. It strips out
the encoders entirely: we invent 32 fixed random "image features" and 32 independent
fixed random "text features", then train ONLY two linear projection heads (into a
128-d space) with the InfoNCE loss. If the loss implementation is correct, the
projections learn to send matching pairs close together and everything else apart —
loss falls from log(32) ~= 3.47 toward ~0 and the similarity matrix becomes
diagonal-dominant.

If this works but Task 6 fails, the bug is in the data/encoders, not the loss.

    python task5/toy_alignment.py            # 500 steps (handbook)
    python task5/toy_alignment.py --steps 1000

Artifacts -> task5/: toy_loss_curve.png, toy_similarity.png
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn as nn

from loss import InfoNCELoss

HERE = os.path.dirname(os.path.abspath(__file__))
N_PAIRS = 32
FEATURE_DIM = 192
PROJ_DIM = 128
SEED = 1337


class ToyAligner(nn.Module):
    """Two independent linear projections + the shared InfoNCE loss."""

    def __init__(self, feature_dim=FEATURE_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.img_proj = nn.Linear(feature_dim, proj_dim, bias=False)
        self.txt_proj = nn.Linear(feature_dim, proj_dim, bias=False)
        self.loss_fn = InfoNCELoss(init_temperature=0.07)

    def forward(self, image_feats, text_feats):
        return self.loss_fn(self.img_proj(image_feats), self.txt_proj(text_feats))

    @torch.no_grad()
    def similarity_matrix(self, image_feats, text_feats):
        import torch.nn.functional as F
        pi = F.normalize(self.img_proj(image_feats), dim=-1)
        pt = F.normalize(self.txt_proj(text_feats), dim=-1)
        return pi @ pt.t()


def run(steps: int = 500) -> dict:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Fixed, independent "image" and "text" features. Independence matters: if they
    # shared structure the task would be trivial for reasons unrelated to the loss.
    image_feats = torch.randn(N_PAIRS, FEATURE_DIM, device=device)
    text_feats = torch.randn(N_PAIRS, FEATURE_DIM, device=device)

    model = ToyAligner().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    baseline = math.log(N_PAIRS)
    print(f"device={device}  pairs={N_PAIRS}  baseline log(N)={baseline:.4f}")

    step_hist, loss_hist = [], []
    for step in range(steps + 1):
        loss = model(image_feats, text_feats)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == steps:
            step_hist.append(step)
            loss_hist.append(loss.item())
            if step % 100 == 0 or step == steps:
                print(f"step {step:4d} | loss {loss.item():.4f} | tau {model.loss_fn.temperature:.4f}")

    sim = model.similarity_matrix(image_feats, text_feats).cpu()
    diag = sim.diagonal()
    off = sim[~torch.eye(N_PAIRS, dtype=torch.bool)]
    stats = {
        "final_loss": loss_hist[-1],
        "avg_diag": diag.mean().item(),
        "avg_offdiag": off.mean().item(),
        "diag_min": diag.min().item(),
        "offdiag_max": off.max().item(),
    }
    print(f"\nfinal loss           : {stats['final_loss']:.4f}  (from baseline {baseline:.4f})")
    print(f"avg diagonal sim     : {stats['avg_diag']:.4f}  (want -> ~1)")
    print(f"avg off-diagonal sim : {stats['avg_offdiag']:.4f}  (want -> low/negative)")
    print(f"worst diagonal       : {stats['diag_min']:.4f}   best off-diagonal: {stats['offdiag_max']:.4f}")

    _plot_loss(step_hist, loss_hist, baseline, os.path.join(HERE, "toy_loss_curve.png"))
    _plot_similarity(sim.numpy(), os.path.join(HERE, "toy_similarity.png"))

    # Correctness gates: the loss MUST drive real alignment.
    assert stats["final_loss"] < 0.1, f"toy loss did not converge: {stats['final_loss']}"
    assert stats["avg_diag"] - stats["avg_offdiag"] > 0.5, "diagonal not clearly dominant"
    print("\nTOY ALIGNMENT OK — InfoNCE implementation causes alignment.")
    return stats


def _plot_loss(steps, losses, baseline, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 5))
    plt.plot(steps, losses, marker="o", markersize=3, label="InfoNCE loss")
    plt.axhline(baseline, color="crimson", ls="--", lw=1, label=f"log(N) = {baseline:.2f}")
    plt.axhline(0.0, color="gray", ls=":", lw=1)
    plt.xlabel("step"); plt.ylabel("loss")
    plt.title("Task 5 — toy alignment: InfoNCE loss over training")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


def _plot_similarity(sim, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(sim, cmap="viridis", vmin=-1, vmax=1)
    ax.set_xlabel("text index"); ax.set_ylabel("image index")
    ax.set_title("Task 5 — learned similarity matrix\n(diagonal should light up)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Toy InfoNCE alignment experiment.")
    p.add_argument("--steps", type=int, default=500)
    args = p.parse_args()
    run(steps=args.steps)
