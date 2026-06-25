"""Build (and execute) task0/tensors.ipynb — the Task 0 Part B tensor warmup.

This script constructs the notebook programmatically with nbformat, executes it
with nbclient so every output is embedded, and writes tensors.ipynb next to it.

Run:
    python task0/build_tensors_notebook.py
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tensors.ipynb")


def md(text):
    return new_markdown_cell(text)


def code(src):
    return new_code_cell(src)


cells = []

cells.append(md(
    "# Task 0 — Part B: Tensor Warmup\n"
    "\n"
    "Six problems, solved with **no `nn.Module` and no library shortcuts** — only "
    "broadcasting, masking, and basic tensor ops. The goal is fluency with every tool "
    "used in the rest of the project.\n"
))

cells.append(code(
    "import torch\n"
    "import torch.nn.functional as F\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "torch.manual_seed(0)\n"
    "print('torch', torch.__version__)\n"
))

# ---- Problem 1 ----
cells.append(md(
    "## Problem 1 — Masked mean\n"
    "\n"
    "Given `x` of shape `(B, T, C)` and `y` of shape `(B, T)`, compute the mean of `x` "
    "across `T`, but only at positions where `y == 1`. Output `(B, C)`. "
    "No Python loops — pure broadcasting + masking.\n"
))
cells.append(code(
    "def masked_mean(x, y):\n"
    "    # x: (B, T, C), y: (B, T) with 0/1 entries -> (B, C)\n"
    "    mask = (y == 1).to(x.dtype)            # (B, T)\n"
    "    masked = x * mask.unsqueeze(-1)        # (B, T, C), zero out unwanted positions\n"
    "    num = masked.sum(dim=1)                # (B, C)\n"
    "    den = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1), avoid div-by-zero\n"
    "    return num / den\n"
    "\n"
    "B, T, C = 4, 6, 3\n"
    "x = torch.randn(B, T, C)\n"
    "y = (torch.rand(B, T) > 0.5).long()\n"
    "out = masked_mean(x, y)\n"
    "print('output shape:', tuple(out.shape))\n"
    "\n"
    "# Verify against an explicit per-batch loop (reference only).\n"
    "ref = torch.stack([\n"
    "    x[b][y[b] == 1].mean(0) if (y[b] == 1).any() else torch.zeros(C)\n"
    "    for b in range(B)\n"
    "])\n"
    "print('matches loop reference:', torch.allclose(out, ref, atol=1e-6))\n"
))

# ---- Problem 2 ----
cells.append(md(
    "## Problem 2 — Softmax from scratch\n"
    "\n"
    "Implement softmax using only `exp`, `sum`, and arithmetic; match `torch.softmax` to 1e-6.\n"
))
cells.append(code(
    "def softmax(x, dim=-1):\n"
    "    # Subtract the max for numerical stability (see writeup below).\n"
    "    x_max = x.max(dim=dim, keepdim=True).values\n"
    "    e = torch.exp(x - x_max)\n"
    "    return e / e.sum(dim=dim, keepdim=True)\n"
    "\n"
    "z = torch.randn(5, 7)\n"
    "mine = softmax(z, dim=-1)\n"
    "ref = torch.softmax(z, dim=-1)\n"
    "print('max abs diff vs torch.softmax:', (mine - ref).abs().max().item())\n"
    "print('matches to 1e-6:', torch.allclose(mine, ref, atol=1e-6))\n"
    "\n"
    "# Demonstrate the stability problem: naive softmax on large logits -> NaN.\n"
    "big = torch.tensor([1000.0, 1001.0, 1002.0])\n"
    "naive = torch.exp(big) / torch.exp(big).sum()\n"
    "print('naive softmax on large logits:', naive)        # nan, nan, nan\n"
    "print('stable softmax on large logits:', softmax(big))\n"
))
cells.append(md(
    "**Why naive softmax is numerically unstable, and the fix.**\n"
    "\n"
    "Softmax computes `exp(x_i) / sum_j exp(x_j)`. Floating point can only represent numbers "
    "up to about `3.4e38` (float32). If any logit is large — say `1000` — then `exp(1000)` "
    "overflows to `+inf`, and `inf / inf = NaN`, poisoning the whole output. Very negative "
    "logits underflow to `0`, which is less catastrophic but loses precision.\n"
    "\n"
    "The standard fix is to subtract the per-row maximum before exponentiating: "
    "`softmax(x) = softmax(x - max(x))`. This is **mathematically identical** because the "
    "constant `exp(-max)` factors out of numerator and denominator and cancels. But now the "
    "largest exponent is `exp(0) = 1`, so nothing overflows. This is the same idea as the "
    "**log-sum-exp** trick, which reappears in the InfoNCE loss later in the project.\n"
))

# ---- Problem 3 ----
cells.append(md(
    "## Problem 3 — Attention scores two ways\n"
    "\n"
    "Given `Q, K` of shape `(B, T, d)`, compute the scores `(B, T, T)` with `einsum` and with "
    "`@` + transpose, and verify they are equal.\n"
))
cells.append(code(
    "B, T, d = 2, 5, 8\n"
    "Q = torch.randn(B, T, d)\n"
    "K = torch.randn(B, T, d)\n"
    "\n"
    "scores_einsum = torch.einsum('btd,bsd->bts', Q, K)\n"
    "scores_matmul = Q @ K.transpose(-2, -1)\n"
    "print('shapes:', tuple(scores_einsum.shape), tuple(scores_matmul.shape))\n"
    "print('max abs diff:', (scores_einsum - scores_matmul).abs().max().item())\n"
    "print('exactly equal:', torch.equal(scores_einsum, scores_matmul))\n"
))

# ---- Problem 4 ----
cells.append(md(
    "## Problem 4 — Causal mask\n"
    "\n"
    "Build a `(T, T)` mask that is `0` on and below the diagonal and `-inf` above it. Add it to "
    "an attention score matrix before softmax, then visualize the post-softmax attention for `T=8`.\n"
))
cells.append(code(
    "T = 8\n"
    "# 1s strictly above the diagonal -> those become -inf.\n"
    "causal_mask = torch.triu(torch.ones(T, T), diagonal=1)\n"
    "causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))\n"
    "print('mask (0 = visible, -inf = blocked):')\n"
    "print(causal_mask)\n"
    "\n"
    "scores = torch.randn(T, T)\n"
    "attn = torch.softmax(scores + causal_mask, dim=-1)\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(4, 4))\n"
    "im = ax.imshow(attn.detach(), cmap='viridis')\n"
    "ax.set_title('Post-softmax causal attention (T=8)')\n"
    "ax.set_xlabel('key position'); ax.set_ylabel('query position')\n"
    "fig.colorbar(im, ax=ax, fraction=0.046)\n"
    "plt.tight_layout(); plt.show()\n"
    "\n"
    "print('each row sums to 1:', torch.allclose(attn.sum(-1), torch.ones(T), atol=1e-6))\n"
    "print('upper triangle is zero:', torch.allclose(attn.triu(1), torch.zeros(T, T), atol=1e-6))\n"
))

# ---- Problem 5 ----
cells.append(md(
    "## Problem 5 — LayerNorm from scratch\n"
    "\n"
    "Normalize over the `C` dimension (per token, per batch), then apply learnable scale `gamma` "
    "and shift `beta`. Match `nn.LayerNorm` to 1e-5.\n"
))
cells.append(code(
    "def layer_norm(x, gamma, beta, eps=1e-5):\n"
    "    # x: (B, T, C); normalize over the last (feature) dim only.\n"
    "    mean = x.mean(dim=-1, keepdim=True)\n"
    "    var = x.var(dim=-1, unbiased=False, keepdim=True)  # population variance (matches nn)\n"
    "    x_hat = (x - mean) / torch.sqrt(var + eps)\n"
    "    return gamma * x_hat + beta\n"
    "\n"
    "B, T, C = 2, 4, 16\n"
    "x = torch.randn(B, T, C)\n"
    "gamma = torch.randn(C)   # learnable scale (random here to test the general case)\n"
    "beta = torch.randn(C)    # learnable shift\n"
    "\n"
    "mine = layer_norm(x, gamma, beta)\n"
    "\n"
    "ln = torch.nn.LayerNorm(C, eps=1e-5)\n"
    "with torch.no_grad():\n"
    "    ln.weight.copy_(gamma)\n"
    "    ln.bias.copy_(beta)\n"
    "ref = ln(x)\n"
    "print('max abs diff vs nn.LayerNorm:', (mine - ref).abs().max().item())\n"
    "print('matches to 1e-5:', torch.allclose(mine, ref, atol=1e-5))\n"
))

# ---- Problem 6 ----
cells.append(md(
    "## Problem 6 — Manual gradients\n"
    "\n"
    "First `y = (x**2).sum()`: by hand `dy/dx_i = 2 x_i`. Then `y = softmax(x).sum()` and explain "
    "the shape of its gradient.\n"
))
cells.append(code(
    "x = torch.randn(4, requires_grad=True)\n"
    "y = (x ** 2).sum()\n"
    "y.backward()\n"
    "print('autograd grad :', x.grad)\n"
    "print('manual  2*x   :', 2 * x.detach())\n"
    "print('match         :', torch.allclose(x.grad, 2 * x.detach(), atol=1e-6))\n"
))
cells.append(code(
    "x2 = torch.randn(4, requires_grad=True)\n"
    "y2 = torch.softmax(x2, dim=0).sum()\n"
    "y2.backward()\n"
    "print('softmax(x).sum() =', y2.item(), '(always 1.0 — probabilities sum to 1)')\n"
    "print('grad =', x2.grad, '(all ~0)')\n"
    "print('grad is ~zero:', torch.allclose(x2.grad, torch.zeros(4), atol=1e-6))\n"
))
cells.append(md(
    "**Why the softmax gradient is (almost exactly) zero.**\n"
    "\n"
    "`softmax(x).sum()` is identically `1` for *every* input `x` — the probabilities always sum "
    "to one. A function that is constant has zero gradient everywhere, so `x.grad` is `0` "
    "(up to floating-point noise).\n"
    "\n"
    "More formally, the softmax Jacobian is `∂p_i/∂x_j = p_i (δ_ij − p_j)`. The gradient of the "
    "sum `L = Σ_i p_i` is `∂L/∂x_j = Σ_i p_i (δ_ij − p_j) = p_j − p_j Σ_i p_i = p_j − p_j·1 = 0`. "
    "The two terms cancel exactly — which is the formal reason the gradient vanishes. We will "
    "use this same Jacobian identity in the Task 2 attention-gradient derivation.\n"
))

cells.append(md(
    "---\n"
    "All six problems solved with outputs above. ✅\n"
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
})


def main():
    try:
        from nbclient import NotebookClient
        client = NotebookClient(nb, timeout=300, kernel_name="python3")
        client.execute()
        print("Executed notebook (outputs embedded).")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not execute notebook ({exc}). Saving without outputs.")
    with open(OUT, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
