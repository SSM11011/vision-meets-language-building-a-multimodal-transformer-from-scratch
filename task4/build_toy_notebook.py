"""Build (and execute) task4/cross_attention_toy.ipynb — Task 4 Part B.

Sanity-checks multi-head cross-attention on toy tensors: shape checks, an
attention heatmap, and a small training loop proving cross-attention can learn
to route information from a specific context position to a specific query
position.

Run:
    python task4/build_toy_notebook.py
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "cross_attention_toy.ipynb")

cells = []

cells.append(new_markdown_cell(
    "# Task 4 — Part B: Cross-Attention on Toy Data\n"
    "\n"
    "Cross-attention is self-attention with one change: the **query** comes from one source "
    "(`x`) and the **keys/values** come from another (`context`). Here we verify shapes, "
    "visualize attention weights, and run a tiny training loop showing the mechanism learns "
    "to route information from a specific context position to a specific query position.\n"
))

cells.append(new_code_cell(
    "import torch\n"
    "import torch.nn as nn\n"
    "import torch.nn.functional as F\n"
    "import matplotlib.pyplot as plt\n"
    "torch.manual_seed(0)\n"
))

cells.append(new_markdown_cell("## The cross-attention modules"))
cells.append(new_code_cell(
    "class CrossAttentionHead(nn.Module):\n"
    "    def __init__(self, n_embd, head_size):\n"
    "        super().__init__()\n"
    "        self.key   = nn.Linear(n_embd, head_size, bias=False)\n"
    "        self.query = nn.Linear(n_embd, head_size, bias=False)\n"
    "        self.value = nn.Linear(n_embd, head_size, bias=False)\n"
    "        self.last_attn = None\n"
    "\n"
    "    def forward(self, x, context):\n"
    "        q = self.query(x)        # (B, T_x, head_size)\n"
    "        k = self.key(context)    # (B, T_c, head_size)\n"
    "        v = self.value(context)  # (B, T_c, head_size)\n"
    "        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # (B, T_x, T_c)\n"
    "        wei = F.softmax(wei, dim=-1)\n"
    "        self.last_attn = wei.detach()\n"
    "        return wei @ v           # (B, T_x, head_size)\n"
    "\n"
    "class MultiHeadCrossAttention(nn.Module):\n"
    "    def __init__(self, n_embd, n_head, head_size):\n"
    "        super().__init__()\n"
    "        self.heads = nn.ModuleList([CrossAttentionHead(n_embd, head_size)\n"
    "                                    for _ in range(n_head)])\n"
    "        self.proj = nn.Linear(n_head * head_size, n_embd)\n"
    "\n"
    "    def forward(self, x, context):\n"
    "        out = torch.cat([h(x, context) for h in self.heads], dim=-1)\n"
    "        return self.proj(out)\n"
))

cells.append(new_markdown_cell("## Step 1-2 — Shapes\n\n"
    "`x: (2, 10, 64)` (query length 10), `context: (2, 5, 64)` (context length 5). "
    "Output should be `(2, 10, 64)` — same length as the query."))
cells.append(new_code_cell(
    "x = torch.randn(2, 10, 64)\n"
    "context = torch.randn(2, 5, 64)\n"
    "mha = MultiHeadCrossAttention(n_embd=64, n_head=4, head_size=16)\n"
    "out = mha(x, context)\n"
    "print('x       :', tuple(x.shape))\n"
    "print('context :', tuple(context.shape))\n"
    "print('output  :', tuple(out.shape), '(== query length, not context length)')\n"
    "assert out.shape == (2, 10, 64)\n"
))

cells.append(new_markdown_cell("## Step 3 — Attention heatmap (one head)\n\n"
    "Rows = query positions (0-9), columns = context positions (0-4). Each row sums to 1."))
cells.append(new_code_cell(
    "attn = mha.heads[0].last_attn[0]  # (T_x, T_c) for batch element 0\n"
    "print('row sums (should all be 1):', attn.sum(-1))\n"
    "fig, ax = plt.subplots(figsize=(4, 6))\n"
    "im = ax.imshow(attn, cmap='viridis', aspect='auto')\n"
    "ax.set_xlabel('context position'); ax.set_ylabel('query position')\n"
    "ax.set_title('Cross-attention (head 0, untrained)')\n"
    "fig.colorbar(im, ax=ax, fraction=0.046); plt.tight_layout(); plt.show()\n"
))

cells.append(new_markdown_cell(
    "## Step 4 — Structured routing test\n"
    "\n"
    "We set up a copy task: the cue vector at query position 5 is made identical to the vector "
    "at context position 2 (`x[:, 5] = context[:, 2]`), and the target output at position 5 is "
    "that same vector. **Crucially we redraw fresh random tensors every training step**, so the "
    "value at position 2 changes constantly — the model cannot memorize a fixed answer with its "
    "value/projection weights. The only way to drive the loss down is to genuinely *route*: "
    "match the query against the keys, find that position 2 is the match, and copy its value. "
    "After training, query position 5 should attend strongly to context position 2.\n"
    "\n"
    "(A single fixed example would let the model cheat by baking the answer into its weights "
    "without ever sharpening the attention — randomizing the batch each step removes that "
    "shortcut.)"
))
cells.append(new_code_cell(
    "torch.manual_seed(0)\n"
    "model = MultiHeadCrossAttention(n_embd=64, n_head=4, head_size=16)\n"
    "\n"
    "def make_batch(B=16):\n"
    "    x = torch.randn(B, 10, 64)\n"
    "    context = torch.randn(B, 5, 64)\n"
    "    x[:, 5] = context[:, 2]            # cue at query 5 matches context position 2\n"
    "    target = context[:, 2].clone()    # output at position 5 should reproduce it\n"
    "    return x, context, target\n"
    "\n"
    "def q5_attention():\n"
    "    # Mean over heads and batch of the attention from query position 5: shape (T_c,)\n"
    "    return torch.stack([h.last_attn[:, 5] for h in model.heads]).mean(0).mean(0)\n"
    "\n"
    "with torch.no_grad():\n"
    "    xb, cb, _ = make_batch()\n"
    "    model(xb, cb)\n"
    "    before = q5_attention()\n"
    "print('BEFORE  attention from q5 over context:', before.numpy().round(3))\n"
    "print('        weight on position 2 =', round(before[2].item(), 3),\n"
    "      '| argmax =', before.argmax().item())\n"
))
cells.append(new_code_cell(
    "opt = torch.optim.Adam(model.parameters(), lr=1e-3)\n"
    "for step in range(2000):\n"
    "    xb, cb, tb = make_batch()\n"
    "    out = model(xb, cb)\n"
    "    loss = F.mse_loss(out[:, 5], tb)\n"
    "    opt.zero_grad(); loss.backward(); opt.step()\n"
    "print('final loss:', round(loss.item(), 6))\n"
    "\n"
    "with torch.no_grad():\n"
    "    xb, cb, _ = make_batch()\n"
    "    model(xb, cb)\n"
    "    after = q5_attention()\n"
    "print('AFTER   attention from q5 over context:', after.numpy().round(3))\n"
    "print('        weight on position 2 =', round(after[2].item(), 3),\n"
    "      '| argmax =', after.argmax().item())\n"
    "print('position 5 now strongly attends to position 2:',\n"
    "      after.argmax().item() == 2 and after[2].item() > 0.5)\n"
))
cells.append(new_code_cell(
    "fig, axes = plt.subplots(1, 2, figsize=(8, 3))\n"
    "for ax, w, title in zip(axes, [before, after], ['before training', 'after 2000 steps']):\n"
    "    ax.bar(range(5), w.numpy())\n"
    "    ax.set_title(f'q5 attention ({title})'); ax.set_xlabel('context position')\n"
    "    ax.set_ylim(0, 1)\n"
    "plt.tight_layout(); plt.show()\n"
))

cells.append(new_markdown_cell(
    "---\n"
    "**Conclusion.** Cross-attention can learn to route information from a chosen context "
    "position to a chosen query position. Because the answer always lives at context position 2 "
    "(and the values are re-randomized each step), query position 5's attention mass "
    "concentrates on position 2 after training — confirming the mechanism works before we wire "
    "it into the real multimodal model.\n"
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
})


def main():
    try:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=600, kernel_name="python3").execute()
        print("Executed notebook (outputs embedded).")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not execute notebook ({exc}). Saving without outputs.")
    with open(OUT, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
