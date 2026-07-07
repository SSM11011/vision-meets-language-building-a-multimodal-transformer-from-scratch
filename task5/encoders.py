"""Task 5 — the two independent encoders for CLIP-style contrastive training.

In the CLIP recipe the image and text towers never talk to each other until the loss:
each maps its input to a sequence of D-dim token features, and a pooling step turns
that sequence into one vector. We reuse exactly the transformer machinery built in
Tasks 2-4 (from scratch, no nn.MultiheadAttention), with two clarity refactors:

  * attention is batched across heads with reshapes (fast on GPU) instead of a
    Python list of single-head modules;
  * self-attention takes an optional key-padding mask so padded text positions do
    not leak into real tokens.

Both encoders expose `.encode(...)` returning the FULL token sequence (B, T, D):
  * ImageEncoder: (B, 1 + num_patches, D)  -- index 0 is the CLS token.
  * TextEncoder:  (B, T_text, D).

The CLS token / pooling and projection heads live in clip_model.py, so these
encoders stay reusable (e.g. Task 6's captioning stretch reuses ImageEncoder).

Handbook defaults: embed_dim=192, image_size=64, patch_size=8 (-> 64 patches),
n_head=6, depth=4, dropout=0.1, max_text_len=32.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Bidirectional multi-head self-attention with an optional key-padding mask."""

    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        if dim % n_head != 0:
            raise ValueError(f"dim ({dim}) must be divisible by n_head ({n_head})")
        self.n_head = n_head
        self.head_dim = dim // n_head
        self.scale = self.head_dim ** -0.5
        # Single fused projection for q, k, v (bias-free, as in ViT/CLIP towers).
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.last_attn: torch.Tensor | None = None  # (B, n_head, T, T), for viz

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                       # each (B, n_head, T, head_dim)

        wei = (q @ k.transpose(-2, -1)) * self.scale          # (B, n_head, T, T)
        if key_padding_mask is not None:
            # key_padding_mask: (B, T) with 1 = real token, 0 = padding. Block attention
            # *into* padded keys so real tokens never absorb padding information.
            block = (key_padding_mask == 0)[:, None, None, :]  # (B, 1, 1, T)
            wei = wei.masked_fill(block, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        self.last_attn = wei.detach()
        wei = self.attn_drop(wei)

        out = (wei @ v).transpose(1, 2).reshape(B, T, C)      # (B, T, C)
        return self.proj_drop(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (self-attn + MLP), bidirectional."""

    def __init__(self, dim: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, n_head, dropout)
        self.ln2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dropout)

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class ImageEncoder(nn.Module):
    """Vision Transformer tower. `encode(images)` -> (B, 1 + num_patches, D)."""

    def __init__(self, image_size: int = 64, patch_size: int = 8, in_chans: int = 3,
                 embed_dim: int = 192, n_head: int = 6, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size ({image_size}) must be divisible by patch_size ({patch_size})")
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_head, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        B = images.size(0)
        x = self.patch_embed(images).flatten(2).transpose(1, 2)  # (B, num_patches, D)
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, D)
        x = torch.cat([cls, x], dim=1)                           # (B, 1 + num_patches, D)
        x = self.dropout(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)

    # Alias so the tower is drop-in for code that calls it like a module.
    forward = encode


class TextEncoder(nn.Module):
    """Token transformer tower. `encode(tokens, mask)` -> (B, T, D)."""

    def __init__(self, vocab_size: int, max_text_len: int = 32,
                 embed_dim: int = 192, n_head: int = 6, depth: int = 4,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.max_text_len = max_text_len
        self.pad_id = pad_id
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_text_len, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, n_head, dropout) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.token_embed.weight, std=0.02)
        with torch.no_grad():                       # keep padding row at zero
            self.token_embed.weight[pad_id].zero_()

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = tokens.shape
        if T > self.max_text_len:
            raise ValueError(f"sequence length {T} exceeds max_text_len {self.max_text_len}")
        x = self.token_embed(tokens) + self.pos_embed[:, :T]
        x = self.dropout(x)
        for blk in self.blocks:
            x = blk(x, key_padding_mask=mask)
        return self.norm(x)

    forward = encode
