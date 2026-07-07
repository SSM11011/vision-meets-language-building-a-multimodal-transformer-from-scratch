"""Task 5 — the full CLIP-style model: two towers meeting only at the InfoNCE loss.

Pipeline (cross-attention deliberately dropped -- that is for conditional tasks like
captioning; for retrieval/alignment two independent towers + contrastive loss are
simpler and stronger):

    image -> ImageEncoder -> CLS token -> image_proj -> (B, D_proj)   [L2-norm in loss]
    text  -> TextEncoder  -> masked mean-pool -> text_proj -> (B, D_proj)
    InfoNCE(image_embed, text_embed)

Projection heads are bias-free linear maps into a smaller alignment space
(projection_dim=128); the encoders keep the richer 192-d features for any downstream
reuse (SimCLR's "evaluate before the projection" finding).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders import ImageEncoder, TextEncoder
from loss import InfoNCELoss


class CLIPStyleModel(nn.Module):
    def __init__(self, vit_encoder: ImageEncoder, text_encoder: TextEncoder,
                 embed_dim: int = 192, projection_dim: int = 128,
                 init_temperature: float = 0.07):
        super().__init__()
        self.vit = vit_encoder
        self.text = text_encoder
        self.image_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, projection_dim, bias=False)
        self.loss_fn = InfoNCELoss(init_temperature=init_temperature)

    # ---- factory: build both towers from a config dict in one call ---------- #
    @classmethod
    def from_config(cls, vocab_size: int, cfg: dict) -> "CLIPStyleModel":
        vit = ImageEncoder(
            image_size=cfg["image_size"], patch_size=cfg["patch_size"],
            embed_dim=cfg["embed_dim"], n_head=cfg["n_head"],
            depth=cfg["vit_depth"], dropout=cfg["dropout"],
        )
        text = TextEncoder(
            vocab_size=vocab_size, max_text_len=cfg["max_text_len"],
            embed_dim=cfg["embed_dim"], n_head=cfg["n_head"],
            depth=cfg["text_depth"], dropout=cfg["dropout"], pad_id=cfg.get("pad_id", 0),
        )
        return cls(vit, text, embed_dim=cfg["embed_dim"], projection_dim=cfg["projection_dim"],
                   init_temperature=cfg.get("init_temperature", 0.07))

    # ---- encoding ----------------------------------------------------------- #
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """(B, 3, H, W) -> (B, D_proj) projected CLS embedding (un-normalized)."""
        feats = self.vit.encode(images)     # (B, 1 + num_patches, D)
        cls = feats[:, 0]                   # (B, D)
        return self.image_proj(cls)         # (B, D_proj)

    def encode_image_patches(self, images: torch.Tensor) -> torch.Tensor:
        """Projected PATCH embeddings (excludes CLS): (B, num_patches, D_proj).

        Used by the Task-6 qualitative similarity map, which dots each patch against
        the caption embedding to localize where the caption 'lands' on the image."""
        feats = self.vit.encode(images)     # (B, 1 + num_patches, D)
        patches = feats[:, 1:]              # (B, num_patches, D)
        return self.image_proj(patches)     # (B, num_patches, D_proj)

    def encode_text(self, text_tokens: torch.Tensor, text_mask: torch.Tensor | None = None) -> torch.Tensor:
        """(B, T) -> (B, D_proj) projected, mask-aware mean-pooled embedding."""
        feats = self.text.encode(text_tokens, mask=text_mask)   # (B, T, D)
        if text_mask is not None:
            m = text_mask.unsqueeze(-1).to(feats.dtype)         # (B, T, 1)
            pooled = (feats * m).sum(1) / m.sum(1).clamp(min=1e-6)
        else:
            pooled = feats.mean(1)
        return self.text_proj(pooled)                           # (B, D_proj)

    # ---- training forward --------------------------------------------------- #
    def forward(self, images, text_tokens, text_mask=None):
        img_e = self.encode_image(images)
        txt_e = self.encode_text(text_tokens, text_mask)
        loss = self.loss_fn(img_e, txt_e)
        return loss, img_e, txt_e


# --------------------------------------------------------------------------- #
# Dummy-data verification (handbook: "same drill as Task 4 Day 3").
# --------------------------------------------------------------------------- #
def _verify() -> None:
    import math

    torch.manual_seed(0)
    cfg = dict(image_size=64, patch_size=8, embed_dim=192, projection_dim=128,
               n_head=6, vit_depth=4, text_depth=4, dropout=0.1, max_text_len=32, pad_id=0)
    vocab, B = 500, 16
    model = CLIPStyleModel.from_config(vocab, cfg)

    images = torch.randn(B, 3, 64, 64)
    tokens = torch.randint(1, vocab, (B, 32))       # avoid pad_id=0 so all real
    mask = torch.ones(B, 32, dtype=torch.long)

    loss, img_e, txt_e = model(images, tokens, mask)
    print(f"params        : {sum(p.numel() for p in model.parameters()):,}")
    print(f"image embed   : {tuple(img_e.shape)}   text embed: {tuple(txt_e.shape)}")
    print(f"patch embeds  : {tuple(model.encode_image_patches(images).shape)}")
    print(f"init loss     : {loss.item():.4f}   (expect ~log(B) = {math.log(B):.4f})")

    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, f"parameters with no gradient: {missing}"
    print(f"gradients     : flow to all {sum(1 for _ in model.parameters())} parameter tensors  OK")
    assert abs(loss.item() - math.log(B)) < 0.6, "init loss should be near log(B)"
    print("VERIFY OK")


if __name__ == "__main__":
    _verify()
