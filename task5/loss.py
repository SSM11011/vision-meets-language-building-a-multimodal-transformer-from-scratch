"""Task 5 — InfoNCE (CLIP) contrastive loss, implemented from scratch.

Given a batch of N image embeddings and N text embeddings where row i of each is a
matching pair, InfoNCE asks the model to make image_i most similar to text_i out of
all N texts (and vice-versa). Mechanically this is cross-entropy over a similarity
matrix whose correct target is the diagonal.

Numerical-stability choices, all of which matter (see the handbook's "three things
people get wrong"):

  1. L2-normalize both sets of embeddings before the dot product, so similarity is
     cosine (direction only) and the model cannot cheat by rescaling magnitudes.
  2. Learn a *log* inverse-temperature parameter, as CLIP does. Optimising in log
     space keeps temperature strictly positive and makes its gradient well-behaved.
  3. Clamp log(1/tau) to [0, log(100)] so temperature stays in [0.01, 1.0] and can
     never collapse the softmax into all-or-nothing (which destroys gradients).

Run `python task5/loss.py` for a tiny self-check; see test_loss.py for the full suite.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """Symmetric InfoNCE loss with a learnable, clamped log inverse-temperature.

    log_inv_tau is parameterised as log(1/tau). It is initialised to log(1/0.07)
    (~2.659) exactly as in CLIP, and clamped to [0, log(100)] on every forward pass.
    """

    LOG_INV_TAU_MAX = math.log(100.0)  # ~4.6052 -> tau floor of 0.01

    def __init__(self, init_temperature: float = 0.07):
        super().__init__()
        if not 0.0 < init_temperature <= 1.0:
            raise ValueError(f"init_temperature must be in (0, 1], got {init_temperature}")
        # Store log(1/tau) so gradient descent operates in a stable, positive space.
        self.log_inv_tau = nn.Parameter(torch.tensor(1.0 / init_temperature).log())

    @property
    def temperature(self) -> float:
        """Current (clamped) temperature tau, for logging."""
        log_inv_tau = self.log_inv_tau.detach().clamp(0.0, self.LOG_INV_TAU_MAX)
        return float((-log_inv_tau).exp())

    def forward(self, image_embeds: torch.Tensor, text_embeds: torch.Tensor) -> torch.Tensor:
        """image_embeds, text_embeds: (N, D). Returns a scalar loss.

        Embeddings need NOT be pre-normalized; normalization happens here so the
        loss is self-contained and impossible to misuse.
        """
        if image_embeds.shape != text_embeds.shape:
            raise ValueError(
                f"image/text embeddings must have identical shape, got "
                f"{tuple(image_embeds.shape)} vs {tuple(text_embeds.shape)}"
            )
        if image_embeds.dim() != 2:
            raise ValueError(f"expected (N, D) embeddings, got {image_embeds.dim()}D")

        # 1. Project every embedding onto the unit sphere (cosine similarity).
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # 2. Clamp log-temperature for stability, then exponentiate to get 1/tau.
        log_inv_tau = self.log_inv_tau.clamp(0.0, self.LOG_INV_TAU_MAX)
        inv_tau = log_inv_tau.exp()

        # 3. Scaled cosine-similarity matrix (N, N); logits[i, j] = sim(img_i, txt_j)/tau.
        logits = inv_tau * image_embeds @ text_embeds.t()

        # 4. Targets are the diagonal: pair i is (image_i, text_i).
        n = image_embeds.size(0)
        labels = torch.arange(n, device=logits.device)

        # 5. Symmetric loss: images-as-queries + texts-as-queries, averaged (CLIP loss).
        loss_i2t = F.cross_entropy(logits, labels)   # rank texts for each image
        loss_t2i = F.cross_entropy(logits.t(), labels)  # rank images for each text
        return 0.5 * (loss_i2t + loss_t2i)


if __name__ == "__main__":
    torch.manual_seed(0)
    n, d = 128, 192
    loss_fn = InfoNCELoss()

    same = torch.randn(n, d)
    print(f"identical embeds  -> loss {loss_fn(same, same).item():.4f}  (expect ~0)")

    a, b = torch.randn(n, d), torch.randn(n, d)
    print(f"random embeds     -> loss {loss_fn(a, b).item():.4f}  (expect ~log(N)={math.log(n):.4f})")
    print(f"init temperature  -> {loss_fn.temperature:.4f}  (expect 0.07)")
