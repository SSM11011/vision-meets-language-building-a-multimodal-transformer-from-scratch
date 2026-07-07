"""Task 6 — model builder + default config.

Reuses the Task 5 CLIP-style model (two independent towers meeting only at the InfoNCE
loss) with the handbook's Task 6 hyperparameters. We add task5/ to sys.path so the
from-scratch encoders/loss are shared verbatim rather than duplicated.
"""
from __future__ import annotations

import os
import sys

_TASK5 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "task5")
if _TASK5 not in sys.path:
    sys.path.insert(0, _TASK5)

from clip_model import CLIPStyleModel  # noqa: E402  (needs sys.path tweak first)

# Handbook Task-6 starting hyperparameters (expect to tune). Model + optim + schedule
# knobs live together so a checkpoint fully documents how it was trained.
DEFAULT_CONFIG = {
    # data / optimisation
    "batch_size": 128,
    "lr": 5e-4,
    "weight_decay": 0.05,
    "warmup_steps": 500,
    "total_steps": 10000,
    "grad_clip": 1.0,
    "val_every": 200,
    # model
    "projection_dim": 128,
    "embed_dim": 192,
    "image_size": 64,
    "patch_size": 8,          # 8x8 patches -> 64 tokens per image
    "max_text_len": 32,
    "vit_depth": 4,
    "text_depth": 4,
    "n_head": 6,
    "dropout": 0.1,
    "pad_id": 0,
    "init_temperature": 0.07,
}


def build_model(vocab_size: int, cfg: dict) -> CLIPStyleModel:
    return CLIPStyleModel.from_config(vocab_size, cfg)
