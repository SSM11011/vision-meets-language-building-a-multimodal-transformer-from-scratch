"""Task 6 — qualitative similarity maps ("where does the caption land on the image?").

For a trained model, each image patch has a projected embedding. Dotting every patch
against the caption's projected embedding gives a per-patch similarity that we reshape
to the 8x8 patch grid and upsample over the image as a heatmap. Patches the caption
"describes" should light up.

We show successes (the model's best-matching caption in a candidate pool is one of the
image's own) and, deliberately, a couple of failure cases — those are often more
informative.

    python task6/qualitative.py            # uses task6/best_model.pt
Saves task6/qualitative.png.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import Flickr8kDataset, build_tokenizer, make_loader
from model import DEFAULT_CONFIG, build_model
from preprocess import load_cache

HERE = os.path.dirname(os.path.abspath(__file__))
_MEAN = np.array([0.485, 0.456, 0.406])
_STD = np.array([0.229, 0.224, 0.225])


def _tokenize_caption(tok, caption, max_len):
    ids = tok.encode(caption)[:max_len]
    pad = max_len - len(ids)
    mask = [1] * len(ids) + [0] * pad
    ids = ids + [tok.PAD] * pad
    return torch.tensor([ids]), torch.tensor([mask])


@torch.no_grad()
def make_figure(n_examples=8, out_path=os.path.join(HERE, "qualitative.png")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DEFAULT_CONFIG
    tok = build_tokenizer()

    val = Flickr8kDataset(tok, split="val", image_size=cfg["image_size"],
                          max_text_len=cfg["max_text_len"])
    model = build_model(tok.vocab_size, cfg).to(device)
    ckpt = os.path.join(HERE, "best_model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print(f"loaded {ckpt}")
    else:
        print("WARNING: no best_model.pt — showing an untrained model (maps will be noise).")
    model.eval()

    # Candidate caption pool to judge success/failure (first 400 val captions).
    pool_loader = make_loader(val, batch_size=128, shuffle=False, num_workers=0)
    pool_caps, pool_embeds, pool_imgids = [], [], []
    for batch in pool_loader:
        te = F.normalize(model.encode_text(batch["tokens"].to(device),
                                           batch["mask"].to(device)), dim=-1).cpu()
        pool_embeds.append(te)
        pool_caps.extend(batch["caption"])
        pool_imgids.extend(batch["image_id"].tolist())
        if len(pool_caps) >= 400:
            break
    pool_embeds = torch.cat(pool_embeds)[:400]
    pool_caps = pool_caps[:400]
    pool_imgids = pool_imgids[:400]

    cache, cache_index = load_cache()
    grid = cfg["image_size"] // cfg["patch_size"]  # 8

    # Pick distinct images; classify each as success/failure by nearest-caption.
    picks, seen = [], set()
    failures = []
    for idx in range(len(val)):
        fn, caption = val.pairs[idx]
        if fn in seen:
            continue
        seen.add(fn)
        img_t = val[idx]["image"].unsqueeze(0).to(device)
        img_e = F.normalize(model.encode_image(img_t), dim=-1).cpu()  # (1, D)
        sims = (img_e @ pool_embeds.t())[0]
        best = int(sims.argmax())
        own_id = val.image_id_of[fn]
        success = pool_imgids[best] == own_id
        entry = (idx, fn, caption, success, pool_caps[best])
        if success and len([p for p in picks if p[3]]) < n_examples - 2:
            picks.append(entry)
        elif not success and len(failures) < 2:
            failures.append(entry)
        if len([p for p in picks if p[3]]) >= n_examples - 2 and len(failures) >= 2:
            break
    picks = picks + failures
    picks = picks[:n_examples]

    cols = 4
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4.3 * rows))
    axes = np.array(axes).reshape(-1)

    for ax, (idx, fn, caption, success, best_cap) in zip(axes, picks):
        # Raw 64x64 image for display.
        if cache is not None and fn in cache_index:
            disp = np.asarray(cache[cache_index[fn]]) / 255.0
        else:
            t = val[idx]["image"]
            disp = np.clip(t.permute(1, 2, 0).numpy() * _STD + _MEAN, 0, 1)

        img_t = val[idx]["image"].unsqueeze(0).to(device)
        tokens, mask = _tokenize_caption(tok, caption, cfg["max_text_len"])
        patch_e = F.normalize(model.encode_image_patches(img_t)[0], dim=-1)      # (64, D)
        cap_e = F.normalize(model.encode_text(tokens.to(device), mask.to(device))[0], dim=-1)  # (D,)
        sim = (patch_e @ cap_e).cpu().numpy().reshape(grid, grid)               # (8, 8)
        sim = (sim - sim.min()) / (np.ptp(sim) + 1e-8)                          # normalize 0..1

        ax.imshow(disp)
        ax.imshow(sim, cmap="jet", alpha=0.5, extent=[0, disp.shape[1], disp.shape[0], 0],
                  interpolation="bilinear")
        tag = "OK" if success else "MISS"
        ax.set_title(f"[{tag}] {caption[:42]}", fontsize=8)
        ax.axis("off")
    for ax in axes[len(picks):]:
        ax.axis("off")

    fig.suptitle("Task 6 — patch-caption similarity maps (jet = high similarity to caption)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}  ({len(picks)} examples: "
          f"{sum(p[3] for p in picks)} success, {sum(1 - p[3] for p in picks)} failure)")


if __name__ == "__main__":
    make_figure()
