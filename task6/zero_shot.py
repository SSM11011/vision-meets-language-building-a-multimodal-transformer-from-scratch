"""Task 6 — Applied Stretch (Application 3): zero-shot image classification.

This is exactly zero-shot CLIP: with NO labels and NO fine-tuning, we encode a set of
class names as text ("a photo of a dog"), encode an image, and predict the class whose
text embedding is most similar to the image embedding. We also demonstrate prompt
ensembling (averaging several templates per class), a real deployed-CLIP technique.

We evaluate on Flickr8k val images, deriving a weak ground-truth label for each image
from keywords in its human captions (an image counts for class C if C's keyword appears
in any of its 5 captions and no other class's does). This gives an honest, if noisy,
accuracy signal without any hand-labelling.

    python task6/zero_shot.py
Saves task6/zero_shot_confusion.png and prints per-class accuracy.
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from dataset import Flickr8kDataset, build_tokenizer, make_loader
from model import DEFAULT_CONFIG, build_model

HERE = os.path.dirname(os.path.abspath(__file__))

# Classes present in Flickr8k, with caption keywords used to derive weak labels.
CLASSES = {
    "dog":      ["dog", "puppy", "dogs"],
    "child":    ["child", "boy", "girl", "kid", "children", "toddler"],
    "water":    ["water", "ocean", "lake", "river", "pool", "beach", "sea"],
    "snow":     ["snow", "snowy", "ski", "skier", "sled"],
    "bike":     ["bike", "bicycle", "cyclist", "biker"],
    "mountain": ["mountain", "cliff", "rocky", "hill"],
}
CLASS_NAMES = list(CLASSES)

# Prompt templates for ensembling.
TEMPLATES = ["a photo of a {}", "a picture of a {}", "an image of a {}", "a {} in a scene"]


def weak_label(captions: list[str]) -> int | None:
    """Return a class index if exactly one class's keywords appear across the captions."""
    text = " ".join(captions).lower()
    hits = [ci for ci, c in enumerate(CLASS_NAMES)
            if any(f" {kw} " in f" {text} " for kw in CLASSES[c])]
    return hits[0] if len(hits) == 1 else None


@torch.no_grad()
def _class_text_embeddings(model, tok, device, ensemble: bool):
    """One normalized text embedding per class (optionally averaged over templates)."""
    max_len = DEFAULT_CONFIG["max_text_len"]
    embeds = []
    for c in CLASS_NAMES:
        prompts = [t.format(c) for t in TEMPLATES] if ensemble else [f"a photo of a {c}"]
        vecs = []
        for p in prompts:
            ids = tok.encode(p)[:max_len]
            pad = max_len - len(ids)
            m = torch.tensor([[1] * len(ids) + [0] * pad])
            t = torch.tensor([ids + [tok.PAD] * pad])
            vecs.append(F.normalize(model.encode_text(t.to(device), m.to(device)), dim=-1))
        v = torch.stack(vecs).mean(0)
        embeds.append(F.normalize(v, dim=-1))          # renormalize the averaged prompt
    return torch.cat(embeds)                            # (n_class, D)


@torch.no_grad()
def run(ensemble: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer()
    val = Flickr8kDataset(tok, split="val", image_size=DEFAULT_CONFIG["image_size"],
                          max_text_len=DEFAULT_CONFIG["max_text_len"])
    model = build_model(tok.vocab_size, DEFAULT_CONFIG).to(device)
    ckpt = os.path.join(HERE, "best_model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print(f"loaded {ckpt}")
    model.eval()

    # Weak ground-truth: one representative caption-set per image.
    caps_by_image: dict[int, list[str]] = {}
    fn_by_image: dict[int, str] = {}
    for fn, cap in val.pairs:
        iid = val.image_id_of[fn]
        caps_by_image.setdefault(iid, []).append(cap)
        fn_by_image[iid] = fn

    labelled = [(iid, weak_label(caps)) for iid, caps in caps_by_image.items()]
    labelled = [(iid, y) for iid, y in labelled if y is not None]
    print(f"weak-labelled images: {len(labelled)} across {len(CLASS_NAMES)} classes")

    class_text = _class_text_embeddings(model, tok, device, ensemble)  # (C, D)

    # Encode each labelled image once (val transform, no aug).
    import numpy as np
    idx_of_first = {}
    for i, (fn, _) in enumerate(val.pairs):
        idx_of_first.setdefault(val.image_id_of[fn], i)

    y_true, y_pred = [], []
    for iid, y in labelled:
        img_t = val[idx_of_first[iid]]["image"].unsqueeze(0).to(device)
        img_e = F.normalize(model.encode_image(img_t), dim=-1)
        pred = int((img_e @ class_text.t())[0].argmax())
        y_true.append(y); y_pred.append(pred)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    acc = (y_true == y_pred).mean() * 100
    print(f"\nzero-shot accuracy ({'ensemble' if ensemble else 'single prompt'}): {acc:.1f}%")
    for ci, c in enumerate(CLASS_NAMES):
        m = y_true == ci
        if m.sum():
            print(f"  {c:9s} n={int(m.sum()):3d}  acc {100*(y_pred[m]==ci).mean():5.1f}%")

    _plot_confusion(y_true, y_pred, os.path.join(HERE, "zero_shot_confusion.png"), acc)
    return acc


def _plot_confusion(y_true, y_pred, path, acc):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(CLASS_NAMES)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("weak ground-truth")
    ax.set_title(f"Task 6 — zero-shot confusion (acc {acc:.1f}%)")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout(); plt.savefig(path, dpi=120); plt.close()
    print(f"Saved {path}")


if __name__ == "__main__":
    print("=== single prompt ===")
    run(ensemble=False)
    print("\n=== prompt ensemble ===")
    run(ensemble=True)
