"""Task 6 — Applied Stretch (Application 2): text-to-image search engine.

Precompute an embedding for every Flickr8k image once, then answer a natural-language
query by encoding it with the text tower and returning the top-K images by cosine
similarity. This is the core algorithm behind Google Images / phone photo search /
e-commerce visual search — minus the FAISS-scale approximate nearest-neighbour index.

    python task6/text_to_image_search.py                       # 3 demo queries -> PNG
    python task6/text_to_image_search.py --query "two dogs playing" --interactive

Saves task6/text_to_image_search.png (a grid of example queries + their top-5).
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from dataset import DATA_DIR
from model import DEFAULT_CONFIG, build_model
from preprocess import load_cache
from dataset import build_tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

DEMO_QUERIES = [
    "a dog running on the beach",
    "a group of people climbing a mountain",
    "a child playing in the snow",
]


@torch.no_grad()
def build_image_index(model, device, batch_size=256):
    """Encode every cached image once. Returns (filenames, normalized embeds (N, D))."""
    cache, index = load_cache()
    if cache is None:
        raise FileNotFoundError("Run task6/preprocess.py first to build the image cache.")
    to_tensor = T.Compose([T.ToTensor(), T.Normalize(_MEAN, _STD)])
    filenames = list(index.keys())

    embeds = []
    for i in range(0, len(filenames), batch_size):
        chunk = filenames[i:i + batch_size]
        imgs = torch.stack([to_tensor(np.array(cache[index[fn]])) for fn in chunk]).to(device)
        e = F.normalize(model.encode_image(imgs), dim=-1).cpu()
        embeds.append(e)
    return filenames, torch.cat(embeds)


@torch.no_grad()
def search(model, tok, device, filenames, image_embeds, query, top_k=5):
    max_len = DEFAULT_CONFIG["max_text_len"]
    ids = tok.encode(query)[:max_len]
    pad = max_len - len(ids)
    mask = torch.tensor([[1] * len(ids) + [0] * pad])
    tokens = torch.tensor([ids + [tok.PAD] * pad])
    q = F.normalize(model.encode_text(tokens.to(device), mask.to(device)), dim=-1).cpu()[0]
    scores = image_embeds @ q
    top = scores.topk(top_k).indices.tolist()
    return [(filenames[i], float(scores[i])) for i in top]


def _load_model(device):
    tok = build_tokenizer()
    model = build_model(tok.vocab_size, DEFAULT_CONFIG).to(device)
    ckpt = os.path.join(HERE, "best_model.pt")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
        print(f"loaded {ckpt}")
    else:
        print("WARNING: no best_model.pt — results will be random.")
    model.eval()
    return model, tok


def demo(queries, out_path=os.path.join(HERE, "text_to_image_search.png"), top_k=5):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok = _load_model(device)
    filenames, image_embeds = build_image_index(model, device)
    print(f"indexed {len(filenames)} images")

    cache, index = load_cache()
    rows = len(queries)
    fig, axes = plt.subplots(rows, top_k, figsize=(2.4 * top_k, 2.7 * rows))
    axes = np.array(axes).reshape(rows, top_k)
    for r, query in enumerate(queries):
        results = search(model, tok, device, filenames, image_embeds, query, top_k)
        for c, (fn, score) in enumerate(results):
            ax = axes[r, c]
            ax.imshow(np.asarray(cache[index[fn]]))
            ax.set_title(f"{score:.2f}", fontsize=8)
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(query[:24], fontsize=9)
        axes[r, 0].text(-0.1, 0.5, query, transform=axes[r, 0].transAxes,
                        rotation=90, va="center", ha="right", fontsize=9)
        print(f"query {query!r} -> {[fn for fn, _ in results]}")
    fig.suptitle("Task 6 — text-to-image search (top-5 per query)", fontsize=12)
    plt.tight_layout(rect=[0.03, 0, 1, 0.96])
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", type=str, default=None)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--interactive", action="store_true", help="prompt for queries in a loop")
    args = ap.parse_args()

    if args.interactive:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, tok = _load_model(device)
        filenames, image_embeds = build_image_index(model, device)
        print(f"indexed {len(filenames)} images. Type a query (blank to quit).")
        while True:
            q = input("query> ").strip()
            if not q:
                break
            for fn, s in search(model, tok, device, filenames, image_embeds, q, args.top_k):
                print(f"  {s:.3f}  {fn}")
    else:
        demo([args.query] if args.query else DEMO_QUERIES, top_k=args.top_k)


if __name__ == "__main__":
    main()
