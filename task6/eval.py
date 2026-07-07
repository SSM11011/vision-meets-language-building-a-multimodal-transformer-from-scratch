"""Task 6 — retrieval evaluation (Recall@K) and validation loss.

Training loss alone does not tell you if alignment works; retrieval does. On the val
split (1,000 images x 5 captions = 5,000 captions):

  * Image -> Text: for each image, rank all 5,000 captions by cosine similarity. It is
    a hit@K if ANY of that image's 5 ground-truth captions is in the top-K.
  * Text -> Image: for each caption, rank all 1,000 images. Hit@K if the caption's own
    source image is in the top-K.

Random-chance Recall@1 for 1,000 images is 0.1%, so anything well above that means the
model learned something. Expected for a small 64x64 model: I->T R@1 ~15-25%.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def _gather_embeddings(model, loader, device):
    """Return per-caption image/text embeddings (normalized) and image ids."""
    model.eval()
    img_list, txt_list, id_list = [], [], []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        img_e = F.normalize(model.encode_image(images), dim=-1)
        txt_e = F.normalize(model.encode_text(tokens, mask), dim=-1)
        img_list.append(img_e.cpu())
        txt_list.append(txt_e.cpu())
        id_list.append(batch["image_id"])
    return torch.cat(img_list), torch.cat(txt_list), torch.cat(id_list)


@torch.no_grad()
def evaluate_retrieval(model, loader, device, ks=(1, 5, 10)) -> dict:
    """Compute Recall@K in both directions. Returns a flat dict of percentages."""
    img_all, txt_all, ids = _gather_embeddings(model, loader, device)

    # Deduplicate images: keep the first embedding seen for each image_id.
    seen: set[int] = set()
    order = []
    for pos, i in enumerate(ids.tolist()):
        if i not in seen:
            seen.add(i)
            order.append(pos)
    image_embeds = img_all[order]                       # (N_img, D)
    image_ids = ids[order]                              # (N_img,)
    row_of_image = {int(u): r for r, u in enumerate(image_ids.tolist())}
    text_embeds = txt_all                               # (N_txt, D)
    text_image_row = torch.tensor([row_of_image[int(i)] for i in ids.tolist()])  # (N_txt,)

    n_img, n_txt = image_embeds.size(0), text_embeds.size(0)
    max_k = max(ks)
    sim = image_embeds @ text_embeds.t()               # (N_img, N_txt)

    # ---- Image -> Text: each image has multiple ground-truth captions --------
    # Ground-truth caption mask (N_img, N_txt): 1 where the caption's image == this image.
    gt = (text_image_row.unsqueeze(0) == torch.arange(n_img).unsqueeze(1))  # (N_img, N_txt)
    topk_txt = sim.topk(min(max_k, n_txt), dim=1).indices                   # (N_img, max_k)
    gt_in_topk = torch.gather(gt, 1, topk_txt)                              # (N_img, max_k) bool
    i2t = {}
    for k in ks:
        hit = gt_in_topk[:, :k].any(dim=1).float().mean().item()
        i2t[k] = 100.0 * hit

    # ---- Text -> Image: each caption has exactly one correct image -----------
    sim_t = sim.t()                                     # (N_txt, N_img)
    topk_img = sim_t.topk(min(max_k, n_img), dim=1).indices                 # (N_txt, max_k)
    correct = text_image_row.unsqueeze(1)                                   # (N_txt, 1)
    t2i_hits = (topk_img == correct)                                        # (N_txt, max_k)
    t2i = {}
    for k in ks:
        hit = t2i_hits[:, :k].any(dim=1).float().mean().item()
        t2i[k] = 100.0 * hit

    result = {"n_img": n_img, "n_txt": n_txt}
    for k in ks:
        result[f"i2t_R@{k}"] = i2t[k]
        result[f"t2i_R@{k}"] = t2i[k]
    return result


@torch.no_grad()
def validation_loss(model, loader, device, max_batches: int | None = None) -> float:
    """Average InfoNCE loss over the validation loader (same objective as training)."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        loss, _, _ = model(images, tokens, mask)
        total += loss.item()
        n += 1
    return total / max(1, n)


def format_retrieval(r: dict) -> str:
    return (f"Retrieval ({r['n_img']} imgs / {r['n_txt']} caps)  "
            f"I->T R@1/5/10 = {r['i2t_R@1']:.1f}/{r['i2t_R@5']:.1f}/{r['i2t_R@10']:.1f}  |  "
            f"T->I R@1/5/10 = {r['t2i_R@1']:.1f}/{r['t2i_R@5']:.1f}/{r['t2i_R@10']:.1f}")


if __name__ == "__main__":
    # Smoke check on random embeddings: recall should be ~chance.
    import argparse
    import os

    from dataset import Flickr8kDataset, build_tokenizer, make_loader
    from model import build_model, DEFAULT_CONFIG

    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "best_model.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = build_tokenizer()
    val = Flickr8kDataset(tok, split="val")
    loader = make_loader(val, batch_size=128, shuffle=False, num_workers=2)
    model = build_model(tok.vocab_size, DEFAULT_CONFIG).to(device)
    if os.path.exists(args.ckpt):
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state["model"])
        print(f"loaded {args.ckpt} (step {state.get('step','?')})")
    print(format_retrieval(evaluate_retrieval(model, loader, device)))
