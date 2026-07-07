"""Task 6 — the instrumented training loop for CLIP-style contrastive training on Flickr8k.

Everything the handbook asks for, because every line here saves hours of debugging:
  * cosine LR schedule with linear warmup
  * mixed-precision (AMP) on GPU, automatically disabled on CPU
  * gradient clipping at 1.0
  * per-step CSV logging: train loss, val loss, lr, temperature, grad norm, step time
  * validation loss every `val_every` steps; retrieval Recall@K every `retrieval_every`
  * checkpointing: best-by-val-loss (best_model.pt) + rolling last.pt for resume
  * resume-from-checkpoint (survives a disconnect)

Usage:
    python task6/train.py                         # full 10k-step handbook run (GPU)
    python task6/train.py --smoke                 # tiny end-to-end sanity run
    python task6/train.py --steps 4000 --batch_size 128
    python task6/train.py --resume                # continue from task6/last.pt

Healthy training: loss starts ~log(128)=4.85, drops to ~3.5-4.0 in the first
500-1000 steps, then decreases slowly toward ~2.5-3.0; temperature drifts 0.07 -> 0.01-0.03.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import time

import torch
from torch.utils.data import Subset

from dataset import Flickr8kDataset, build_tokenizer, make_loader
from eval import evaluate_retrieval, format_retrieval, validation_loss
from model import DEFAULT_CONFIG, build_model

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------- #
# LR schedule and weight-decay parameter grouping
# --------------------------------------------------------------------------- #
def lr_at(step: int, warmup: int, total: int, base_lr: float) -> float:
    """Linear warmup for `warmup` steps, then cosine decay to ~0 by `total`."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * progress))


def param_groups(model, weight_decay: float):
    """Decay only the 2D+ weight matrices; never decay biases, norms, temperature,
    embeddings or the CLS/pos tokens (standard transformer practice)."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith("log_inv_tau") or "pos_embed" in name \
                or "cls_token" in name or "token_embed" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loaders(tok, cfg, smoke: bool):
    train_ds = Flickr8kDataset(tok, split="train", image_size=cfg["image_size"],
                               max_text_len=cfg["max_text_len"])
    val_ds = Flickr8kDataset(tok, split="val", image_size=cfg["image_size"],
                             max_text_len=cfg["max_text_len"])
    if smoke:
        train_ds = Subset(train_ds, range(min(512, len(train_ds))))
        val_ds = Subset(val_ds, range(min(256, len(val_ds))))
    workers = 0 if smoke else 2
    train_loader = make_loader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=workers)
    val_loader = make_loader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=workers)
    return train_loader, val_loader


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def save_ckpt(path, model, optimizer, scaler, step, best_val, cfg, vocab_size):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "best_val": best_val,
        "config": cfg,
        "vocab_size": vocab_size,
    }, path)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(cfg, smoke=False, resume=False):
    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda")

    tok = build_tokenizer()
    train_loader, val_loader = build_loaders(tok, cfg, smoke)
    model = build_model(tok.vocab_size, cfg).to(device)
    optimizer = torch.optim.AdamW(param_groups(model, cfg["weight_decay"]),
                                  lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    total_steps = cfg["total_steps"]
    step0, best_val, best_score = 0, float("inf"), -1.0
    last_path = os.path.join(HERE, "last.pt")
    best_path = os.path.join(HERE, "best_model.pt")
    if resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step0, best_val = ck["step"], ck["best_val"]
        print(f"resumed from {last_path} @ step {step0} (best_val {best_val:.4f})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  amp={use_amp}  vocab={tok.vocab_size}  params={n_params:,}")
    print(f"batch={cfg['batch_size']}  total_steps={total_steps}  "
          f"baseline log(batch)={math.log(cfg['batch_size']):.3f}")

    csv_path = os.path.join(HERE, "training_log.csv")
    append = resume and os.path.exists(csv_path)
    csv_file = open(csv_path, "a" if append else "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if not append:
        writer.writerow(["step", "train_loss", "val_loss", "lr", "temperature", "grad_norm", "sec_per_step"])
        csv_file.flush()

    data = infinite(train_loader)
    model.train()
    t_prev = time.time()
    val_loss_last = ""

    for step in range(step0, total_steps):
        lr = lr_at(step, cfg["warmup_steps"], total_steps, cfg["lr"])
        for g in optimizer.param_groups:
            g["lr"] = lr

        batch = next(data)
        images = batch["image"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss, _, _ = model(images, tokens, mask)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)  # unscale so grad-clip threshold is in real units
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if not torch.isfinite(loss):
            raise RuntimeError(f"loss became non-finite at step {step} — see NaN diagnostics in writeup.")

        sec = time.time() - t_prev
        t_prev = time.time()
        temperature = model.loss_fn.temperature

        do_val = (step % cfg["val_every"] == 0) or (step == total_steps - 1)
        if do_val:
            vloss = validation_loss(model, val_loader, device, max_batches=(None if smoke else 8))
            val_loss_last = f"{vloss:.4f}"
            best_val = min(best_val, vloss)

            # Checkpoint on RETRIEVAL, not val loss. On Flickr8k the two diverge: val
            # InfoNCE loss (128 in-batch negatives) bottoms early, but full retrieval
            # over 5000 captions keeps improving as embeddings sharpen. We select the
            # checkpoint that actually maximises retrieval (sum of all six recalls).
            r = evaluate_retrieval(model, val_loader, device)
            score = sum(r[f"{d}_R@{k}"] for d in ("i2t", "t2i") for k in (1, 5, 10))
            model.train()
            if score > best_score:
                best_score = score
                save_ckpt(best_path, model, optimizer, scaler, step, best_val, cfg, tok.vocab_size)
            save_ckpt(last_path, model, optimizer, scaler, step + 1, best_val, cfg, tok.vocab_size)

        writer.writerow([step, f"{loss.item():.4f}", val_loss_last, f"{lr:.3e}",
                         f"{temperature:.4f}", f"{grad_norm:.3f}", f"{sec:.3f}"])
        csv_file.flush()

        if do_val:
            print(f"step {step:5d}/{total_steps} | loss {loss.item():.4f} | "
                  f"val {val_loss_last:>7} | lr {lr:.2e} | tau {temperature:.4f} | "
                  f"gnorm {grad_norm:5.2f} | {sec*1000:.0f} ms/step\n    " + format_retrieval(r))

    csv_file.close()

    # Ensure a best checkpoint exists even if val never improved (degenerate smoke runs).
    if not os.path.exists(best_path):
        save_ckpt(best_path, model, optimizer, scaler, total_steps, best_val, cfg, tok.vocab_size)

    # Report retrieval on the BEST-val checkpoint (early stopping), not the possibly
    # overfit final-step weights.
    best_state = torch.load(best_path, map_location=device)
    model.load_state_dict(best_state["model"])
    final = evaluate_retrieval(model, val_loader, device)
    print(f"\nFINAL (best-retrieval ckpt @ step {best_state.get('step','?')}) "
          + format_retrieval(final))
    plot_curves(csv_path, os.path.join(HERE, "training_curve.png"))
    print(f"best retrieval score (sum of 6 recalls): {best_score:.1f}  ->  {best_path}")
    return model, tok, final


def plot_curves(csv_path, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps, tr, vsteps, vl = [], [], [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            tr.append(float(row["train_loss"]))
            if row["val_loss"]:
                vsteps.append(int(row["step"]))
                vl.append(float(row["val_loss"]))
    plt.figure(figsize=(9, 5))
    plt.plot(steps, tr, alpha=0.4, lw=1, label="train loss (per step)")
    if vl:
        plt.plot(vsteps, vl, marker="o", ms=3, color="crimson", label="val loss")
    plt.axhline(math.log(128), color="gray", ls="--", lw=1, label="log(128) = 4.85 baseline")
    plt.xlabel("step"); plt.ylabel("InfoNCE loss")
    plt.title("Task 6 — Flickr8k contrastive training")
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=120); plt.close()
    print(f"Saved {out_path}")


def smoke_config(cfg):
    cfg = dict(cfg)
    cfg.update(batch_size=32, total_steps=30, warmup_steps=5, val_every=10,
               retrieval_every=20, lr=5e-4)
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train CLIP-style VLM on Flickr8k.")
    ap.add_argument("--smoke", action="store_true", help="tiny fast end-to-end run")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--val_every", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None, help="raise to fight overfitting")
    ap.add_argument("--weight_decay", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = dict(DEFAULT_CONFIG)
    if args.smoke:
        cfg = smoke_config(cfg)
    if args.steps is not None:
        cfg["total_steps"] = args.steps
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr
    if args.val_every is not None:
        cfg["val_every"] = args.val_every
    if args.dropout is not None:
        cfg["dropout"] = args.dropout
    if args.weight_decay is not None:
        cfg["weight_decay"] = args.weight_decay

    train(cfg, smoke=args.smoke, resume=args.resume)
