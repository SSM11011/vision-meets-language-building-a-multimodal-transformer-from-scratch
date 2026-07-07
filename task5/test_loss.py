"""Task 5 — sanity tests for InfoNCELoss (handbook Day-2 checks).

These are deliberately isolated from any encoder: they feed hand-crafted embeddings
whose expected loss we can reason about analytically. Run directly:

    python task5/test_loss.py

Each check prints its measured value and PASS/FAIL. The script exits non-zero if any
check fails, so it doubles as a CI-style gate. (Also importable under pytest.)
"""
from __future__ import annotations

import math

import torch

from loss import InfoNCELoss


def _loss_no_grad(image_embeds, text_embeds, temperature=0.07):
    """Loss with the temperature FIXED at init (no optimisation), for predictability."""
    with torch.no_grad():
        return InfoNCELoss(init_temperature=temperature)(image_embeds, text_embeds).item()


def test_identical_embeddings_low_loss():
    """Test 1: identical image/text embeddings are already perfectly aligned.

    Expected: a small loss. With tau=0.07 the diagonal logit is 1/0.07 ~= 14.3 while
    off-diagonals are cosine sims of random 192-d vectors (~0) scaled by 14.3, so the
    softmax puts almost all mass on the diagonal -> loss near zero (not exactly zero,
    since random off-diagonals are not perfectly orthogonal)."""
    torch.manual_seed(0)
    x = torch.randn(128, 192)
    loss = _loss_no_grad(x, x)
    print(f"[1] identical embeds       loss = {loss:.5f}  (expect ~0, < 0.05)")
    assert loss < 0.05, loss
    return loss


def test_random_embeddings_log_n():
    """Test 2: independent random embeddings -> loss ~= log(N).

    Uncorrelated embeddings give a near-uniform softmax over N candidates, so the
    correct one gets ~1/N probability and cross-entropy is -log(1/N) = log(N).

    This identity is exact only when the logits are near zero, i.e. temperature = 1
    (unscaled cosine similarity). With D = 192 the off-diagonal cosines have std
    ~1/sqrt(D) ~= 0.07, so at tau = 1 the softmax is nearly uniform and the loss
    lands right on log(N). (At the sharp CLIP init tau = 0.07 the same random
    logits get multiplied by ~14, gaining real spread, so the loss sits somewhat
    ABOVE log(N) -- which is why the training-init baseline is only 'approximately'
    log(N).)"""
    torch.manual_seed(1)
    n = 256
    img = torch.randn(n, 192)
    txt = torch.randn(n, 192)
    loss = _loss_no_grad(img, txt, temperature=1.0)  # unscaled -> exact log(N) regime
    expected = math.log(n)
    print(f"[2] random embeds (tau=1)  loss = {loss:.5f}  (expect ~log(N) = {expected:.5f})")
    assert abs(loss - expected) < 0.1, (loss, expected)
    return loss


def test_mixture_between():
    """Test 3: half aligned + half random -> loss strictly between tests 1 and 2."""
    torch.manual_seed(2)
    n, d = 128, 192
    img = torch.randn(n, d)
    txt = img.clone()
    # Corrupt the second half of the text embeddings with independent noise.
    txt[n // 2:] = torch.randn(n // 2, d)

    # All three measured at the SAME temperature so the comparison is apples-to-apples.
    low = _loss_no_grad(img, img)              # fully aligned reference
    high = _loss_no_grad(img, torch.randn(n, d))  # fully random reference
    mixed = _loss_no_grad(img, txt)
    print(f"[3] half-aligned mixture   loss = {mixed:.5f}  (expect between {low:.4f} and {high:.4f})")
    assert low < mixed < high, (low, mixed, high)
    return mixed


def test_gradients_flow():
    """Test 4: backward() populates grads on log_inv_tau and BOTH input embeddings."""
    torch.manual_seed(3)
    n, d = 64, 192
    img = torch.randn(n, d, requires_grad=True)
    txt = torch.randn(n, d, requires_grad=True)
    loss_fn = InfoNCELoss()
    loss = loss_fn(img, txt)
    loss.backward()

    assert loss_fn.log_inv_tau.grad is not None, "no grad on log_inv_tau"
    assert img.grad is not None and torch.isfinite(img.grad).all(), "bad grad on image embeds"
    assert txt.grad is not None and torch.isfinite(txt.grad).all(), "bad grad on text embeds"
    assert loss_fn.log_inv_tau.grad.abs().item() > 0, "log_inv_tau grad is exactly zero"
    print(f"[4] gradients flow         d loss/d log_inv_tau = {loss_fn.log_inv_tau.grad.item():+.5f}, "
          f"|d loss/d img| = {img.grad.abs().mean().item():.3e}  (all present, finite)")
    return True


def main() -> int:
    print("=" * 68)
    print("Task 5 — InfoNCE sanity tests")
    print("=" * 68)
    checks = [
        test_identical_embeddings_low_loss,
        test_random_embeddings_log_n,
        test_mixture_between,
        test_gradients_flow,
    ]
    failures = 0
    for check in checks:
        try:
            check()
        except AssertionError as e:  # explicit: a numeric expectation was violated
            failures += 1
            print(f"    FAIL {check.__name__}: {e}")
    print("=" * 68)
    print("ALL TESTS PASSED" if failures == 0 else f"{failures} TEST(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
