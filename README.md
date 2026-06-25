# SOC Multimodal Transformer — VLM from Scratch

**Seasons of Code · IIT Bombay — "Vision Meets Language: Building a Multimodal Transformer from Scratch"**

This repository contains my implementation of the *VLM from Scratch* mentee handbook,
Tasks 0 through 4. Everything is built from scratch in PyTorch — no `nn.MultiheadAttention`,
no `nn.TransformerEncoderLayer`. Every line of code corresponds to a piece of math.

Handbooks: `VLM_Mentee_Handbook_Tasks_0_1_2.pdf`, `VLM_Mentee_Handbook_Tasks_3_4.pdf`.

## Timeline / task map

| Task | Theme | Folder |
|------|-------|--------|
| Task 0 | Environment setup, tensor warmup, paper reading | [`task0/`](task0/) |
| Task 1 | Bigram language model + a single self-attention head | [`task1/`](task1/) |
| Task 2 | Multi-head attention, full transformer decoder, attention-gradient derivation | [`task2/`](task2/) |
| Task 3 | Vision Transformer (ViT) — patches, position embeddings, image classification | [`task3/`](task3/) |
| Task 4 | Cross-attention — joining vision and language | [`task4/`](task4/) |

## Repository layout

```
task0/  tensors.ipynb, paper_response.md
        build_tensors_notebook.py            (generates+executes the notebook)
task1/  bigram.py, attention.py, samples.txt, writeup.md
        download_data.py, make_samples.py
task2/  transformer.py, math.md -> math.pdf, ablation_plot.png, samples.txt, writeup.md
        build_math_pdf.py
task3/  images_as_tensors.ipynb, cnn_baseline.py, cnn_curves.png, vit.py, vit_curves.png,
        comparison_plot.png, writeup.md
        build_images_notebook.py, comparison.py
task4/  cross_attention_toy.ipynb, multimodal.py, training_curve.png, attention_viz.png,
        samples.txt, writeup.md
        build_toy_notebook.py
```

Notebooks are produced by their `build_*.py` script, which constructs the cells and
executes them (via `nbclient`) so all outputs are embedded.

## Results achieved

All code is implemented from scratch and verified end-to-end. Headline numbers from
the runs in this repo:

| Task | Result | Handbook target |
|------|--------|-----------------|
| 1 — bigram | val loss **2.46** | ~2.5 |
| 1 — single head | val loss **2.40** | ~2.4 |
| 2 — transformer (baseline) | val loss **1.82** @1500 steps (→~1.5 with full 5000) | ~1.5 |
| 2 — ablation | no-residual **stalls @3.35**, no-LayerNorm **2.36**, baseline **1.82** | residual removal is catastrophic |
| 3 — TinyCNN | **76.2%** val acc (10 ep) | 65–70% |
| 3 — ViT | **64.4%** val acc (6 ep, still climbing) | 65–72% @30 ep |
| 4 — multimodal | loss **3.14 → 0.065**, grammatical captions | →~0.5 or lower |

> **Compute note.** Everything was trained on **CPU**. The handbook's full scale
> (Task 2: 5000 steps × 3; Task 3 ViT: 30 epochs) targets a GPU/Colab and would take
> many hours here, so the artifacts in this repo use honest reduced runs (Task 2:
> 1500 steps; ViT: 6 epochs; Task 4: 1500 steps) that reproduce every expected
> behavior. The source keeps the exact handbook defaults — run without `--smoke`/
> `--steps`/`--epochs` overrides on a GPU to reproduce the headline numbers.

## Setup

```bash
python -m pip install -r requirements.txt
```

PyTorch CPU is sufficient for everything here. A GPU (or Colab T4) makes Tasks 3–4 faster.

## How training scripts are parameterised

Every training script keeps the **handbook default hyperparameters** but also accepts a
`--smoke` flag (and a few overrides such as `--steps` / `--epochs`). `--smoke` runs a tiny,
fast version that verifies the whole pipeline executes end to end without bugs. Run without
`--smoke` to reproduce the handbook results.

```bash
python task1/bigram.py --smoke        # fast sanity run
python task1/bigram.py                 # full handbook run (3000 steps)
```

## Verifying the install

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.tensor([1.0, 2.0]) @ torch.tensor([3.0, 4.0]))
```
