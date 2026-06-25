"""Build (and execute) task3/images_as_tensors.ipynb — Task 3 Part A.

Explores how PyTorch represents images: MNIST and CIFAR-10 shapes, the (C, H, W)
convention, visualization, pixel statistics, and a DataLoader.

Run:
    python task3/build_images_notebook.py
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "images_as_tensors.ipynb")
DATA_ROOT = os.path.join(HERE, "data").replace("\\", "/")

cells = []

cells.append(new_markdown_cell(
    "# Task 3 — Part A: Images as Tensors\n"
    "\n"
    "How does PyTorch represent an image? We look at MNIST and CIFAR-10, the channel-first "
    "`(C, H, W)` convention, pixel statistics, and how a `DataLoader` batches images.\n"
))

cells.append(new_code_cell(
    "import torch\n"
    "import torchvision\n"
    "import torchvision.transforms as T\n"
    "from torch.utils.data import DataLoader\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    f"DATA_ROOT = r'{DATA_ROOT}'\n"
    "print('torch', torch.__version__, '| torchvision', torchvision.__version__)\n"
))

# ---- MNIST ----
cells.append(new_markdown_cell("## MNIST — 28x28 grayscale handwritten digits"))
cells.append(new_code_cell(
    "mnist = torchvision.datasets.MNIST(DATA_ROOT, train=True, download=True,\n"
    "                                   transform=T.ToTensor())\n"
    "img, label = mnist[0]\n"
    "print('single image shape (C, H, W):', tuple(img.shape), '| label:', label)\n"
    "print('C=1 because MNIST is grayscale; natural RGB images have C=3.')\n"
))
cells.append(new_markdown_cell("### Visualize the first 16 images"))
cells.append(new_code_cell(
    "fig, axes = plt.subplots(2, 8, figsize=(12, 3))\n"
    "for i, ax in enumerate(axes.flat):\n"
    "    img, label = mnist[i]\n"
    "    ax.imshow(img.squeeze(), cmap='gray')  # squeeze drops the channel dim for display\n"
    "    ax.set_title(str(label)); ax.axis('off')\n"
    "plt.tight_layout(); plt.show()\n"
))
cells.append(new_markdown_cell("### Pixel statistics (MNIST is already in [0, 1])"))
cells.append(new_code_cell(
    "img, _ = mnist[0]\n"
    "print(f'min={img.min():.4f}  max={img.max():.4f}  mean={img.mean():.4f}')\n"
    "print('Real photographs usually need standardization (subtract mean, divide by std).')\n"
))

# ---- CIFAR-10 ----
cells.append(new_markdown_cell("## CIFAR-10 — 32x32 color images, 10 classes"))
cells.append(new_code_cell(
    "CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']\n"
    "cifar = torchvision.datasets.CIFAR10(DATA_ROOT, train=True, download=True,\n"
    "                                     transform=T.ToTensor())\n"
    "img, label = cifar[0]\n"
    "print('single image shape (C, H, W):', tuple(img.shape), '| class:', CLASSES[label])\n"
    "print('C=3 (RGB), H=W=32.')\n"
))
cells.append(new_markdown_cell("### Visualize 16 CIFAR-10 images with class labels"))
cells.append(new_code_cell(
    "fig, axes = plt.subplots(2, 8, figsize=(12, 3.5))\n"
    "for i, ax in enumerate(axes.flat):\n"
    "    img, label = cifar[i]\n"
    "    # permute (C, H, W) -> (H, W, C) for matplotlib, which expects channels last.\n"
    "    ax.imshow(img.permute(1, 2, 0))\n"
    "    ax.set_title(CLASSES[label], fontsize=8); ax.axis('off')\n"
    "plt.tight_layout(); plt.show()\n"
))

# ---- DataLoader ----
cells.append(new_markdown_cell("## DataLoader — batching with batch size 64"))
cells.append(new_code_cell(
    "loader = DataLoader(cifar, batch_size=64, shuffle=True, num_workers=0)\n"
    "x, y = next(iter(loader))\n"
    "print('batch images x shape:', tuple(x.shape))   # (64, 3, 32, 32)\n"
    "print('batch labels y shape:', tuple(y.shape))   # (64,)\n"
    "print('x dtype:', x.dtype, '| y dtype:', y.dtype)\n"
))

cells.append(new_markdown_cell(
    "---\n"
    "**Takeaways.** PyTorch images are channel-first `(C, H, W)`; matplotlib and most image "
    "libraries are channel-last `(H, W, C)`, so we `permute` before `imshow`. A batch from a "
    "`DataLoader` stacks images into `(B, C, H, W)` with integer labels `(B,)`.\n"
))

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.x"},
})


def main():
    try:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=600, kernel_name="python3").execute()
        print("Executed notebook (outputs embedded).")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not execute notebook ({exc}). Saving without outputs.")
    with open(OUT, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
