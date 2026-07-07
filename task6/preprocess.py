"""Task 6 — one-time image cache so data loading doesn't starve the GPU.

Decoding 8,091 full-resolution JPEGs on every access made a data-only epoch take ~110s
(handbook target: 10-30s). Since this project fixes the image size at 64x64, we decode
+ resize every image ONCE to a 64x64 uint8 array and memory-map it. Augmentation
(flip / colour jitter / normalize) is then applied cheaply on the tiny cached image.

    python task6/preprocess.py

Writes task6/data/images_64.npy (uint8, N x 64 x 64 x 3) and images_64_index.json
(filename -> row). Idempotent: skips if the cache already covers all images.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image

from dataset import DATA_DIR, find_image_dir

CACHE_SIZE = 64


def cache_paths(data_dir: str = DATA_DIR):
    return (os.path.join(data_dir, "images_64.npy"),
            os.path.join(data_dir, "images_64_index.json"))


def build_cache(data_dir: str = DATA_DIR) -> None:
    img_dir = find_image_dir(data_dir)
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(".jpg"))
    npy_path, idx_path = cache_paths(data_dir)

    if os.path.exists(npy_path) and os.path.exists(idx_path):
        index = json.load(open(idx_path, encoding="utf-8"))
        if len(index) == len(files):
            print(f"[skip] cache already covers {len(files)} images")
            return

    print(f"Caching {len(files)} images to {CACHE_SIZE}x{CACHE_SIZE} ...")
    arr = np.zeros((len(files), CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
    index = {}
    for i, fn in enumerate(files):
        img = Image.open(os.path.join(img_dir, fn)).convert("RGB").resize(
            (CACHE_SIZE, CACHE_SIZE), Image.BILINEAR)
        arr[i] = np.asarray(img, dtype=np.uint8)
        index[fn] = i
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(files)}")
    np.save(npy_path, arr)
    json.dump(index, open(idx_path, "w", encoding="utf-8"))
    print(f"[done] wrote {npy_path} ({arr.nbytes/1e6:.0f} MB) and index")


def load_cache(data_dir: str = DATA_DIR):
    """Return (memmapped uint8 array, filename->row index) or (None, None) if absent."""
    npy_path, idx_path = cache_paths(data_dir)
    if not (os.path.exists(npy_path) and os.path.exists(idx_path)):
        return None, None
    arr = np.load(npy_path, mmap_mode="r")
    index = json.load(open(idx_path, encoding="utf-8"))
    return arr, index


if __name__ == "__main__":
    build_cache()
