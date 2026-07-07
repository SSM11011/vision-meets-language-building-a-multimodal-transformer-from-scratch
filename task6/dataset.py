"""Task 6 — Flickr8k PyTorch Dataset and DataLoader helpers.

Flickr8k: 8,091 images, each with 5 human captions. We use the dataset's own standard
6,000 / 1,000 / 1,000 train/dev/test split lists (the classic split; we do not invent
our own). Each (image, caption) is an independent contrastive example, so an epoch of
the train split has ~30,000 pairs.

Layout expected under task6/data/ (produced by download_data.py):
    Flicker8k_Dataset/            8091 *.jpg   (note the mirror's spelling)
    Flickr8k.token.txt            "img.jpg#k <TAB> caption"
    Flickr_8k.trainImages.txt     split lists (one filename per line)
    Flickr_8k.devImages.txt
    Flickr_8k.testImages.txt

Each __getitem__ returns a dict: image (3,H,W) tensor, tokens (max_text_len,) long,
mask (max_text_len,) long, caption (str), image_id (int -- same for an image's 5
captions, so retrieval can group them).
"""
from __future__ import annotations

import os

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

# ImageNet normalization stats (handbook).
_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_SPLIT_FILES = {
    "train": "Flickr_8k.trainImages.txt",
    "val": "Flickr_8k.devImages.txt",
    "test": "Flickr_8k.testImages.txt",
}


def find_image_dir(data_dir: str = DATA_DIR) -> str:
    """Return the image folder, tolerating the 'Flicker'/'Flickr' spelling variants."""
    for name in ("Flicker8k_Dataset", "Flickr8k_Dataset", "Images", "images"):
        cand = os.path.join(data_dir, name)
        if os.path.isdir(cand):
            return cand
    raise FileNotFoundError(
        f"No Flickr8k image folder under {data_dir}. Run: python task6/download_data.py")


def _find_captions_file(data_dir: str) -> str:
    for name in ("Flickr8k.token.txt", "captions.txt"):
        cand = os.path.join(data_dir, name)
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"No captions file under {data_dir}.")


def _load_split_filenames(data_dir: str, split: str) -> list[str]:
    path = os.path.join(data_dir, _SPLIT_FILES[split])
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _load_caption_map(captions_file: str) -> dict[str, list[str]]:
    """Parse 'img.jpg#k <TAB> caption' into {filename: [caption, ...]}."""
    caps: dict[str, list[str]] = {}
    with open(captions_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if "\t" in line:
                key, caption = line.split("\t", 1)
            else:  # some mirrors use "img.jpg#k caption" (space) or CSV "img.jpg,caption"
                parts = line.split(",", 1) if "," in line and "#" not in line else line.split(None, 1)
                if len(parts) != 2:
                    continue
                key, caption = parts
            filename = key.split("#")[0].strip()
            caption = caption.strip()
            if filename and caption and filename.lower() != "image":  # skip a CSV header
                caps.setdefault(filename, []).append(caption)
    return caps


class Flickr8kDataset(Dataset):
    def __init__(self, tokenizer, split: str = "train", image_size: int = 64,
                 max_text_len: int = 32, data_dir: str = DATA_DIR):
        if split not in _SPLIT_FILES:
            raise ValueError(f"split must be one of {list(_SPLIT_FILES)}, got {split!r}")
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.image_dir = find_image_dir(data_dir)

        # Fast path: use the 64x64 uint8 cache if present and we want 64x64 images.
        # Falls back to decoding JPEGs on the fly otherwise.
        self._cache = self._cache_index = None
        if image_size == 64:
            from preprocess import load_cache
            self._cache, self._cache_index = load_cache(data_dir)

        caption_map = _load_caption_map(_find_captions_file(data_dir))
        split_files = _load_split_filenames(data_dir, split)

        # Stable image_id per filename (sorted) so ids are reproducible across runs.
        present = [fn for fn in sorted(split_files) if fn in caption_map]
        self.image_id_of = {fn: i for i, fn in enumerate(present)}

        # Flatten to (filename, caption) pairs — one training example per caption.
        self.pairs: list[tuple[str, str]] = []
        for fn in present:
            for cap in caption_map[fn]:
                self.pairs.append((fn, cap))

        if not self.pairs:
            raise RuntimeError(f"No (image, caption) pairs for split={split!r}. "
                               f"Check the dataset under {data_dir}.")

        if split == "train":
            # Stronger augmentation than the handbook's plain Resize: a mild
            # random-resized-crop plus flip + colour jitter. On a small dataset like
            # Flickr8k this materially reduces overfitting (train/val gap).
            self.transform = T.Compose([
                T.RandomResizedCrop((image_size, image_size), scale=(0.6, 1.0),
                                    antialias=True),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2),
                T.ToTensor(),
                T.Normalize(mean=_MEAN, std=_STD),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=_MEAN, std=_STD),
            ])

    @property
    def num_images(self) -> int:
        return len(self.image_id_of)

    def all_captions(self) -> list[str]:
        return [cap for _, cap in self.pairs]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        filename, caption = self.pairs[idx]
        if self._cache is not None and filename in self._cache_index:
            import numpy as np
            image = Image.fromarray(np.asarray(self._cache[self._cache_index[filename]]))
        else:
            image = Image.open(os.path.join(self.image_dir, filename)).convert("RGB")
        image = self.transform(image)

        ids = self.tokenizer.encode(caption)[: self.max_text_len]
        pad = self.max_text_len - len(ids)
        mask = [1] * len(ids) + [0] * pad
        ids = ids + [self.tokenizer.PAD] * pad

        return {
            "image": image,
            "tokens": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "caption": caption,
            "image_id": self.image_id_of[filename],
        }


def collate(batch: list[dict]) -> dict:
    """Stack tensors; keep captions as a python list of strings."""
    return {
        "image": torch.stack([b["image"] for b in batch]),
        "tokens": torch.stack([b["tokens"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "caption": [b["caption"] for b in batch],
        "image_id": torch.tensor([b["image_id"] for b in batch], dtype=torch.long),
    }


def make_loader(dataset: Flickr8kDataset, batch_size: int = 128, shuffle: bool = True,
                num_workers: int = 2) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate, pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,  # drop last partial batch only while training (stable N of negatives)
        persistent_workers=(num_workers > 0),
    )


def build_tokenizer(data_dir: str = DATA_DIR, min_freq: int = 2):
    """Build (or load) the word tokenizer from the TRAIN captions only."""
    from tokenizer import WordTokenizer
    vocab_path = os.path.join(data_dir, "vocab.json")
    if os.path.exists(vocab_path):
        return WordTokenizer.load(vocab_path)
    caption_map = _load_caption_map(_find_captions_file(data_dir))
    train_files = set(_load_split_filenames(data_dir, "train"))
    train_caps = [c for fn, caps in caption_map.items() if fn in train_files for c in caps]
    tok = WordTokenizer.build(train_caps, min_freq=min_freq)
    tok.save(vocab_path)
    return tok


if __name__ == "__main__":
    # Day-1 verification (handbook): shapes, caption readability, timing.
    import time

    tok = build_tokenizer()
    print(f"vocab_size = {tok.vocab_size}")
    ds = Flickr8kDataset(tok, split="train")
    print(f"train pairs = {len(ds)}  over {ds.num_images} images")

    s = ds[0]
    print(f"image {tuple(s['image'].shape)}  tokens {tuple(s['tokens'].shape)}  "
          f"mask sum {int(s['mask'].sum())}")
    print(f"caption: {s['caption']!r}")
    print(f"decoded: {tok.decode(s['tokens'])!r}")

    print("\n10 sample captions (check for unicode/punctuation corruption):")
    for i in range(0, len(ds), max(1, len(ds) // 10))[:10]:
        print(f"  [{i}] {ds.pairs[i][1]!r}")

    loader = make_loader(ds, batch_size=128, num_workers=2)
    t0 = time.time()
    nb = 0
    for _ in loader:
        nb += 1
    print(f"\nOne epoch data-only pass: {nb} batches in {time.time() - t0:.1f}s "
          f"(handbook target 10-30s).")
