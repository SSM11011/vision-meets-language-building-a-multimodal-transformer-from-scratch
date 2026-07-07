"""Download and extract Flickr8k (images + captions/splits) into task6/data/.

Public mirror (no Kaggle auth needed): jbrownlee/Datasets GitHub release.
  Flickr8k_Dataset.zip  (~1.1 GB) -> 8091 JPEGs
  Flickr8k_text.zip     (~2.3 MB) -> captions (Flickr8k.token.txt) + standard
                                      6000/1000/1000 train/dev/test split lists.

Idempotent: skips downloads/extractions that already exist. Streams to disk with
a .part temp file so a partial download is never mistaken for a complete one.

    python task6/download_data.py
"""
from __future__ import annotations

import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)

FILES = {
    "Flickr8k_Dataset.zip": "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip",
    "Flickr8k_text.zip":    "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip",
}


def download(url: str, dest: str) -> None:
    if os.path.exists(dest):
        print(f"[skip] {os.path.basename(dest)} already downloaded ({os.path.getsize(dest)/1e6:.1f} MB)")
        return
    part = dest + ".part"
    print(f"[get ] {url}")
    with urllib.request.urlopen(url) as r:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        with open(part, "wb") as f:
            while True:
                chunk = r.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r       {done/1e6:7.1f}/{total/1e6:7.1f} MB ({pct:5.1f}%)", end="", flush=True)
        print()
    os.replace(part, dest)
    print(f"[done] {os.path.basename(dest)}")


def extract(zip_path: str, out_dir: str, marker: str) -> None:
    if os.path.exists(os.path.join(out_dir, marker)):
        print(f"[skip] {marker} already extracted")
        return
    print(f"[unzip] {os.path.basename(zip_path)} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print("[done ] extraction")


def main() -> int:
    for name, url in FILES.items():
        download(url, os.path.join(DATA, name))

    # Images zip extracts to Flicker8k_Dataset/ (note the mirror's spelling) with 8091 jpgs.
    extract(os.path.join(DATA, "Flickr8k_Dataset.zip"), DATA, "Flicker8k_Dataset")
    # Text zip extracts caption/split .txt files directly into DATA.
    extract(os.path.join(DATA, "Flickr8k_text.zip"), DATA, "Flickr8k.token.txt")

    # Report what we got.
    img_dir = os.path.join(DATA, "Flicker8k_Dataset")
    n_imgs = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")]) if os.path.isdir(img_dir) else 0
    print(f"\nImages: {n_imgs} jpgs in {img_dir}")
    for f in ["Flickr8k.token.txt", "Flickr_8k.trainImages.txt",
              "Flickr_8k.devImages.txt", "Flickr_8k.testImages.txt"]:
        p = os.path.join(DATA, f)
        print(f"  {'OK ' if os.path.exists(p) else 'MISSING'} {f}")
    return 0 if n_imgs == 8091 else 1


if __name__ == "__main__":
    sys.exit(main())
