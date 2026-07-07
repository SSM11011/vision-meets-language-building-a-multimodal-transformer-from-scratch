"""Task 6 — a simple word-level tokenizer for Flickr8k captions.

Flickr8k has only a few thousand distinct words, so a word-level vocabulary built
from the training captions is plenty. Reserved ids (per the handbook):

    0 = <pad>   padding
    1 = <unk>   unknown / out-of-vocabulary word
    2 = <bos>   start-of-sequence (prepended to every caption)

The vocabulary is built ONCE from the training split and saved to JSON so training,
evaluation, zero-shot classification and search all share identical token ids.
"""
from __future__ import annotations

import json
import re
from collections import Counter

# Lowercase word tokens; drops punctuation and digits. Applied identically everywhere.
_WORD_RE = re.compile(r"[a-z]+")


class WordTokenizer:
    PAD, UNK, BOS = 0, 1, 2
    SPECIALS = ["<pad>", "<unk>", "<bos>"]

    def __init__(self, stoi: dict[str, int]):
        self.stoi = dict(stoi)
        self.itos = {i: w for w, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)
        # Sanity: reserved ids must be where we expect them.
        assert self.stoi.get("<pad>") == self.PAD
        assert self.stoi.get("<unk>") == self.UNK
        assert self.stoi.get("<bos>") == self.BOS

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return _WORD_RE.findall(text.lower())

    @classmethod
    def build(cls, captions, min_freq: int = 2) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for cap in captions:
            counter.update(cls.tokenize(cap))
        stoi = {tok: i for i, tok in enumerate(cls.SPECIALS)}
        for word, freq in counter.most_common():
            if freq >= min_freq:
                stoi[word] = len(stoi)
        return cls(stoi)

    def encode(self, text: str) -> list[int]:
        """Text -> list of ids, prefixed with <bos>. No padding here (dataset pads)."""
        return [self.BOS] + [self.stoi.get(w, self.UNK) for w in self.tokenize(text)]

    def decode(self, ids) -> str:
        skip = {self.PAD, self.BOS}
        return " ".join(self.itos.get(int(i), "<unk>") for i in ids if int(i) not in skip)

    # ---- persistence -------------------------------------------------------- #
    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"stoi": self.stoi}, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["stoi"])


if __name__ == "__main__":
    # Tiny self-check.
    tok = WordTokenizer.build(["A dog runs.", "a DOG jumps!", "two dogs play"], min_freq=1)
    print("vocab_size:", tok.vocab_size)
    ids = tok.encode("A dog flies")     # 'flies' is OOV -> <unk>
    print("encode 'A dog flies':", ids)
    print("decode:", tok.decode(ids))
    assert ids[0] == tok.BOS and tok.UNK in ids
    print("tokenizer OK")
