"""Train both Task 1 models (full runs) and write task1/samples.txt with a
200-character sample from each, as required by the deliverables.

Run:
    python task1/make_samples.py            # full runs (bigram 3000, attention 5000)
    python task1/make_samples.py --smoke     # fast sanity run
"""
import argparse
import os

from attention import train_attention
from bigram import train_bigram

HERE = os.path.dirname(os.path.abspath(__file__))


def main(smoke=False):
    print("===== Bigram =====")
    _, _, bigram_sample = train_bigram(smoke=smoke)
    print("\n===== Single-head attention =====")
    _, _, attn_sample = train_attention(smoke=smoke)

    out = os.path.join(HERE, "samples.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("===== Bigram model — 200-character sample =====\n")
        f.write(bigram_sample + "\n\n")
        f.write("===== Single-head attention model — 200-character sample =====\n")
        f.write(attn_sample + "\n")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    main(smoke=args.smoke)
