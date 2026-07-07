"""Render task5/derivation.md into task5/derivation.pdf (monospace, paginated).

Plain-notation math so it renders reliably without a LaTeX install (same approach as
task2/build_math_pdf.py). Run:
    python task5/build_derivation_pdf.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "derivation.md")
OUT = os.path.join(HERE, "derivation.pdf")

LINES_PER_PAGE = 52


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]
    with PdfPages(OUT) as pdf:
        for pg in pages:
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.text(0.05, 0.98, "\n".join(pg), va="top", ha="left",
                    family="monospace", fontsize=9.0, transform=ax.transAxes)
            pdf.savefig(fig); plt.close(fig)
    print(f"Wrote {OUT} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
