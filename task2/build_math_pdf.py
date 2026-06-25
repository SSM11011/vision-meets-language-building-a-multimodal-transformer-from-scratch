"""Render task2/math.md into task2/math.pdf (clean typeset, monospace, paginated).

The math is written in plain readable notation so it renders reliably without a
LaTeX installation. Run:
    python task2/build_math_pdf.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "math.md")
OUT = os.path.join(HERE, "math.pdf")

LINES_PER_PAGE = 50


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")

    # Paginate into chunks of LINES_PER_PAGE lines.
    pages = [lines[i:i + LINES_PER_PAGE] for i in range(0, len(lines), LINES_PER_PAGE)]

    with PdfPages(OUT) as pdf:
        for pg in pages:
            fig = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
            fig.subplots_adjust(left=0.07, right=0.97, top=0.97, bottom=0.03)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            text = "\n".join(pg)
            ax.text(0.06, 0.97, text, va="top", ha="left", family="monospace",
                    fontsize=9.5, transform=ax.transAxes)
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Wrote {OUT} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
