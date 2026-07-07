"""Task 5 — Applied Stretch: a tiny semantic search engine.

Same mechanism as the toy alignment, but with a *pretrained* sentence encoder instead
of a randomly-initialised one, so the embeddings are actually meaningful. We embed a
small corpus once, then answer free-text queries by cosine similarity — exactly how a
vector database (Pinecone, Weaviate, FAISS) works, minus the approximate-nearest-
neighbour indexing that makes it scale.

Encoder: sentence-transformers/all-MiniLM-L6-v2 (small, free, 384-d).

    python task5/semantic_search.py                       # runs 3 demo queries
    python task5/semantic_search.py --query "space travel" # single query

Writes a transcript to task5/semantic_search_demo.txt.

Scaling to a million documents (the 50-word answer, also in writeup.md): stop doing an
exact O(N) dot product per query. Precompute and persist embeddings, then put them in
an approximate-nearest-neighbour index (FAISS IVF/HNSW or a managed vector DB) so
lookup is sub-linear; quantise vectors (int8/PQ) to cut memory; shard across machines
and batch-encode queries on a GPU.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# A small, deliberately varied corpus so semantic (not keyword) matching is visible.
CORPUS = [
    "The cat sat lazily on the warm windowsill in the afternoon sun.",
    "A kitten chased a ball of yarn across the living room floor.",
    "Dogs are loyal companions that love to play fetch in the park.",
    "The golden retriever splashed happily through the shallow river.",
    "Scientists discovered a new species of frog in the Amazon rainforest.",
    "The rainforest canopy teems with insects, birds, and hidden mammals.",
    "NASA launched a new probe to study the rings of Saturn.",
    "Astronomers detected water vapour in the atmosphere of a distant exoplanet.",
    "The rocket lifted off at dawn, trailing a column of white smoke.",
    "Stock markets fell sharply today amid fears of rising interest rates.",
    "The central bank raised rates to combat stubbornly high inflation.",
    "Investors poured money into technology startups this quarter.",
    "The chef simmered the tomato sauce slowly for several hours.",
    "Fresh basil and garlic give the pasta dish its aromatic flavour.",
    "A bakery on the corner sells warm croissants every morning.",
    "The marathon runner collapsed just meters before the finish line.",
    "She trained for months to compete in the Olympic swimming trials.",
    "The football team celebrated their championship victory downtown.",
    "Heavy rain and strong winds battered the coastal town overnight.",
    "A severe drought has left the region's reservoirs dangerously low.",
    "Firefighters battled the wildfire spreading through the dry hills.",
    "The orchestra performed a haunting symphony to a silent audience.",
    "A jazz quartet improvised late into the night at the small club.",
    "The museum unveiled a rare collection of Renaissance paintings.",
    "Researchers trained a neural network to translate ancient scripts.",
    "The new smartphone features a faster chip and a brighter screen.",
    "Engineers designed a bridge that can withstand powerful earthquakes.",
    "The electric car accelerated silently down the empty highway.",
    "Farmers harvested the wheat before the first autumn frost arrived.",
    "The old lighthouse guided ships safely past the rocky shore.",
    "Children built an elaborate sandcastle near the crashing waves.",
    "The hikers reached the snowy summit just as the clouds parted.",
    "A gentle breeze carried the scent of pine through the forest trail.",
    "The startup announced record profits and plans to hire hundreds.",
    "Doctors urged the public to get vaccinated before flu season.",
    "The library extended its hours to accommodate exam-week students.",
    "A power outage left thousands of homes without electricity for hours.",
    "The volcano erupted, sending ash miles into the darkening sky.",
    "Programmers debated the merits of different sorting algorithms online.",
    "The puppy curled up and fell asleep in its owner's lap.",
]


def build_index(model):
    embeddings = model.encode(CORPUS, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def search(model, index, query: str, top_k: int = 5):
    import numpy as np
    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
    scores = index @ q                      # cosine sim (both normalized)
    order = np.argsort(-scores)[:top_k]
    return [(CORPUS[i], float(scores[i])) for i in order]


def main() -> int:
    parser = argparse.ArgumentParser(description="Tiny semantic search demo.")
    parser.add_argument("--query", type=str, default=None, help="run a single query")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("This stretch demo needs sentence-transformers:\n"
              "    pip install sentence-transformers", file=sys.stderr)
        return 1

    print("Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    # CPU by default: this tiny model is fast on CPU and avoids competing for GPU VRAM
    # with any concurrent training run.
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    index = build_index(model)
    print(f"Indexed {len(CORPUS)} sentences.\n")

    queries = [args.query] if args.query else [
        "a pet playing indoors",
        "exploring outer space",
        "the economy and money",
    ]

    lines = []
    for q in queries:
        header = f"Query: {q!r}"
        print(header)
        lines.append(header)
        for rank, (sent, score) in enumerate(search(model, index, q, args.top_k), 1):
            row = f"  {rank}. ({score:.3f}) {sent}"
            print(row)
            lines.append(row)
        print()
        lines.append("")

    out = os.path.join(HERE, "semantic_search_demo.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("Task 5 Applied Stretch — semantic search demo\n")
        f.write("Encoder: all-MiniLM-L6-v2 | corpus: {} sentences\n".format(len(CORPUS)))
        f.write("=" * 60 + "\n")
        f.write("\n".join(lines) + "\n")
    print(f"Saved transcript -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
