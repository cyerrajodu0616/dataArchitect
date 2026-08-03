"""
Week 1, Day 4 — Code 2: Hybrid Search (BM25 + Dense Vector) with RRF Fusion

Demonstrates why hybrid search handles exact SKU lookup AND semantic queries
without touching model weights.

Requires:
  uv add sentence-transformers rank-bm25 numpy
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ── 1. Retail Product Catalog ────────────────────────────────────────────
CATALOG = [
    "Men's Waterproof Steel Toe Safety Boots – SKU-9921-ST",         # 0
    "Women's Cushion Run Sneaker – Ultra-Light EVA Sole – SKU-4410",  # 1
    "Workwear Steel Blue Denim Jeans – Reinforced Rivets",            # 2
    "Children's Trail Runner – Size 7K – Waterproof – SKU-7712",      # 3
    "Stainless Steel Non-Stick 10pc Cookware Set",                    # 4
    "Men's Leather Slip-On Loafers – Cognac Brown – SKU-3301",        # 5
    "Cushioned Anti-Fatigue Insole – Fits Steel Toe Boots",           # 6
    "Footwear Return Policy: 30-day full refund on all shoes",        # 7
]

QUERIES = [
    "SKU-9921-ST",                          # exact part number — BM25 should win
    "comfortable footwear reduces fatigue",  # pure semantic — dense should win
    "steel toe boots for work",              # hybrid advantage
]

# ── 2. BM25 Index ────────────────────────────────────────────────────────
tokenized_catalog = [doc.lower().split() for doc in CATALOG]
bm25 = BM25Okapi(tokenized_catalog)

# ── 3. Dense Embedding Index ─────────────────────────────────────────────
print("Loading bi-encoder...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
doc_vecs = model.encode(CATALOG, normalize_embeddings=True)  # unit-norm for cosine


def bm25_rank(query: str) -> list:
    """Return catalog indices sorted by BM25 score (best first)."""
    scores = bm25.get_scores(query.lower().split())
    return list(np.argsort(scores)[::-1])


def dense_rank(query: str) -> list:
    """Return catalog indices sorted by cosine similarity (best first)."""
    q_vec = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True
    )
    sims = doc_vecs @ q_vec   # dot == cosine since both normalized
    return list(np.argsort(sims)[::-1])


def rrf_fuse(ranks_a: list, ranks_b: list, k: int = 60) -> list:
    """
    Reciprocal Rank Fusion: RRF(d) = 1/(k + rank_a(d)) + 1/(k + rank_b(d))
    Returns catalog indices sorted by fused score (best first).
    k=60 is empirically stable (Robertson et al. 2009).
    """
    scores = {}
    for rank, idx in enumerate(ranks_a):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    for rank, idx in enumerate(ranks_b):
        scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── 4. Run All Three Modes ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("HYBRID SEARCH DEMO — BM25 vs Dense vs RRF Fusion")
print("=" * 65)

for query in QUERIES:
    bm25_results  = bm25_rank(query)
    dense_results = dense_rank(query)
    rrf_results   = rrf_fuse(bm25_results, dense_results)

    print(f"\nQuery: \"{query}\"")
    print(f"  BM25  Top-1 : {CATALOG[bm25_results[0]][:60]}")
    print(f"  Dense Top-1 : {CATALOG[dense_results[0]][:60]}")
    print(f"  RRF   Top-1 : {CATALOG[rrf_results[0]][:60]}")

print("\n" + "=" * 65)
print("KEY INSIGHT: RRF never needs you to touch a single model weight.")
print("Exact SKU queries: BM25 IDF scores part numbers near-perfectly.")
print("Semantic queries: Dense vectors cover zero-overlap paraphrases.")
print("RRF fuses both systems by rank — no scale normalisation needed.")
