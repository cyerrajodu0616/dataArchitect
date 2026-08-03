"""
Week 1, Day 3 — Deep Dive Code 1: Bi-Encoder vs. Cross-Encoder 2-Stage Retrieval

Demonstrates:
  Stage 1: Bi-Encoder vector retrieval (O(1) index lookup to get top K candidates)
  Stage 2: Cross-Encoder joint re-ranking (O(K) transformer cross-attention pass)

Requires:
  uv add sentence-transformers numpy
"""

import numpy as np
from sentence-transformers import SentenceTransformer, CrossEncoder

# 1. Load Bi-Encoder (Stage 1) and Cross-Encoder (Stage 2)
print("Loading Stage 1 Bi-Encoder (bge-small-en-v1.5)...")
bi_encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

print("Loading Stage 2 Cross-Encoder (bge-reranker-base)...")
cross_encoder = CrossEncoder("BAAI/bge-reranker-base")

# 2. Product Catalog (Documents)
DOCUMENTS = [
    "Men's Waterproof Trail Running Boots - Heavy Duty Steel Toe",
    "Women's Lightweight Cushion Running Sneakers - Blue / Size 8",
    "Workwear Denim Jeans with Reinforced Steel Rivets",
    "Stainless Steel Non-Stick 10-Piece Kitchen Cookware Set",
    "Men's Casual Leather Slip-On Loafers - Brown",
]

QUERY = "steel toe boots for work"

print(f"\nQuery: '{QUERY}'\n" + "-"*50)

# --- STAGE 1: Bi-Encoder Retrieval (Top 4 Candidates) ---
# Note: BGE requires query prefix
query_embedding = bi_encoder.encode(f"Represent this sentence for searching relevant passages: {QUERY}")
doc_embeddings = bi_encoder.encode(DOCUMENTS)

# Cosine similarity calculation
scores_stage1 = np.dot(doc_embeddings, query_embedding) / (
    np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
)

# Rank documents by Stage 1 score
top_k_indices = np.argsort(scores_stage1)[::-1][:4]

print("\n--- STAGE 1: Bi-Encoder Vector Search Results (Top 4) ---")
candidate_docs = []
for rank, idx in enumerate(top_k_indices):
    score = scores_stage1[idx]
    doc_text = DOCUMENTS[idx]
    candidate_docs.append(doc_text)
    print(f"Rank {rank+1} (Score: {score:.4f}): {doc_text}")

# --- STAGE 2: Cross-Encoder Re-ranking ---
# Build (Query, Document) pairs for joint cross-attention scoring
pairs = [[QUERY, doc] for doc in candidate_docs]
rerank_scores = cross_encoder.predict(pairs)

# Rank candidates by Stage 2 score
reranked_indices = np.argsort(rerank_scores)[::-1]

print("\n--- STAGE 2: Cross-Encoder Reranked Final Results ---")
for rank, idx in enumerate(reranked_indices):
    score = rerank_scores[idx]
    doc_text = candidate_docs[idx]
    print(f"Final Rank {rank+1} (Rerank Score: {score:.4f}): {doc_text}")
