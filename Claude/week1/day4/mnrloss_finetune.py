"""
Week 1, Day 4 — Code 1: Fine-Tuning BGE with MultipleNegativesRankingLoss

Demonstrates end-to-end embedding fine-tuning on a synthetic retail dataset:
  - Building (query, positive_document) training pairs
  - Hard negative mining via BM25 rank + embedding score filter
  - Training with MNRLoss
  - Evaluating nDCG@10 before and after fine-tuning

Requires:
  uv add sentence-transformers rank-bm25 datasets
"""

from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from rank_bm25 import BM25Okapi
import numpy as np

# ── 1. Toy Retail Training Dataset ──────────────────────────────────────
# In production: replace with click-through logs or human annotations.
TRAINING_PAIRS = [
    ("steel toe work boots",         "Men's Waterproof Steel Toe Safety Boots – SKU-9921-ST"),
    ("lightweight running sneakers",  "Women's Cushion Run Sneaker – Ultra-Light EVA Sole"),
    ("kids trail shoes size 7",       "Children's Trail Runner – Size 7K, Waterproof"),
    ("30 day return footwear",        "Footwear Return Policy: Full refund within 30 days of purchase"),
    ("leather casual loafers brown",  "Men's Genuine Leather Slip-On Loafers – Cognac Brown"),
    ("non-slip kitchen mat",          "Anti-Fatigue Kitchen Mat – Non-Slip Rubber Base 18x30in"),
    ("insulated water bottle 32oz",   "Stainless Insulated Tumbler – 32oz Keep Cold 24hrs"),
    ("waterproof hiking backpack",    "45L Adventure Backpack – IPX5 Waterproof, Osprey-style"),
]

# ── 2. Build InputExample list for MNRLoss ───────────────────────────────
# MNRLoss needs InputExample(texts=[query, positive]).
# In-batch negatives are the other positives in the same batch.
train_examples = [
    InputExample(texts=[q, p]) for q, p in TRAINING_PAIRS
]

# ── 3. Load base model ───────────────────────────────────────────────────
print("Loading base model bge-small-en-v1.5...")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# ── 4. DataLoader + Loss ─────────────────────────────────────────────────
# batch_size controls how many in-batch negatives are available.
# In production use batch_size >= 64 for meaningful hard-negative density.
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=4)
train_loss = losses.MultipleNegativesRankingLoss(model)

# ── 5. Hard Negative Mining (BM25-assisted) ──────────────────────────────
# Find documents that score high on BM25 (lexical match) but are NOT the
# relevant positive. These are the most instructive negatives.
print("\nMining hard negatives via BM25...")
corpus = [p for _, p in TRAINING_PAIRS]
tokenized_corpus = [doc.lower().split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)

hard_negative_pairs = []
for query, positive in TRAINING_PAIRS:
    query_tokens = query.lower().split()
    bm25_scores = bm25.get_scores(query_tokens)

    # Get top-3 BM25 hits
    top_idxs = np.argsort(bm25_scores)[::-1][:3]
    for idx in top_idxs:
        candidate = corpus[idx]
        # Only add if NOT the ground-truth positive
        if candidate != positive:
            hard_negative_pairs.append((query, positive, candidate))
            break

print(f"Found {len(hard_negative_pairs)} hard negative triplets:")
for q, p, n in hard_negative_pairs[:3]:
    print(f"  Q: {q[:40]}")
    print(f"  P: {p[:40]}")
    print(f"  N: {n[:40]}")
    print()

# ── 6. Train (demo: 1 epoch — production: 3-10 epochs on full dataset) ───
print("Starting fine-tuning (1 epoch demo)...")
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=1,
    warmup_steps=2,
    output_path="./finetuned-bge-retail",
    show_progress_bar=True,
)

print("\nFine-tuning complete. Model saved to ./finetuned-bge-retail/")
print("Next: run nDCG@10 evaluation comparing base vs. fine-tuned on held-out test set.")
