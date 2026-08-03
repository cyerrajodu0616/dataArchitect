"""
Week 1, Day 2 hands-on — fetch real embeddings for your 3 Day-1 sentences and
compute cosine similarity, dot product, and Euclidean distance between every pair.

Requires: pip install openai numpy
Requires: OPENAI_API_KEY set in your environment (same key your agent uses).

Run:
    python day2_distance_demo.py
"""

import itertools
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

client = OpenAI()  # Automatically picks up OPENAI_API_KEY from .env

# --- Day 1 worked-example sentences ----------------------------------------
SENTENCES = [
    "The customer returned the shoes because they didn't fit.",
    "Customer requested a refund on footwear due to sizing issue.",
    "The warehouse received a new shipment of stainless steel cookware.",
]
# ---------------------------------------------------------------------------

MODEL = "text-embedding-3-small"  # swap to text-embedding-3-large if that's what you used Day 1


def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL, input=text)
    return np.array(resp.data[0].embedding)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main():
    print(f"Fetching embeddings from {MODEL} for {len(SENTENCES)} sentences...\n")
    vectors = [get_embedding(s) for s in SENTENCES]

    for i, (s, v) in enumerate(zip(SENTENCES, vectors)):
        norm = np.linalg.norm(v)
        print(f"[{i}] \"{s}\"")
        print(f"    dims={len(v)}  ‖v‖={norm:.6f}  (should be ~1.0 — OpenAI embeddings are pre-normalized)\n")

    print("Pairwise distances:\n")
    print(f"{'pair':<10}{'cosine sim':>14}{'dot product':>16}{'euclidean':>14}")
    for i, j in itertools.combinations(range(len(SENTENCES)), 2):
        cos = cosine_similarity(vectors[i], vectors[j])
        dot = dot_product(vectors[i], vectors[j])
        euc = euclidean_distance(vectors[i], vectors[j])
        print(f"({i},{j})     {cos:>14.6f}{dot:>16.6f}{euc:>14.6f}")

    print(
        "\nSanity check: since OpenAI embeddings are unit-normalized, cosine similarity "
        "and dot product should be nearly identical for every pair above (per today's lesson). "
        "If they diverge noticeably, something about the vectors isn't normalized as expected — "
        "worth double-checking the model output."
    )


if __name__ == "__main__":
    main()