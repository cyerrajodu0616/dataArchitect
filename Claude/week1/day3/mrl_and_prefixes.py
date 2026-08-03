"""
Week 1, Day 3 — Deep Dive Code 2: Matryoshka Embeddings (MRL) & Prefix Mechanics

Demonstrates:
  1. Truncating 1536-dim OpenAI MRL embeddings down to 512d & 256d with re-normalization
  2. The impact of prefixes ('query: ' vs 'passage: ') on open-source E5/BGE models

Requires:
  uv add openai numpy python-dotenv sentence-transformers
"""

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence-transformers import SentenceTransformer

load_dotenv()
client = OpenAI()

# --- PART 1: Matryoshka Representation Learning (MRL) Vector Truncation ---
print("=== PART 1: OpenAI MRL Vector Truncation (1536d -> 512d -> 256d) ===")

TEXT_A = "Customer requested a refund on footwear due to sizing issue."
TEXT_B = "The customer returned the shoes because they didn't fit."
TEXT_C = "The warehouse received a new shipment of stainless steel cookware."

def get_openai_vec(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding)

vec_a_1536 = get_openai_vec(TEXT_A)
vec_b_1536 = get_openai_vec(TEXT_B)
vec_c_1536 = get_openai_vec(TEXT_C)

def truncate_and_normalize(vec: np.ndarray, dim: int) -> np.ndarray:
    truncated = vec[:dim]
    return truncated / np.linalg.norm(truncated)

def cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

for dim in [1536, 512, 256]:
    a = truncate_and_normalize(vec_a_1536, dim)
    b = truncate_and_normalize(vec_b_1536, dim)
    c = truncate_and_normalize(vec_c_1536, dim)
    
    sim_ab = cos_sim(a, b)
    sim_ac = cos_sim(a, c)
    
    print(f"\n--- Dimension {dim:4d} ---")
    print(f"  Sim(Shoes Refund, Returned Shoes) : {sim_ab:.5f}")
    print(f"  Sim(Shoes Refund, Cookware)       : {sim_ac:.5f}")
    print(f"  Margin (Related vs Irrelevant)    : {sim_ab - sim_ac:.5f}")


# --- PART 2: Asymmetric Model Prefixes (E5 / BGE) ---
print("\n\n=== PART 2: Asymmetric Model Prefixes (E5 Model) ===")
e5_model = SentenceTransformer("intfloat/e5-small-v2")

query = "shoes return policy"
passage = "Customers may return footwear within 30 days of purchase for a full refund."

# Scenario A: WITH correct prefixes
q_correct = e5_model.encode(f"query: {query}")
p_correct = e5_model.encode(f"passage: {passage}")
sim_with_prefix = float(np.dot(q_correct, p_correct) / (np.linalg.norm(q_correct) * np.linalg.norm(p_correct)))

# Scenario B: WITHOUT prefixes
q_wrong = e5_model.encode(query)
p_wrong = e5_model.encode(passage)
sim_without_prefix = float(np.dot(q_wrong, p_wrong) / (np.linalg.norm(q_wrong) * np.linalg.norm(p_wrong)))

print(f"Similarity WITH 'query:' and 'passage:' prefixes : {sim_with_prefix:.5f}")
print(f"Similarity WITHOUT prefixes                     : {sim_without_prefix:.5f}")
print(f"Accuracy penalty for omitting prefixes           : {(sim_with_prefix - sim_without_prefix):.5f}")
