"""
Week 1, Day 2 — 2D Dimensionality Reduction & Visualization

Projects 1536-dim OpenAI embeddings into 2D space using PCA (Principal Component Analysis)
and MDS (Multidimensional Scaling), and plots them with matplotlib.

Requires:
    uv add matplotlib scikit-learn openai numpy python-dotenv

Run:
    python week1/visualizeEmbeddings.py
"""

import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

client = OpenAI()

SENTENCES = [
    "The customer returned the shoes because they didn't fit.",
    "Customer requested a refund on footwear due to sizing issue.",
    "The warehouse received a new shipment of stainless steel cookware.",
]

MODEL = "text-embedding-3-small"


def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model=MODEL, input=text)
    return np.array(resp.data[0].embedding)


def plot_2d(coords: np.ndarray, labels: list[str], title: str, filename: str):
    """Generates a 2D scatter plot with labels and vector lines from origin."""
    plt.figure(figsize=(9, 7))
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    
    # Plot points
    for i, (x, y) in enumerate(coords):
        short_label = f"[{i}] {labels[i][:35]}..."
        plt.scatter(x, y, color=colors[i % len(colors)], s=120, zorder=5, label=short_label)
        
        # Annotate each point
        plt.annotate(
            f"Point {i}\n({x:.3f}, {y:.3f})",
            (x, y),
            textcoords="offset points",
            xytext=(10, 10),
            ha="left",
            fontsize=9,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[i % len(colors)], alpha=0.8)
        )
        
        # Draw vector from origin (0,0) to point
        plt.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=colors[i % len(colors)], alpha=0.3)

    # Draw lines connecting the pairs to visualize distance
    for (i, j) in itertools.combinations(range(len(coords)), 2):
        x_values = [coords[i][0], coords[j][0]]
        y_values = [coords[i][1], coords[j][1]]
        dist = np.linalg.norm(coords[i] - coords[j])
        plt.plot(x_values, y_values, 'k--', alpha=0.3, label=f"Dist ({i},{j}): {dist:.3f}")

    plt.axhline(0, color='gray', linewidth=0.8, linestyle=':')
    plt.axvline(0, color='gray', linewidth=0.8, linestyle=':')
    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Component 1", fontsize=11)
    plt.ylabel("Component 2", fontsize=11)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout()
    
    out_path = os.path.join(os.path.dirname(__file__), filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved plot to: {out_path}")


def main():
    print(f"Fetching embeddings from {MODEL} for {len(SENTENCES)} sentences...")
    vectors = np.array([get_embedding(s) for s in SENTENCES])
    print(f"Original shape: {vectors.shape} (3 vectors of {vectors.shape[1]} dimensions)\n")

    # 1. PCA Reduction (Principal Component Analysis)
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(vectors)
    print("PCA 2D Coordinates:")
    for i, (x, y) in enumerate(coords_pca):
        print(f"  Sentence [{i}]: ({x:.4f}, {y:.4f})")
    print(f"  Explained Variance Ratio: {pca.explained_variance_ratio_.sum() * 100:.2f}%\n")

    # 2. MDS Reduction (Multidimensional Scaling - preserves pairwise distances)
    mds = MDS(n_components=2, random_state=42, normalized_stress='auto')
    coords_mds = mds.fit_transform(vectors)

    # Generate and save plots
    plot_2d(coords_pca, SENTENCES, "2D PCA Projection of Sentence Embeddings", "embeddings_2d_pca.png")
    plot_2d(coords_mds, SENTENCES, "2D MDS Projection of Sentence Embeddings", "embeddings_2d_mds.png")


if __name__ == "__main__":
    main()
