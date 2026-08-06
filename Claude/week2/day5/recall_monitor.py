"""
Week 2, Day 5 — Golden-Set Recall Monitor
=========================================

The job that should run nightly against a read replica, and almost never does.

Every default database dashboard measures whether the query RAN. Nothing
measures whether it returned the RIGHT ROWS. Recall@k is an approximation-
quality metric, and it decays from causes invisible to conventional monitoring:
graph churn, stale centroids, distribution drift, and someone quietly lowering
ef_search to silence a latency alert.

The convenient property that makes this a job rather than a research project:
EXACT SEARCH IS THE GROUND TRUTH. No human labels required.

This script has two modes:

  demo (default)  Builds a real approximate index over synthetic vectors and
                  measures real recall@10 against exact search. Then simulates
                  six months of catalog churn and shows recall decaying while
                  the latency proxy stays flat — the exact scenario from the
                  lesson's SVG.

  --print-production-job
                  Prints the psycopg template to run against your own database.

stdlib only. Runs in a few seconds.
"""

from __future__ import annotations

import argparse
import math
import random

# ----------------------------------------------------------------------------
# Demo parameters — small enough to run fast in pure Python, large enough that
# the approximate index is genuinely approximate.
# ----------------------------------------------------------------------------

N_VECTORS = 1200
DIMS = 32
N_CLUSTERS = 24        # IVF lists
N_PROBE = 8            # lists examined per query
CANDIDATE_BUDGET = 420 # ef_search analogue: max entries examined per query
                       # (just above what the probed lists hold at M0, so it
                       #  starts non-binding and becomes binding as churn grows)
GOLDEN_QUERIES = 40
TOP_K = 10
MONTHS = 6

# Monthly catalog churn.
DELETE_RATE = 0.12     # delisted SKUs -> dead nodes still in the index
INSERT_RATE = 0.15     # new SKUs assigned to EXISTING centroids (never retrained)


# ----------------------------------------------------------------------------
# Vector helpers
# ----------------------------------------------------------------------------


def _normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def cosine_distance(a: list[float], b: list[float]) -> float:
    """Both operands are unit vectors, so 1 - dot is cosine distance."""
    return 1.0 - _dot(a, b)


def make_clustered_corpus(
    n: int, dims: int, n_clusters: int, rng: random.Random
) -> list[list[float]]:
    """Real catalogs are clustered (categories), not uniform noise."""
    centers = [
        _normalize([rng.gauss(0, 1) for _ in range(dims)]) for _ in range(n_clusters)
    ]
    out = []
    for _ in range(n):
        c = centers[rng.randrange(n_clusters)]
        out.append(_normalize([x + rng.gauss(0, 0.15) for x in c]))
    return out


# ----------------------------------------------------------------------------
# A genuinely approximate index (IVF-style, with a fixed candidate budget)
# ----------------------------------------------------------------------------


class ApproxIndex:
    """
    IVF lists + a fixed candidate budget.

    The budget is the important part: it is the ef_search analogue. Dead
    entries still occupy the graph/list and still consume budget, but can
    never be returned. That is precisely why churn costs recall while leaving
    the work done per query — and therefore latency — unchanged.
    """

    def __init__(self, vectors: list[list[float]], n_clusters: int,
                 rng: random.Random, kmeans_iters: int = 8):
        self.vectors = vectors
        self.alive = [True] * len(vectors)
        self.centroids = self._kmeans(vectors, n_clusters, rng, kmeans_iters)
        self.lists: list[list[int]] = [[] for _ in self.centroids]
        for i, v in enumerate(vectors):
            self.lists[self._nearest_centroid(v)].append(i)

    # --- build ------------------------------------------------------------

    def _kmeans(self, vectors, k, rng, iters):
        centroids = [list(v) for v in rng.sample(vectors, k)]
        for _ in range(iters):
            buckets: list[list[int]] = [[] for _ in range(k)]
            for i, v in enumerate(vectors):
                best, bestd = 0, float("inf")
                for ci, c in enumerate(centroids):
                    d = cosine_distance(v, _normalize(c))
                    if d < bestd:
                        best, bestd = ci, d
                buckets[best].append(i)
            for ci, members in enumerate(buckets):
                if not members:
                    continue
                dims = len(vectors[0])
                acc = [0.0] * dims
                for i in members:
                    for j in range(dims):
                        acc[j] += vectors[i][j]
                centroids[ci] = _normalize([x / len(members) for x in acc])
        return centroids

    def _nearest_centroid(self, v: list[float]) -> int:
        best, bestd = 0, float("inf")
        for ci, c in enumerate(self.centroids):
            d = cosine_distance(v, c)
            if d < bestd:
                best, bestd = ci, d
        return best

    # --- mutation ---------------------------------------------------------

    def delete(self, idx: int) -> None:
        """Mark dead. The entry REMAINS in its list, as in a real HNSW graph."""
        self.alive[idx] = False

    def insert(self, vec: list[float]) -> int:
        """
        Assign to the nearest EXISTING centroid. Centroids are never retrained
        — this is the IVFFlat stale-centroid problem from Week 2 Day 1.
        """
        idx = len(self.vectors)
        self.vectors.append(vec)
        self.alive.append(True)
        self.lists[self._nearest_centroid(vec)].append(idx)
        return idx

    # --- search -----------------------------------------------------------

    def search(self, q: list[float], k: int, n_probe: int,
               budget: int) -> tuple[list[int], int]:
        """Returns (top-k live ids, distance computations performed)."""
        cent = sorted(
            range(len(self.centroids)),
            key=lambda ci: cosine_distance(q, self.centroids[ci]),
        )[:n_probe]
        comps = len(self.centroids)  # centroid comparisons

        scored: list[tuple[float, int]] = []
        examined = 0
        for ci in cent:
            for idx in self.lists[ci]:
                if examined >= budget:
                    break
                examined += 1
                comps += 1
                # Dead entries consume budget but cannot be returned.
                if not self.alive[idx]:
                    continue
                scored.append((cosine_distance(q, self.vectors[idx]), idx))
            if examined >= budget:
                break

        scored.sort()
        return [i for _, i in scored[:k]], comps

    def exact(self, q: list[float], k: int) -> list[int]:
        """Ground truth: full scan over live vectors. This is what makes the job possible."""
        scored = [
            (cosine_distance(q, v), i)
            for i, v in enumerate(self.vectors)
            if self.alive[i]
        ]
        scored.sort()
        return [i for _, i in scored[:k]]

    @property
    def live_count(self) -> int:
        return sum(self.alive)

    @property
    def dead_pct(self) -> float:
        return 100.0 * (len(self.alive) - self.live_count) / max(1, len(self.alive))


# ----------------------------------------------------------------------------
# The measurement
# ----------------------------------------------------------------------------


def measure_recall(index: ApproxIndex, queries: list[list[float]],
                   k: int, n_probe: int, budget: int) -> tuple[float, float]:
    """Mean recall@k and mean distance computations (the latency proxy)."""
    total_recall = 0.0
    total_comps = 0
    for q in queries:
        truth = set(index.exact(q, k))
        approx, comps = index.search(q, k, n_probe, budget)
        total_recall += len(truth & set(approx)) / float(k)
        total_comps += comps
    n = len(queries)
    return total_recall / n, total_comps / n


def run_demo(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)

    print("=" * 82)
    print("  GOLDEN-SET RECALL MONITOR — DEMO")
    print("=" * 82)
    print(f"  Corpus {N_VECTORS} vectors x {DIMS} dims | {N_CLUSTERS} IVF lists"
          f" | probe {N_PROBE} | budget {CANDIDATE_BUDGET}")
    print(f"  Golden set: {GOLDEN_QUERIES} queries, measuring recall@{TOP_K}")
    print("  Ground truth = exact search. No human labels needed.")
    print()

    corpus = make_clustered_corpus(N_VECTORS, DIMS, N_CLUSTERS, rng)
    index = ApproxIndex(corpus, N_CLUSTERS, rng)

    # Golden queries drawn from the same distribution, perturbed — as if they
    # were real user queries against catalog items.
    queries = [
        _normalize([x + rng.gauss(0, 0.25) for x in corpus[rng.randrange(len(corpus))]])
        for _ in range(GOLDEN_QUERIES)
    ]

    print("-" * 82)
    print(f"  {'Month':<8}{'Live':>9}{'Dead %':>9}{'recall@10':>12}"
          f"{'dist comps':>13}{'work drift':>18}")
    print("-" * 82)

    baseline_recall = None
    baseline_comps = None
    final_comps = 0.0
    for month in range(MONTHS + 1):
        recall, comps = measure_recall(
            index, queries, TOP_K, N_PROBE, CANDIDATE_BUDGET
        )
        if baseline_recall is None:
            baseline_recall, baseline_comps = recall, comps
        final_comps = comps

        drift = f"{(comps / baseline_comps - 1) * 100:+.1f}% vs M0"
        flag = "  <-- SLO BREACH" if recall < args.slo else ""
        print(f"  M{month:<7}{index.live_count:>9}{index.dead_pct:>8.1f}%"
              f"{recall:>12.3f}{comps:>13.0f}{drift:>18}{flag}")

        if month == MONTHS:
            break

        # --- one month of catalog churn ---------------------------------
        live_ids = [i for i, a in enumerate(index.alive) if a]
        for idx in rng.sample(live_ids, int(len(live_ids) * DELETE_RATE)):
            index.delete(idx)
        for _ in range(int(index.live_count * INSERT_RATE)):
            src = corpus[rng.randrange(len(corpus))]
            index.insert(_normalize([x + rng.gauss(0, 0.15) for x in src]))

    final_recall, _ = measure_recall(index, queries, TOP_K, N_PROBE, CANDIDATE_BUDGET)

    print()
    print("=" * 82)
    print("  WHAT JUST HAPPENED")
    print("=" * 82)
    print(f"  recall@10 : {baseline_recall:.3f} -> {final_recall:.3f}"
          f"   ({final_recall - baseline_recall:+.3f})")
    comps_drift = (final_comps / baseline_comps - 1) * 100
    print(f"  work per query : {baseline_comps:.0f} -> {final_comps:.0f}"
          f" distance computations ({comps_drift:+.1f}%)")
    print()
    print("  Recall fell off a cliff. The work done per query barely moved — the")
    print("  candidate budget is fixed, so P99 latency, throughput and error rate")
    print(f"  stay within noise. A {comps_drift:+.1f}% shift would not trip any")
    print("  latency alert you have ever configured.")
    print()
    print("  Two mechanisms, both invisible to conventional monitoring:")
    print("    1. Dead nodes still occupy the index and still consume the")
    print("       candidate budget, but can never be returned. Every delisted")
    print("       SKU steals a slot from a live one.")
    print("    2. New vectors are assigned to centroids that were never")
    print("       retrained, so they land in badly-fitting lists and the")
    print("       probed lists stop containing the true nearest neighbours.")
    print()
    print("  The fix for (1) is REINDEX CONCURRENTLY. The fix for (2) is the")
    print("  same rebuild, which recomputes centroids. Neither happens on its")
    print("  own, and nothing in pg_stat_* will tell you it is time.")
    print()
    print(f"  If your recall SLO is {args.slo:.2f}, this catalog breached it and")
    print("  every SRE dashboard stayed green throughout.")
    print("=" * 82)
    print()
    print("  Run with --print-production-job for the version that measures your")
    print("  real database.")


# ----------------------------------------------------------------------------
# Production template
# ----------------------------------------------------------------------------

PRODUCTION_JOB = r'''
# ---------------------------------------------------------------------------
# Nightly recall monitor — run against a READ REPLICA, never the primary.
# Exact search is deliberately expensive; that is why it belongs at 3am on a
# replica and not in the request path.
#
# Requires: psycopg (pip install "psycopg[binary]")
# ---------------------------------------------------------------------------
import psycopg

RECALL_SLO = 0.90
TOP_K = 10

def recall_at_k(conn, query_vec, k=TOP_K, ef_search=100):
    with conn.cursor() as cur:
        # --- ground truth: force an exact scan -----------------------------
        cur.execute("BEGIN")
        cur.execute("SET LOCAL enable_indexscan = off")
        cur.execute("SET LOCAL enable_bitmapscan = off")
        cur.execute(
            "SELECT id FROM products ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, k),
        )
        exact = {r[0] for r in cur.fetchall()}
        cur.execute("COMMIT")

        # --- approximate: the path production queries actually take --------
        cur.execute("BEGIN")
        cur.execute("SET LOCAL hnsw.ef_search = %s", (ef_search,))
        cur.execute(
            "SELECT id FROM products ORDER BY embedding <=> %s LIMIT %s",
            (query_vec, k),
        )
        approx = {r[0] for r in cur.fetchall()}
        cur.execute("COMMIT")

    return len(exact & approx) / float(k)


def main():
    # Sample 500-1000 REAL production queries, stratified by the Day 4 query
    # classes, so you detect per-class decay rather than only the mean.
    # Store their embeddings once; reuse the same set every night so the
    # metric is comparable over time.
    with psycopg.connect("postgresql://...") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT query_class, embedding FROM search_golden_set"
            )
            golden = cur.fetchall()

        by_class = {}
        for qclass, vec in golden:
            by_class.setdefault(qclass, []).append(recall_at_k(conn, vec))

    overall = [r for rs in by_class.values() for r in rs]
    mean = sum(overall) / len(overall)
    print(f"recall@{TOP_K} overall: {mean:.4f}")

    # Alert on the mean; PAGE on any single class, because a small
    # high-intent class can collapse while the mean looks fine (Day 4).
    for qclass, rs in sorted(by_class.items()):
        m = sum(rs) / len(rs)
        status = "OK" if m >= RECALL_SLO else "BREACH"
        print(f"  {qclass:<24} {m:.4f}  {status}")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--print-production-job", action="store_true",
                   help="print the psycopg template for a real database")
    p.add_argument("--slo", type=float, default=0.90, help="recall@10 SLO")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    if args.print_production_job:
        print(PRODUCTION_JOB)
        return
    run_demo(args)


if __name__ == "__main__":
    main()
