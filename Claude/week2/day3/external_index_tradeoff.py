"""
Week 2, Day 3 — External Index: Base + Delta Tradeoff Model
===========================================================

If you move the ANN index out of Postgres and build it as an offline artifact
(FAISS / usearch / hnswlib), you gain build isolation and lose freshness. The
standard fix is base + delta + tombstones — which is an LSM-tree with vectors
in the leaves, and it has the same tradeoff you already know:

    rebuild often   -> fresh, but you pay build cost constantly
    rebuild rarely  -> cheap builds, but the delta grows until querying it
                       costs more than querying the base did

This script finds where that crossover sits for your numbers, and reports the
freshness lag you are accepting in exchange.

Key insight it makes concrete: **compaction cadence is set by delta size, not
by the clock.** "Rebuild nightly" is an answer only if the delta stays small
enough for 24 hours at your churn rate.

stdlib only.

Usage:
    python external_index_tradeoff.py
    python external_index_tradeoff.py --domain fintech-txn
    python external_index_tradeoff.py --vectors 4000000 --churn-per-day 0.05
"""

from __future__ import annotations

import argparse

# ----------------------------------------------------------------------------
# Cost model. These are planning heuristics — measure your own build rate.
# ----------------------------------------------------------------------------

# Offline build throughput (vectors/sec). Far higher than in-Postgres HNSW
# builds because there is no WAL, no MVCC, no page format, and you can use a
# GPU. This IS the reason to move the build out.
BUILD_RATE_CPU = 20_000.0      # FAISS/hnswlib on a many-core box
BUILD_RATE_GPU = 250_000.0     # FAISS GPU index build

# Query cost. Searching the base is sublinear (graph/IVF). Searching the delta
# is effectively a flat scan while it is small — cheap per vector, but linear,
# so it overtakes the base once the delta grows.
BASE_QUERY_US = 800.0          # microseconds to search a well-built base index
DELTA_US_PER_1K = 45.0         # microseconds per 1,000 vectors in a flat delta

DOMAIN_PRESETS = {
    "retail": dict(vectors=4_000_000, churn_per_day=0.03, gpu=False,
                   note="Catalog edits and new SKUs. A nightly rebuild keeps the "
                        "delta small; freshness lag of a few hours is usually fine "
                        "for search but NOT for price/stock, which should come from "
                        "Postgres at hydration time anyway."),
    "insurance": dict(vectors=200_000, churn_per_day=0.001, gpu=False,
                      note="Filed forms change rarely and the corpus is append-only. "
                           "Build is trivial at this size — the reason to use an "
                           "external artifact here is REPRODUCIBILITY for audit, not "
                           "performance."),
    "fintech-txn": dict(vectors=2_000_000_000, churn_per_day=0.02, gpu=True,
                        note="Transactions are append-heavy and naturally batch-"
                             "oriented. GPU build is the only affordable option at "
                             "this size. Fraud scoring needs recent transactions, so "
                             "the delta path is load-bearing, not a nicety."),
}


def build_seconds(n: int, gpu: bool) -> float:
    return n / (BUILD_RATE_GPU if gpu else BUILD_RATE_CPU)


def query_us(delta_vectors: float) -> float:
    """Total query cost: base search + flat scan of the delta."""
    return BASE_QUERY_US + (delta_vectors / 1000.0) * DELTA_US_PER_1K


def report(args: argparse.Namespace) -> None:
    n = args.vectors
    churn = args.churn_per_day
    per_day = n * churn
    per_hour = per_day / 24.0

    full_build_s = build_seconds(n, args.gpu)

    print("=" * 84)
    print("  EXTERNAL INDEX — BASE + DELTA TRADEOFF")
    print("=" * 84)
    if args.domain:
        print(f"  Domain          : {args.domain.upper()}")
    print(f"  Corpus          : {n:,} vectors")
    print(f"  Churn           : {churn:.1%}/day = {per_day:,.0f} vectors/day"
          f" ({per_hour:,.0f}/hour)")
    print(f"  Build hardware  : {'GPU' if args.gpu else 'CPU'}"
          f"  ({BUILD_RATE_GPU if args.gpu else BUILD_RATE_CPU:,.0f} vectors/sec)")
    print(f"  Full base build : {full_build_s / 60:,.1f} min")
    print(f"  Baseline query  : {BASE_QUERY_US:,.0f} us (base only, delta empty)")
    print()

    print("-" * 84)
    print("  REBUILD CADENCE vs QUERY COST vs FRESHNESS")
    print("-" * 84)
    print(f"  {'Cadence':<14}{'Delta at cutover':>19}{'Query us':>11}"
          f"{'vs base':>10}{'Build/day':>12}{'Worst lag':>12}")
    print("  " + "-" * 80)

    cadences = [
        ("15 min", 0.25), ("hourly", 1.0), ("4 hours", 4.0),
        ("12 hours", 12.0), ("nightly", 24.0), ("weekly", 168.0),
    ]

    recommended = None
    for label, hours in cadences:
        delta = per_hour * hours
        q = query_us(delta)
        ratio = q / BASE_QUERY_US
        builds_per_day = 24.0 / hours
        build_load = full_build_s * builds_per_day / 86400.0  # fraction of a machine

        flag = ""
        if ratio > args.max_query_inflation:
            flag = "  <-- delta too big"
        elif recommended is None:
            recommended = (label, hours, delta, q, ratio, build_load)
            flag = "  <-- cheapest that holds"

        lag = f"{hours * 60:.0f} min" if hours < 1 else f"{hours:.0f} h"
        print(f"  {label:<14}{delta:>18,.0f}{q:>11,.0f}{ratio:>9.2f}x"
              f"{build_load:>11.2f}x{lag:>12}{flag}")

    print()
    print("-" * 84)
    print("  READING IT")
    print("-" * 84)
    print(f"  'vs base' is query cost relative to a freshly-built index. Your")
    print(f"  threshold is {args.max_query_inflation:.2f}x — past that, the delta scan")
    print("  dominates and you are paying for staleness twice: slower AND older.")
    print()
    print("  'Build/day' is the fraction of one build machine kept busy. Above 1.0x")
    print("  you cannot rebuild fast enough to sustain the cadence at all.")
    print()

    if recommended:
        label, hours, delta, q, ratio, load = recommended
        print("=" * 84)
        print("  RECOMMENDATION")
        print("=" * 84)
        print(f"  Rebuild cadence : {label}")
        print(f"  Delta at cutover: {delta:,.0f} vectors ({delta / n:.3%} of corpus)")
        print(f"  Query cost      : {q:,.0f} us ({ratio:.2f}x a fresh base)")
        print(f"  Build machines  : {load:.2f}x sustained")
        lag = f"{hours * 60:.0f} minutes" if hours < 1 else f"{hours:.0f} hours"
        print(f"  FRESHNESS LAG   : up to {lag} — this is what you are trading away")
        print()
        print("  The lag is the honest cost. Note what it means in practice: anything")
        print("  the user must see immediately (price, stock, balance, policy status)")
        print("  has to come from Postgres at hydration time, NOT from index metadata.")
        print("  Use the external index for RANKING only, and let the source of truth")
        print("  supply every field the answer actually depends on.")
    else:
        print("=" * 84)
        print("  NO CADENCE WORKS")
        print("=" * 84)
        print("  Even the most frequent rebuild leaves a delta that inflates query cost")
        print(f"  past {args.max_query_inflation:.2f}x. Options:")
        print("    - index the delta properly instead of flat-scanning it")
        print("    - shard so each base index is smaller and faster to rebuild")
        print("    - keep the index in Postgres, where writes update it incrementally")

    if args.note:
        print()
        print("  DOMAIN NOTE")
        words, line = args.note.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 74:
                print(f"    {line}"); line = w
            else:
                line = f"{line} {w}".strip()
        print(f"    {line}")

    print()
    print("=" * 84)
    print("  THE POINT")
    print("=" * 84)
    print("  Compaction cadence is set by DELTA SIZE, not by the clock. 'Rebuild")
    print("  nightly' is an answer only if the delta stays small enough for 24 hours")
    print("  at your churn rate — which is a property of your data, not your cron.")
    print()
    print("  This is an LSM-tree: base + delta + tombstones + compaction. You have")
    print("  reasoned about write amplification and compaction cadence before; this")
    print("  is the same tradeoff with vectors in the leaves.")
    print("=" * 84)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--vectors", type=int)
    p.add_argument("--churn-per-day", type=float,
                   help="fraction of the corpus inserted or updated per day")
    p.add_argument("--gpu", action="store_true", default=None,
                   help="build on a GPU instead of CPU")
    p.add_argument("--max-query-inflation", type=float, default=1.5,
                   help="how much slower than a fresh base you will tolerate")
    args = p.parse_args()

    base = dict(vectors=4_000_000, churn_per_day=0.03, gpu=False, note=None)
    if args.domain:
        base.update(DOMAIN_PRESETS[args.domain])
    args.note = base.pop("note")
    for k, v in base.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    report(args)


if __name__ == "__main__":
    main()
