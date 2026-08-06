"""
Week 2, Day 3 — The pgvector Mitigation Ladder
==============================================

Before migrating off Postgres, you climb a ladder. This script shows what each
rung actually buys, because the answer is usually "years of runway" and the
migration conversation ends there.

The key structural fact from the lesson: pgvector's HNSW index stores a full
copy of every vector, and at 1536 dims that copy is ~96% of the index. So
every effective lever shrinks the VECTOR, not the graph.

Rungs modelled:
    0. baseline            float32, full dims
    1. halfvec (fp16)      2x,  ~0.3% recall cost
    2. MRL truncation      3x at 1536->512 (Week 1 Day 3), ~2% nDCG cost
    3. halfvec + MRL       6x combined
    4. binary + rerank     ~32x on the index; recall recovered by rescoring

Recall figures are representative planning numbers, not measurements. Measure
on YOUR corpus with a golden query set before committing (Day 5 builds that
harness).

stdlib only.

Usage:
    python mitigation_ladder.py
    python mitigation_ladder.py --ram-gb 256 --target-vectors 50000000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

LINK_BYTES = 8
ELEMENT_OVERHEAD = 24
PAGE_FILL_FACTOR = 0.90
RAM_HEADROOM = 1.8


@dataclass
class Rung:
    name: str
    dims: int
    bytes_per_dim: float
    recall_delta_pct: float  # relative to baseline, negative = worse
    effort: str
    caveat: str = ""


def build_ladder(base_dims: int) -> list[Rung]:
    mrl_dims = max(64, base_dims // 3)
    return [
        Rung("0. baseline (float32)", base_dims, 4.0, 0.0,
             "none"),
        Rung("1. halfvec (fp16)", base_dims, 2.0, -0.3,
             "ALTER COLUMN + REINDEX",
             "pgvector 0.7+; arithmetic still done in fp32"),
        Rung(f"2. MRL truncation -> {mrl_dims}d", mrl_dims, 4.0, -2.0,
             "re-embed + REINDEX",
             "only valid for MRL-trained models (text-embedding-3-*)"),
        Rung(f"3. halfvec + MRL {mrl_dims}d", mrl_dims, 2.0, -2.3,
             "re-embed + ALTER + REINDEX",
             "the sweet spot for most catalogs"),
        Rung("4. binary + exact rerank", base_dims, 0.125, -0.5,
             "two-stage query rewrite",
             "index only; full vectors stay in heap for rescoring top-200"),
    ]


def index_bytes_per_row(dims: int, bytes_per_dim: float, m: int) -> float:
    raw = dims * bytes_per_dim + (2 * m) * LINK_BYTES + ELEMENT_OVERHEAD
    return raw / PAGE_FILL_FACTOR


def report(args: argparse.Namespace) -> None:
    ladder = build_ladder(args.dims)
    base = ladder[0]
    base_per_row = index_bytes_per_row(base.dims, base.bytes_per_dim, args.m)

    print("=" * 96)
    print("  pgvector MITIGATION LADDER")
    print("=" * 96)
    print(f"  Baseline corpus : {args.vectors:,} vectors x {args.dims} dims (float32), m={args.m}")
    print(f"  Instance RAM    : {args.ram_gb:.0f} GB")
    print(f"  Target corpus   : {args.target_vectors:,} vectors")
    print()

    print(f"  {'Rung':<28}{'Index':>10}{'RAM req':>10}{'Gain':>8}"
          f"{'Recall':>9}{'Max corpus':>16}{'Target?':>10}")
    print("  " + "-" * 92)

    for r in ladder:
        per_row = index_bytes_per_row(r.dims, r.bytes_per_dim, args.m)
        index_gb = per_row * args.vectors / 1e9
        ram_req = index_gb * RAM_HEADROOM
        gain = base_per_row / per_row
        max_corpus = int(args.ram_gb * 1e9 / (per_row * RAM_HEADROOM))
        hits_target = "YES" if max_corpus >= args.target_vectors else "no"

        print(f"  {r.name:<28}{index_gb:>9.1f}G{ram_req:>9.1f}G{gain:>7.1f}x"
              f"{r.recall_delta_pct:>8.1f}%{max_corpus:>16,}{hits_target:>10}")

    print()
    print("  " + "-" * 92)
    print("  DETAIL")
    print("  " + "-" * 92)
    for r in ladder[1:]:
        print(f"  {r.name}")
        print(f"      effort : {r.effort}")
        if r.caveat:
            print(f"      caveat : {r.caveat}")
    print()

    # --- runway analysis ---------------------------------------------------
    print("=" * 96)
    print("  RUNWAY ANALYSIS")
    print("=" * 96)

    reachable = [
        r for r in ladder
        if int(args.ram_gb * 1e9
               / (index_bytes_per_row(r.dims, r.bytes_per_dim, args.m) * RAM_HEADROOM))
        >= args.target_vectors
    ]

    if not reachable:
        print(f"  No rung reaches {args.target_vectors:,} vectors on {args.ram_gb:.0f} GB.")
        print("  The in-Postgres ladder is exhausted. Remaining options:")
        print("    - pgvectorscale StreamingDiskANN (index designed to live on NVMe)")
        print("    - partition by tenant/category so no single graph is this large")
        print("    - a bigger instance (check the price before dismissing it)")
        print("    - shard: Citus, or migrate to Milvus")
    else:
        cheapest = reachable[0]
        print(f"  Cheapest rung that reaches {args.target_vectors:,} vectors:")
        print(f"    -> {cheapest.name}")
        print(f"       effort: {cheapest.effort}")
        if cheapest.caveat:
            print(f"       caveat: {cheapest.caveat}")
        print(f"       recall cost: {cheapest.recall_delta_pct:.1f}%")
        print()
        if args.growth_pct_per_year > 0:
            per_row = index_bytes_per_row(cheapest.dims, cheapest.bytes_per_dim, args.m)
            max_corpus = int(args.ram_gb * 1e9 / (per_row * RAM_HEADROOM))
            years = _years_to_reach(args.vectors, max_corpus, args.growth_pct_per_year)
            print(f"  At {args.growth_pct_per_year:.0f}%/yr catalog growth from"
                  f" {args.vectors:,} vectors,")
            if years is None:
                print("  this rung does not run out within a 10-year horizon.")
            else:
                print(f"  this rung lasts ~{years:.1f} years before the next cliff.")

    print()
    print("  THE POINT: rungs 1-3 are configuration changes and a reindex. They")
    print("  routinely buy 6x. A migration to a distributed vector DB is a quarter")
    print("  of engineering plus a permanent sync pipeline (Day 2). Exhaust the")
    print("  ladder first, and be able to show the numbers that say you did.")
    print("=" * 96)


def _years_to_reach(start: int, limit: int, growth_pct: float) -> float | None:
    if start >= limit:
        return 0.0
    n, years = float(start), 0.0
    while n < limit and years < 10.0:
        n *= 1 + growth_pct / 100.0
        years += 1.0
    return None if years >= 10.0 else years


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--vectors", type=int, default=4_000_000)
    p.add_argument("--target-vectors", type=int, default=25_000_000)
    p.add_argument("--dims", type=int, default=1536)
    p.add_argument("--m", type=int, default=16)
    p.add_argument("--ram-gb", type=float, default=64.0)
    p.add_argument("--growth-pct-per-year", type=float, default=40.0)
    report(p.parse_args())


if __name__ == "__main__":
    main()
