"""
Week 2, Day 3 — Quantization & Architecture Cost Calculator
===========================================================

The interview-winning artifact for a cost-sensitive employer.

The insight it makes concrete: **the expensive thing is the vectors, not the
graph.** At 1536 dims a float32 vector is 6,144 bytes and the HNSW neighbour
list is ~130. So every serious memory lever compresses the VECTOR, and the
biggest lever by far is product quantization — 6,144 bytes down to 64.

    float32   6,144 B/vector
    PQ-64        64 B/vector      ~96x smaller

At 100M vectors that is the difference between ~700 GB of RAM and ~7 GB, which
is the difference between a five-figure monthly bill and a two-figure one.

It also prices the honest counterpoint: PQ costs recall, and recovering it
needs a rerank stage against full-precision vectors — which must live
somewhere. This calculator charges you for that.

Every price is an editable constant. They are Q1-2026 ballparks; refresh before
quoting. stdlib only.

Usage:
    python quantization_cost_calc.py
    python quantization_cost_calc.py --dims 1536 --recall-slo 0.95
    python quantization_cost_calc.py --build-weeks 1     # agent-assisted build
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# PRICES — verify against current vendor pricing before quoting.
# ----------------------------------------------------------------------------

MANAGED_PG_RAM_GB_MONTH = 11.0   # managed Postgres, RAM-heavy instance class
MANAGED_PG_HA_MULTIPLIER = 2.0   # primary + standby
EC2_RAM_GB_MONTH = 5.5           # self-managed r-family, on-demand
RESERVED_DISCOUNT = 0.60         # 1-yr commitment ≈ 40% off
SSD_GB_MONTH = 0.10              # gp3-class block storage
OBJECT_GB_MONTH = 0.023          # S3-class

# Postgres needs headroom for the whole database; a dedicated ANN process needs
# much less because it does one thing.
PG_RAM_HEADROOM = 1.8
ANN_RAM_HEADROOM = 1.3

# Engineering
ENGINEER_ANNUAL = 180_000
AMORTISE_MONTHS = 24
OPS_FTE_EXTERNAL = 0.20          # running your own ANN service
OPS_FTE_PGVECTOR = 0.05          # absorbed by the existing Postgres function

# Index structure overheads (bytes per vector)
HNSW_GRAPH_BYTES = 136           # 2*M*4 + header, M=16, external lib (4-byte ids)
IVF_LIST_BYTES = 8               # id in an inverted list; no per-vector graph

SCALE_TIERS = [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]


@dataclass
class Encoding:
    name: str
    bytes_per_dim: float | None    # None => fixed bytes_fixed
    bytes_fixed: int | None
    graph_bytes: int
    recall: float                  # representative, WITHOUT rerank
    recall_reranked: float | None  # achievable WITH exact rerank of top-N
    needs_full_vectors: bool
    note: str


def encodings(_dims: int) -> list[Encoding]:
    return [
        Encoding("float32 flat + HNSW", 4, None, HNSW_GRAPH_BYTES,
                 0.98, None, False,
                 "What pgvector stores. Baseline quality, worst memory."),
        Encoding("float16 + HNSW", 2, None, HNSW_GRAPH_BYTES,
                 0.98, None, False,
                 "halfvec. Free 2x, no meaningful recall cost."),
        Encoding("int8 SQ + HNSW", 1, None, HNSW_GRAPH_BYTES,
                 0.97, None, False,
                 "Scalar quantization. 4x for ~1 point of recall — the best "
                 "quality-per-byte on this list, and badly underused."),
        Encoding("PQ-96 (IVF)", None, 96, IVF_LIST_BYTES,
                 0.82, 0.96, True,
                 "Product quantization, 96 subquantizers. Needs rerank."),
        Encoding("PQ-64 (IVF)", None, 64, IVF_LIST_BYTES,
                 0.75, 0.94, True,
                 "Aggressive PQ. ~96x smaller than float32 at 1536 dims."),
        Encoding("binary + rerank", 0.125, None, HNSW_GRAPH_BYTES,
                 0.70, 0.95, True,
                 "1 bit/dim, Hamming distance via popcount. Very fast."),
    ]


def per_vector_bytes(e: Encoding, dims: int) -> float:
    if e.bytes_fixed is not None:
        return e.bytes_fixed + e.graph_bytes
    return dims * e.bytes_per_dim + e.graph_bytes


def _fmt_bytes(b: float) -> str:
    if b >= 1e12: return f"{b / 1e12:.2f} TB"
    if b >= 1e9:  return f"{b / 1e9:.1f} GB"
    if b >= 1e6:  return f"{b / 1e6:.0f} MB"
    return f"{b / 1e3:.0f} KB"


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _monthly_eng(fraction: float) -> float:
    return ENGINEER_ANNUAL / 12.0 * fraction


def _build_amortised(weeks: float) -> float:
    return (ENGINEER_ANNUAL / 52.0) * weeks / AMORTISE_MONTHS


# ----------------------------------------------------------------------------


def report(args: argparse.Namespace) -> None:
    dims = args.dims
    encs = encodings(dims)
    ram_price = EC2_RAM_GB_MONTH * (RESERVED_DISCOUNT if args.reserved else 1.0)

    print("=" * 100)
    print("  QUANTIZATION & ARCHITECTURE COST — WHERE THE MONEY ACTUALLY GOES")
    print("=" * 100)
    print(f"  Dimensions: {dims}   float32 vector = {dims * 4:,} B"
          f"   HNSW graph = {HNSW_GRAPH_BYTES} B ({HNSW_GRAPH_BYTES / (dims * 4):.1%} of it)")
    print(f"  Recall SLO: {args.recall_slo:.2f}   External build: {args.build_weeks} weeks"
          f"   RAM: {_usd(ram_price)}/GB/mo{' (reserved)' if args.reserved else ' (on-demand)'}")
    print()
    print("  THE POINT IN ONE LINE: the graph is ~2% of the memory. Compressing the")
    print("  VECTOR is the only lever that matters.")
    print()

    # ---- encoding comparison ------------------------------------------
    print("-" * 100)
    print("  ENCODINGS — MEMORY PER VECTOR AND WHAT IT COSTS YOU IN RECALL")
    print("-" * 100)
    print(f"  {'Encoding':<24}{'B/vector':>11}{'vs fp32':>10}{'recall':>9}"
          f"{'+rerank':>10}  {'note'}")
    print("  " + "-" * 96)
    base = per_vector_bytes(encs[0], dims)
    for e in encs:
        b = per_vector_bytes(e, dims)
        rr = f"{e.recall_reranked:.2f}" if e.recall_reranked else "  —"
        print(f"  {e.name:<24}{b:>11,.0f}{base / b:>9.0f}x{e.recall:>9.2f}{rr:>10}  {e.note[:38]}")
    print()

    # ---- architecture cost by scale ------------------------------------
    for n in SCALE_TIERS:
        print("-" * 100)
        print(f"  {n:,} VECTORS")
        print("-" * 100)
        print(f"  {'Architecture':<30}{'RAM':>11}{'Infra/mo':>11}{'Eng/mo':>10}"
              f"{'TOTAL/mo':>11}{'TOTAL/yr':>12}{'recall':>9}")
        print("  " + "-" * 96)

        rows = []

        # --- A. pgvector in managed Postgres (float32) -------------------
        pg_bytes = n * (dims * 4 + 2 * 16 * 8 + 24) / 0.9
        pg_ram = pg_bytes * PG_RAM_HEADROOM / 1e9
        pg_infra = pg_ram * MANAGED_PG_RAM_GB_MONTH * MANAGED_PG_HA_MULTIPLIER
        pg_eng = _monthly_eng(OPS_FTE_PGVECTOR)
        rows.append(("pgvector HNSW (managed PG)", pg_bytes * PG_RAM_HEADROOM,
                     pg_infra, pg_eng, 0.98, True))

        # --- B. pgvector with halfvec ------------------------------------
        pgh_bytes = n * (dims * 2 + 2 * 16 * 8 + 24) / 0.9
        pgh_ram = pgh_bytes * PG_RAM_HEADROOM / 1e9
        rows.append(("pgvector halfvec (managed PG)", pgh_bytes * PG_RAM_HEADROOM,
                     pgh_ram * MANAGED_PG_RAM_GB_MONTH * MANAGED_PG_HA_MULTIPLIER,
                     _monthly_eng(OPS_FTE_PGVECTOR), 0.98, True))

        # --- C/D/E. external ANN service, various encodings --------------
        # Postgres still holds the vectors in its HEAP (no index) as the durable
        # source of truth — costs disk, not RAM. That is the recommended variant.
        heap_bytes = n * (dims * 4 + 40)
        heap_disk_cost = heap_bytes / 1e9 * SSD_GB_MONTH
        # Postgres serves hydration: fetch ~10-100 rows BY ID per query. That is a
        # b-tree point lookup, so only the PK index needs to stay resident — not
        # the whole table. Size on the index (~48 B/row), not on row width.
        pg_metadata_ram = max(8.0, n * 48 / 1e9 * PG_RAM_HEADROOM)
        pg_metadata_cost = pg_metadata_ram * MANAGED_PG_RAM_GB_MONTH * MANAGED_PG_HA_MULTIPLIER

        for e in encs:
            if e.name.startswith("float32"):
                continue
            b = per_vector_bytes(e, dims) * n
            ann_ram = b * ANN_RAM_HEADROOM / 1e9
            infra = ann_ram * ram_price + pg_metadata_cost + heap_disk_cost
            eng = _build_amortised(args.build_weeks) + _monthly_eng(OPS_FTE_EXTERNAL)
            recall = e.recall_reranked if e.recall_reranked else e.recall
            rows.append((f"external {e.name}", b * ANN_RAM_HEADROOM,
                         infra, eng, recall, False))

        # --- print ------------------------------------------------------
        eligible = [r for r in rows if r[4] >= args.recall_slo]
        cheapest = min((r[2] + r[3] for r in eligible), default=None)

        for name, ram, infra, eng, recall, _ in rows:
            total = infra + eng
            mark = ""
            if recall < args.recall_slo:
                mark = "  (below SLO)"
            elif cheapest is not None and abs(total - cheapest) < 0.01:
                mark = "  <-- cheapest meeting SLO"
            print(f"  {name:<30}{_fmt_bytes(ram):>11}{_usd(infra):>11}{_usd(eng):>10}"
                  f"{_usd(total):>11}{_usd(total * 12):>12}{recall:>9.2f}{mark}")

        # savings headline
        pg_total = rows[0][2] + rows[0][3]
        if cheapest is not None and cheapest < pg_total:
            print(f"  {'':<30}savings vs pgvector float32: "
                  f"{_usd(pg_total - cheapest)}/mo  "
                  f"({_usd((pg_total - cheapest) * 12)}/yr, "
                  f"{(1 - cheapest / pg_total):.0%} cheaper)")
        print()

    # ---- the honest part ------------------------------------------------
    print("=" * 100)
    print("  WHAT THIS TABLE DOES NOT PRICE")
    print("=" * 100)
    print("  1. PQ recall figures assume a RERANK stage against full-precision")
    print("     vectors. Without it, PQ-64 lands near 0.75 — unusable for most")
    print("     retrieval. The rerank needs the real vectors, which is why the")
    print("     Postgres heap stays in every external row above (as disk, not RAM).")
    print()
    print("  2. The external rows re-create the dual-write problem: freshness lag,")
    print("     reconciliation, a rebuild pipeline, and a service to operate. The")
    print("     eng/mo column covers build + ops but not the incident you will have.")
    print()
    print("  3. You lose SQL WHERE inside the index. Filtering moves to your code,")
    print("     and Day 1's pre-filter dilemma comes back harder.")
    print()
    print("=" * 100)
    print("  HOW TO USE THIS IN AN INTERVIEW")
    print("=" * 100)
    print("  At 1M vectors pgvector wins outright — the external option costs MORE")
    print("  once engineering is priced. Say that first; it shows you optimise for")
    print("  total cost, not for looking clever.")
    print()
    print("  At 100M+ the ordering inverts hard, and the lever is quantization, not")
    print("  the choice of ANN library. 'We cut the vector store bill by an order of")
    print("  magnitude with int8/PQ plus a rerank stage, and kept recall above 0.95'")
    print("  is a sentence that gets you hired at a cost-sensitive company.")
    print()
    print("  Then name the price you paid: a freshness lag, a rebuild pipeline, and")
    print("  a service on the on-call rotation. Engineers who quote only the savings")
    print("  are not trusted with the budget.")
    print("=" * 100)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dims", type=int, default=1536)
    p.add_argument("--recall-slo", type=float, default=0.94)
    p.add_argument("--build-weeks", type=float, default=4.0,
                   help="weeks to build the external ANN service + sync pipeline")
    p.add_argument("--reserved", action="store_true",
                   help="price RAM at reserved-instance rates instead of on-demand")
    report(p.parse_args())


if __name__ == "__main__":
    main()
