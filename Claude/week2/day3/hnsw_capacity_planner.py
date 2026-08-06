"""
Week 2, Day 3 — pgvector HNSW Capacity Planner
==============================================

Answers the only question that matters when someone asks "will pgvector handle
this?": WHICH WALL DO I HIT FIRST, and at what corpus size?

Models the five walls from the lesson:
    1. RAM      — does the HNSW index stay cached? (a cliff, not a slope)
    2. CPU      — QPS ceiling, and how many cores search steals from OLTP
    3. Writes   — insert cost, graph decay, REINDEX duration
    4. Replicas — read scaling is replication, so cost is linear
    5. Connections — Little's Law, and how the RAM cliff compounds into one

The memory model (the part Day 1's rule of thumb got wrong):
pgvector's HNSW index stores a FULL COPY of every vector, because traversal
needs the vector at each node to compute distance. So the index is ~105-115%
of the heap at 1536 dims, not the 30-60% a pointer-based index would cost.

stdlib only.

Usage:
    python hnsw_capacity_planner.py
    python hnsw_capacity_planner.py --vectors 25000000 --ram-gb 256 --qps 2000
    python hnsw_capacity_planner.py --dims 512 --halfvec
"""

from __future__ import annotations

import argparse

# ----------------------------------------------------------------------------
# Model constants — planning heuristics. Validate against your own hardware
# with pgbench before quoting any of these in a design doc.
# ----------------------------------------------------------------------------

LINK_BYTES = 8           # one HNSW neighbour link (ItemPointer + alignment)
ELEMENT_OVERHEAD = 24    # per-element header in the index
HEAP_TUPLE_OVERHEAD = 40 # tuple header + a few small columns alongside the vector
PAGE_FILL_FACTOR = 0.90  # usable fraction of an 8 KB page

# RAM you actually need = index + heap working set + Postgres + OS.
RAM_HEADROOM = 1.8

# Latency regimes (ms) for ~100 serially-dependent graph hops.
LATENCY_CACHED_MS = 2.0
LATENCY_NVME_MS = 10.0
LATENCY_NETWORK_SSD_MS = 80.0

# Throughput heuristic at 1536 dims / ef_search=100, index fully cached.
QPS_PER_CORE_AT_1536 = 300.0

# ----------------------------------------------------------------------------
# Domain presets. The point of these is that the SAME model produces three
# genuinely different architectures — that is what makes it a model rather than
# a memorised threshold.
# ----------------------------------------------------------------------------

DOMAIN_PRESETS = {
    "retail": dict(
        vectors=4_000_000, dims=1536, qps=500, ram_gb=64.0, cores=16,
        monthly_churn=0.30, storage="network",
        note="Product catalog search. The middle case: fits comfortably, but "
             "search shares cores with checkout — CPU is the first wall.",
    ),
    "insurance": dict(
        vectors=200_000, dims=1536, qps=20, ram_gb=32.0, cores=8,
        monthly_churn=0.02, storage="network",
        note="Policy forms, riders, underwriting guidelines. Corpus is tiny and "
             "traffic is low; no wall is reachable. Churn is near zero because "
             "regulated corpora are append-only — superseded forms are retained, "
             "not deleted, so there is little graph decay. Spend the engineering "
             "on filter correctness and auditability instead.",
    ),
    "fintech-txn": dict(
        vectors=2_000_000_000, dims=768, qps=4_000, ram_gb=512.0, cores=64,
        monthly_churn=0.60, storage="network",
        note="Transaction/merchant embeddings for fraud similarity. Every wall at "
             "once. Replication cannot help — each replica needs the whole index. "
             "This is the case where the pgvector answer genuinely flips.",
    ),
    "fintech-docs": dict(
        vectors=150_000, dims=1536, qps=30, ram_gb=32.0, cores=8,
        monthly_churn=0.05, storage="network",
        note="Disclosures, dispute procedures, KYC guidance. NOT the same workload "
             "as transaction embeddings — conflating them is a common interview "
             "stumble. This one looks like insurance, not like fintech-txn.",
    ),
}

# HNSW build throughput per parallel worker (vectors/sec) at 1536 dims, with
# maintenance_work_mem large enough that the build stays in memory. Building a
# graph is not a linear scan: each insert runs a search plus link updates on
# ~2m existing nodes, so throughput is far lower than people expect. If the
# build spills to disk (maintenance_work_mem too small) divide these by ~5.
BUILD_VECTORS_PER_SEC_PER_WORKER_AT_1536 = 200.0


def _gb(b: float) -> float:
    return b / 1e9


# ----------------------------------------------------------------------------
# Sizing
# ----------------------------------------------------------------------------


def vector_bytes(dims: int, halfvec: bool) -> int:
    return dims * (2 if halfvec else 4)


def heap_bytes_per_row(dims: int, halfvec: bool) -> int:
    return vector_bytes(dims, halfvec) + HEAP_TUPLE_OVERHEAD


def hnsw_bytes_per_row(dims: int, m: int, halfvec: bool) -> float:
    """Vector copy + layer-0 neighbour list (2m links) + header, over fill factor."""
    raw = vector_bytes(dims, halfvec) + (2 * m) * LINK_BYTES + ELEMENT_OVERHEAD
    return raw / PAGE_FILL_FACTOR


def sizing(n: int, dims: int, m: int, halfvec: bool) -> dict:
    heap = n * heap_bytes_per_row(dims, halfvec)
    index = n * hnsw_bytes_per_row(dims, m, halfvec)
    return {
        "heap_gb": _gb(heap),
        "index_gb": _gb(index),
        "total_gb": _gb(heap + index),
        "ram_needed_gb": _gb(index) * RAM_HEADROOM,
        "index_pct_of_heap": index / heap * 100.0,
        "link_pct_of_index": (2 * m) * LINK_BYTES / hnsw_bytes_per_row(dims, m, halfvec) * 100.0,
    }


# ----------------------------------------------------------------------------
# The five walls
# ----------------------------------------------------------------------------


def wall_ram(s: dict, ram_gb: float, storage: str) -> dict:
    fits = s["ram_needed_gb"] <= ram_gb
    if fits:
        latency, regime = LATENCY_CACHED_MS, "CACHED — design target"
    elif storage == "nvme":
        latency, regime = LATENCY_NVME_MS, "SPILLING to local NVMe — degraded"
    else:
        latency, regime = LATENCY_NETWORK_SSD_MS, "SPILLING to network SSD — P99 collapse"

    # Corpus size at which this instance falls off the cliff.
    per_row_ram = s["ram_needed_gb"] / max(1, s["_n"]) * 1e9
    max_vectors = int(ram_gb * 1e9 / per_row_ram)

    return {
        "fits": fits,
        "latency_ms": latency,
        "regime": regime,
        "headroom_pct": (ram_gb - s["ram_needed_gb"]) / ram_gb * 100.0,
        "max_vectors_on_this_instance": max_vectors,
    }


def wall_cpu(dims: int, qps: int, cores: int) -> dict:
    # Distance cost scales with dimensionality (memory-bandwidth bound).
    qps_per_core = QPS_PER_CORE_AT_1536 * (1536.0 / dims)
    ceiling = qps_per_core * cores
    cores_used = qps / qps_per_core
    return {
        "qps_per_core": qps_per_core,
        "ceiling_qps": ceiling,
        "cores_consumed": cores_used,
        "pct_of_instance": cores_used / cores * 100.0,
        "saturated": qps > ceiling,
    }


def wall_writes(n: int, dims: int, monthly_churn: float, workers: int) -> dict:
    rate = BUILD_VECTORS_PER_SEC_PER_WORKER_AT_1536 * (1536.0 / dims)
    build_s = n / (rate * max(1, workers))
    return {
        "reindex_minutes": build_s / 60.0,
        "reindex_transient_disk_note": "needs ~2x index size free during CONCURRENTLY",
        "rows_churned_per_month": int(n * monthly_churn),
        "dead_nodes_per_month": int(n * monthly_churn),
    }


def wall_replicas(s: dict, qps: int, cpu: dict, ram_gb: float) -> dict:
    needed = max(1, int(-(-qps // max(1.0, cpu["ceiling_qps"]))))
    return {
        "replicas_needed": needed,
        "ram_per_replica_gb": s["ram_needed_gb"],
        "total_ram_gb": needed * max(ram_gb, s["ram_needed_gb"]),
        "note": "each replica holds a FULL copy — replication scales QPS, not corpus",
    }


def wall_connections(qps: int, latency_ms: float) -> dict:
    concurrent = qps * (latency_ms / 1000.0)
    return {
        "concurrent_connections": concurrent,
        "needs_pooler": concurrent > 50,
    }


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------


def report(args: argparse.Namespace) -> None:
    n, dims, m = args.vectors, args.dims, args.m
    s = sizing(n, dims, m, args.halfvec)
    s["_n"] = n

    ram = wall_ram(s, args.ram_gb, args.storage)
    cpu = wall_cpu(dims, args.qps, args.cores)
    wr = wall_writes(n, dims, args.monthly_churn, args.build_workers)
    rep = wall_replicas(s, args.qps, cpu, args.ram_gb)
    conn = wall_connections(args.qps, ram["latency_ms"])

    vtype = "halfvec (fp16)" if args.halfvec else "vector (float32)"

    print("=" * 78)
    print("  pgvector HNSW CAPACITY PLAN")
    print("=" * 78)
    if getattr(args, "domain", None):
        print(f"  Domain      : {args.domain.upper()}")
    print(f"  Corpus      : {n:,} vectors x {dims} dims  [{vtype}]  m={m}")
    print(f"  Instance    : {args.cores} vCPU / {args.ram_gb:.0f} GB RAM / {args.storage}")
    print(f"  Peak load   : {args.qps:,} QPS")
    print()

    print("-" * 78)
    print("  SIZING")
    print("-" * 78)
    print(f"  Heap                       : {s['heap_gb']:>8.1f} GB")
    print(f"  HNSW index                 : {s['index_gb']:>8.1f} GB"
          f"   ({s['index_pct_of_heap']:.0f}% of heap)")
    print(f"  Total on disk              : {s['total_gb']:>8.1f} GB")
    print(f"  RAM needed to stay cached  : {s['ram_needed_gb']:>8.1f} GB"
          f"   ({RAM_HEADROOM}x index)")
    lp = s["link_pct_of_index"]
    if lp < 8:
        note = "the vector copy dominates — shrink the vector, not the graph"
    else:
        note = "graph overhead is material here — raising m is expensive at low dims"
    print(f"  Neighbour links are {lp:.1f}% of the index — {note}")
    print()

    print("-" * 78)
    print("  THE FIVE WALLS")
    print("-" * 78)

    status = "OK " if ram["fits"] else "HIT"
    print(f"  [{status}] 1. RAM        {ram['regime']}")
    print(f"            expected P99 ~{ram['latency_ms']:.0f} ms"
          f"   |  headroom {ram['headroom_pct']:+.0f}%")
    print(f"            this instance holds up to "
          f"{ram['max_vectors_on_this_instance']:,} vectors")

    status = "HIT" if cpu["saturated"] else "OK "
    print(f"  [{status}] 2. CPU        ceiling ~{cpu['ceiling_qps']:,.0f} QPS"
          f"  ({cpu['qps_per_core']:,.0f} QPS/core x {args.cores})")
    print(f"            at {args.qps:,} QPS you consume {cpu['cores_consumed']:.1f} cores"
          f" ({cpu['pct_of_instance']:.0f}% of the box)")
    if cpu["pct_of_instance"] > 25 and not cpu["saturated"]:
        print("            ^ shared with OLTP: under a spike, CHECKOUT fails before search")

    rb = wr["reindex_minutes"]
    rb_str = f"{rb:.0f} min" if rb < 120 else f"{rb / 60:.1f} h"
    print(f"  [   ] 3. WRITES     REINDEX CONCURRENTLY ~{rb_str}"
          f" ({args.build_workers} workers)")
    print(f"            {wr['dead_nodes_per_month']:,} dead graph nodes/month at"
          f" {args.monthly_churn:.0%} churn")
    print(f"            {wr['reindex_transient_disk_note']}"
          f" (+{s['index_gb']:.0f} GB)")

    status = "HIT" if rep["replicas_needed"] > 1 else "OK "
    print(f"  [{status}] 4. REPLICAS   {rep['replicas_needed']} node(s) to serve"
          f" {args.qps:,} QPS")
    print(f"            {rep['ram_per_replica_gb']:.0f} GB RAM EACH —"
          " replication scales QPS, not corpus size")

    status = "HIT" if conn["needs_pooler"] else "OK "
    print(f"  [{status}] 5. CONNECTIONS ~{conn['concurrent_connections']:.0f} concurrent"
          f" (Little's Law at {ram['latency_ms']:.0f} ms)")
    if conn["needs_pooler"]:
        print("            PgBouncer in transaction mode is mandatory here")
    print()

    # --- which wall first --------------------------------------------------
    print("=" * 78)
    print("  VERDICT")
    print("=" * 78)

    first: list[str] = []
    if not ram["fits"]:
        first.append("RAM — the index no longer fits; this is the cliff")
    if cpu["saturated"]:
        first.append("CPU — QPS exceeds the instance ceiling")
    elif cpu["pct_of_instance"] > 40:
        first.append("CPU — search is crowding out OLTP on shared cores")
    if conn["needs_pooler"]:
        first.append("CONNECTIONS — pool exhaustion risk")

    if not first:
        pct = n / max(1, ram["max_vectors_on_this_instance"]) * 100.0
        print(f"  No wall hit. Using {pct:.0f}% of this instance's vector capacity.")
        print(f"  Runway: {ram['max_vectors_on_this_instance'] - n:,} more vectors"
              " before the RAM cliff.")
        print()
        print("  Recommended next move when you approach it: halfvec (2x) then")
        print("  MRL truncation to 512 dims = ~4.9x runway. See mitigation_ladder.py")
    else:
        print("  First wall(s) hit:")
        for f in first:
            print(f"    -> {f}")
        print()
        if not ram["fits"]:
            # Best case on the whole ladder is binary quantization + exact rerank.
            vb = vector_bytes(dims, args.halfvec)
            fixed = (2 * m) * LINK_BYTES + ELEMENT_OVERHEAD
            best_per_row = (vb / 32 + fixed) / PAGE_FILL_FACTOR
            best_ram = _gb(n * best_per_row) * RAM_HEADROOM
            if best_ram <= args.ram_gb:
                print("  Before migrating, run mitigation_ladder.py — halfvec + MRL")
                print("  truncation typically buys ~4.9x and moves this out by years.")
            else:
                print(f"  LADDER EXHAUSTED: even binary quantization + exact rerank")
                print(f"  needs ~{best_ram:,.0f} GB, against {args.ram_gb:,.0f} GB available.")
                print("  No in-Postgres lever closes a gap this size. Sharding is the")
                print("  only structural answer — and replication cannot substitute,")
                print("  because every replica needs the entire index.")
        elif cpu["saturated"] or cpu["pct_of_instance"] > 40:
            print("  This is an ISOLATION problem, not a database-choice problem.")
            print("  Move search to a dedicated read replica first.")

    if getattr(args, "note", None):
        print()
        print("  DOMAIN NOTE")
        words, line = args.note.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(f"    {line}")
                line = w
            else:
                line = f"{line} {w}".strip()
        print(f"    {line}")
    print("=" * 78)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS),
                   help="load a domain preset; individual flags still override it")
    p.add_argument("--vectors", type=int)
    p.add_argument("--dims", type=int)
    p.add_argument("--m", type=int, default=16, help="HNSW max links per layer")
    p.add_argument("--halfvec", action="store_true", help="use fp16 instead of float32")
    p.add_argument("--ram-gb", type=float)
    p.add_argument("--cores", type=int)
    p.add_argument("--qps", type=int)
    p.add_argument("--storage", choices=["nvme", "network"])
    p.add_argument("--monthly-churn", type=float)
    p.add_argument("--build-workers", type=int, default=8)
    args = p.parse_args()

    # Preset supplies defaults; anything given explicitly on the CLI wins.
    base = dict(vectors=4_000_000, dims=1536, qps=500, ram_gb=64.0, cores=16,
                monthly_churn=0.30, storage="network", note=None)
    if args.domain:
        base.update(DOMAIN_PRESETS[args.domain])
    args.note = base.pop("note")
    for k, v in base.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    report(args)


if __name__ == "__main__":
    main()
