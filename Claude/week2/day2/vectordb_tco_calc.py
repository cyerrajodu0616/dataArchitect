"""
Week 2, Day 2 — Vector Store TCO Calculator
===========================================

Four-way total-cost-of-ownership model across scale tiers:

    pgvector (managed Postgres)  |  Pinecone serverless
    Weaviate Cloud               |  self-hosted Milvus on K8s

The point of this script is NOT the vendor prices — those move, and you should
refresh them before quoting a number. The point is the two columns that vendor
comparisons leave out:

    1. the amortised engineering cost of the sync pipeline (outbox, CDC,
       reconciliation) that every non-co-located option forces you to build,
    2. the ongoing on-call/ops load of each additional system.

Change SCALE_TIERS and the pricing constants and re-run. stdlib only.

Usage:
    python vectordb_tco_calc.py
    python vectordb_tco_calc.py --qps 1500 --engineer-cost 190000
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ============================================================================
# ASSUMPTIONS — verify against current vendor pricing before quoting these.
# Figures below are order-of-magnitude ballparks, not live list prices.
# ============================================================================

DIMS = 1536  # OpenAI text-embedding-3-small
BYTES_PER_FLOAT = 4
HNSW_INDEX_OVERHEAD = 0.60  # index adds ~60% on top of raw vectors (Day 1)

# --- Team / engineering ------------------------------------------------------
ENGINEER_FULLY_LOADED_ANNUAL = 180_000  # salary + benefits + overhead
SYNC_PIPELINE_BUILD_WEEKS = 6           # outbox + CDC + reconcile + dashboards
AMORTISE_BUILD_OVER_MONTHS = 24         # spread one-time build across 2 years

# Ongoing ops load, expressed as fraction of one FTE per month.
OPS_FTE = {
    "pgvector": 0.05,   # absorbed by the existing Postgres function
    "pinecone": 0.15,   # no index to run, but the sync pipeline is yours
    "weaviate": 0.25,   # managed cluster + sync pipeline
    "milvus":   0.75,   # K8s + etcd + object store + message queue + sync
}

# --- pgvector on managed Postgres -------------------------------------------
# RAM-dominated: you want the index in shared_buffers. Rule of thumb used here:
# provision RAM ≈ 1.5 × (heap + index) so there is room for everything else.
PG_RAM_HEADROOM = 1.5
PG_COST_PER_GB_RAM_MONTH = 11.0   # managed Postgres, RAM-heavy instance class
PG_COST_PER_GB_DISK_MONTH = 0.13
PG_HA_MULTIPLIER = 2.0            # primary + standby

# --- Pinecone serverless -----------------------------------------------------
PINECONE_STORAGE_PER_GB_MONTH = 0.33
PINECONE_READ_PER_MILLION_RU = 16.0
PINECONE_WRITE_PER_MILLION_WU = 4.0
PINECONE_RU_PER_QUERY = 5.0       # scales with top_k and metadata volume

# --- Weaviate Cloud ----------------------------------------------------------
# Weaviate Cloud bills roughly per million stored dimensions per month.
WEAVIATE_PER_MILLION_DIMS_MONTH = 0.095
WEAVIATE_MIN_MONTHLY = 25.0

# --- Self-hosted Milvus ------------------------------------------------------
# Distributed deployment: query nodes + data nodes + index nodes + coordinator,
# plus etcd, object storage and a message queue.
MILVUS_NODE_COST_MONTH = 320.0     # per general-purpose worker node
MILVUS_GB_RAM_PER_NODE = 64.0
MILVUS_SUPPORT_NODES = 4           # etcd (3) + coordinator, always on
MILVUS_OBJECT_STORAGE_PER_GB_MONTH = 0.023
MILVUS_MQ_MONTH = 250.0            # managed Kafka/Pulsar baseline

# --- Scale tiers to evaluate -------------------------------------------------
SCALE_TIERS = [1_000_000, 4_000_000, 20_000_000, 100_000_000]
DEFAULT_QPS = 500  # PEAK QPS — the number you size infrastructure for
MONTHLY_WRITE_FRACTION = 0.30  # 30% of the catalog is re-embedded each month

# Retail search traffic is heavily diurnal: a weekday evening peak, a near-dead
# overnight trough. Provisioned capacity (pgvector, Milvus) is sized for PEAK.
# Consumption billing (Pinecone) charges for the AVERAGE. Conflating the two is
# the single most common way these comparisons go wrong — using peak QPS as a
# sustained rate turns 500 QPS into 1.3 billion queries/month and inflates the
# consumption options by ~4x.
PEAK_TO_AVERAGE_RATIO = 0.25  # mean QPS ≈ 25% of peak

SECONDS_PER_MONTH = 30 * 24 * 3600


# ============================================================================


@dataclass
class Costs:
    name: str
    infra: float
    sync_build_amortised: float
    ops: float

    @property
    def total(self) -> float:
        return self.infra + self.sync_build_amortised + self.ops


def _monthly_fte_cost(fraction: float) -> float:
    return ENGINEER_FULLY_LOADED_ANNUAL / 12.0 * fraction


def _sync_build_amortised() -> float:
    weekly = ENGINEER_FULLY_LOADED_ANNUAL / 52.0
    return weekly * SYNC_PIPELINE_BUILD_WEEKS / AMORTISE_BUILD_OVER_MONTHS


def raw_gb(n_vectors: int) -> float:
    return n_vectors * DIMS * BYTES_PER_FLOAT / 1e9


def indexed_gb(n_vectors: int) -> float:
    return raw_gb(n_vectors) * (1 + HNSW_INDEX_OVERHEAD)


# --- Per-option models -------------------------------------------------------


def cost_pgvector(n: int, _qps: int) -> Costs:
    total_gb = indexed_gb(n)
    ram_gb = total_gb * PG_RAM_HEADROOM
    infra = (
        ram_gb * PG_COST_PER_GB_RAM_MONTH
        + total_gb * PG_COST_PER_GB_DISK_MONTH
    ) * PG_HA_MULTIPLIER
    # No sync pipeline: the vector is written in the same transaction as the row.
    return Costs("pgvector (managed PG)", infra, 0.0,
                 _monthly_fte_cost(OPS_FTE["pgvector"]))


def cost_pinecone(n: int, peak_qps: int) -> Costs:
    storage = raw_gb(n) * PINECONE_STORAGE_PER_GB_MONTH
    # Consumption billing follows the AVERAGE rate, not the peak you provision for.
    queries_per_month = peak_qps * PEAK_TO_AVERAGE_RATIO * SECONDS_PER_MONTH
    reads = queries_per_month * PINECONE_RU_PER_QUERY / 1e6 * PINECONE_READ_PER_MILLION_RU
    writes = n * MONTHLY_WRITE_FRACTION / 1e6 * PINECONE_WRITE_PER_MILLION_WU
    return Costs("Pinecone serverless", storage + reads + writes,
                 _sync_build_amortised(),
                 _monthly_fte_cost(OPS_FTE["pinecone"]))


def cost_weaviate(n: int, _qps: int) -> Costs:
    million_dims = n * DIMS / 1e6
    infra = max(WEAVIATE_MIN_MONTHLY,
                million_dims * WEAVIATE_PER_MILLION_DIMS_MONTH)
    return Costs("Weaviate Cloud", infra, _sync_build_amortised(),
                 _monthly_fte_cost(OPS_FTE["weaviate"]))


def cost_milvus(n: int, peak_qps: int) -> Costs:
    # Provisioned capacity is sized for PEAK, not average.
    nodes_for_ram = max(1, int(-(-indexed_gb(n) // MILVUS_GB_RAM_PER_NODE)))
    nodes_for_qps = max(1, int(-(-peak_qps // 400)))  # ~400 QPS per query node
    worker_nodes = max(nodes_for_ram, nodes_for_qps) + 2  # + data/index nodes
    infra = (
        (worker_nodes + MILVUS_SUPPORT_NODES) * MILVUS_NODE_COST_MONTH
        + raw_gb(n) * MILVUS_OBJECT_STORAGE_PER_GB_MONTH
        + MILVUS_MQ_MONTH
    )
    return Costs("Milvus (self-hosted)", infra, _sync_build_amortised(),
                 _monthly_fte_cost(OPS_FTE["milvus"]))


# --- Reporting ---------------------------------------------------------------


def _usd(x: float) -> str:
    return f"${x:,.0f}"


def _fits_single_node(n: int) -> tuple[bool, str]:
    """Can pgvector realistically hold this on one machine?"""
    need = indexed_gb(n) * PG_RAM_HEADROOM
    if need <= 128:
        return True, "comfortable"
    if need <= 768:
        return True, "large instance, viable"
    if need <= 2048:
        return True, "very large instance, expensive"
    return False, "EXCEEDS single-node RAM — trigger fires"


def report(qps: int) -> None:
    print("=" * 92)
    print("  VECTOR STORE TCO — MONTHLY, ALL-IN (infra + amortised sync build + ops)")
    print("=" * 92)
    avg_qps = qps * PEAK_TO_AVERAGE_RATIO
    print(f"  Dimensions: {DIMS}   Peak {qps} QPS / avg {avg_qps:.0f} QPS "
          f"(diurnal ratio {PEAK_TO_AVERAGE_RATIO:.0%})   "
          f"Monthly re-embed: {MONTHLY_WRITE_FRACTION:.0%} of catalog")
    print("  Provisioned options sized for PEAK; consumption options billed on AVERAGE.")
    print(f"  Engineer fully loaded: {_usd(ENGINEER_FULLY_LOADED_ANNUAL)}/yr   "
          f"Sync pipeline build: {SYNC_PIPELINE_BUILD_WEEKS} wks "
          f"amortised over {AMORTISE_BUILD_OVER_MONTHS} mo")
    print("  NOTE: vendor prices are ballparks — refresh before quoting.")
    print()

    models = [cost_pgvector, cost_pinecone, cost_weaviate, cost_milvus]

    for n in SCALE_TIERS:
        fits, note = _fits_single_node(n)
        print("-" * 92)
        print(f"  {n:>12,} vectors   "
              f"raw {raw_gb(n):.0f} GB / indexed {indexed_gb(n):.0f} GB   "
              f"[pgvector single-node: {note}]")
        print("-" * 92)
        print(f"  {'Option':<26}{'Infra':>12}{'Sync build':>13}"
              f"{'Ops':>12}{'TOTAL/mo':>13}{'TOTAL/yr':>14}")

        results = [m(n, qps) for m in models]
        cheapest = min(r.total for r in results)

        for r in results:
            flag = "  <-- lowest" if r.total == cheapest else ""
            label = r.name
            if label.startswith("pgvector") and not fits:
                label += " *"
            print(f"  {label:<26}{_usd(r.infra):>12}"
                  f"{_usd(r.sync_build_amortised):>13}{_usd(r.ops):>12}"
                  f"{_usd(r.total):>13}{_usd(r.total * 12):>14}{flag}")

        if not fits:
            print("  * pgvector figure is arithmetic only — at this scale the index no")
            print("    longer fits a single node's RAM, so the row is not a real option.")
        print()

    print("=" * 92)
    print("  HOW TO READ THIS")
    print("=" * 92)
    print("  1. pgvector's advantage is not the infra line — it is the two zeroes:")
    print("     no sync-pipeline build, and ops absorbed by an existing function.")
    print("  2. Pinecone's cost is traffic-shaped. Model your peak, not your mean;")
    print("     a Black Friday spike is a bill, not just a latency event.")
    print("  3. Self-hosted Milvus has poor economics until the scale at which")
    print("     pgvector is no longer viable — which is precisely its use case.")
    print("  4. The crossover point is where you should set the migration trigger,")
    print("     and you should name that number before anyone asks for it.")
    print("=" * 92)


def main() -> None:
    global ENGINEER_FULLY_LOADED_ANNUAL

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--qps", type=int, default=DEFAULT_QPS)
    p.add_argument("--engineer-cost", type=float,
                   default=ENGINEER_FULLY_LOADED_ANNUAL,
                   help="fully loaded annual cost of one engineer")
    args = p.parse_args()

    ENGINEER_FULLY_LOADED_ANNUAL = args.engineer_cost

    report(args.qps)


if __name__ == "__main__":
    main()
