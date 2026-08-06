"""
Week 2, Day 2 — Dual-Write Drift Simulator
==========================================

Quantifies the cost that no vector-DB pricing page shows: what happens to
correctness when your embeddings live in a *different* system from your source
of truth.

Two architectures, same retail catalog workload:

  A) CO-LOCATED  (pgvector)  — the SKU row and its embedding are written in one
                               Postgres transaction. Drift is impossible.
  B) DISAGGREGATED (Pinecone / Weaviate / Milvus) — the row goes to Postgres,
                               the vector goes over the network to a second
                               system. Now you own an outbox + CDC pipeline,
                               retries, and a reconciliation job.

The simulator models a day of catalog churn against architecture B and reports:
  - stale-vector minutes  (embedding does not match the current description)
  - orphaned vectors      (row deleted, vector delete never landed)
  - the fraction of customer searches that would hit an incorrect result

Everything here is stdlib. Deterministic by default (fixed seed) so two runs
are comparable; pass a different --seed to explore variance.

Usage:
    python dual_write_drift_sim.py
    python dual_write_drift_sim.py --hours 24 --failure-rate 0.02 --no-reconcile
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Workload assumptions — edit these to match your own catalog
# ----------------------------------------------------------------------------

CATALOG_SIZE = 4_000_000  # SKUs with embeddings

# Catalog churn per hour. Retail catalogs are far more mutable than people
# assume: description edits, re-categorisation, seasonal re-listing, delisting.
UPDATES_PER_HOUR = 6_000  # description/attribute edits -> embedding must change
DELETES_PER_HOUR = 400  # delistings -> vector must be removed
INSERTS_PER_HOUR = 1_200  # new SKUs

# Sync pipeline characteristics (outbox -> CDC -> vector DB upsert).
CDC_LAG_MEAN_S = 45.0  # mean end-to-end lag, source commit -> vector visible
CDC_LAG_TAIL_S = 900.0 # occasional consumer-lag spike (rebalance, backpressure)
CDC_TAIL_PROB = 0.03   # how often the tail case happens

UPSERT_FAILURE_RATE = 0.005  # transient failures at the vector DB API (0.5%)
MAX_RETRIES = 3
RETRY_BACKOFF_S = [30, 120, 600]  # exponential-ish backoff between retries

# Poison writes: fail *deterministically* on every retry, so they land in the
# dead-letter queue rather than converging. Schema mismatch after a catalog
# migration, payload over the vendor's metadata size limit, unknown namespace
# for a newly-onboarded storefront, dimension mismatch after a model swap.
# This — not transient 5xx — is where real orphaned vectors come from: retrying
# independent transient failures 4× makes permanent loss vanishingly rare, but
# retrying a malformed payload 4× just produces four identical rejections.
POISON_RATE = 0.0004  # 4 in 10,000 writes are structurally undeliverable

# Reconciliation job: the thing you have to build and then keep working.
RECONCILE_ENABLED = True
RECONCILE_EVERY_HOURS = 24  # nightly full sweep

# Read workload — used to convert drift into a customer-visible number.
SEARCHES_PER_HOUR = 500 * 3600  # 500 QPS
RESULTS_PER_SEARCH = 10


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------


@dataclass
class DriftEvent:
    """A single write that must reach the external vector store."""

    kind: str  # "update" | "delete" | "insert"
    committed_at_s: float  # when Postgres committed the source row
    landed_at_s: float | None  # when the vector store reflected it (None = never)
    attempts: int
    poison: bool = False  # structurally undeliverable -> dead-letter queue
    recovered_by_reconcile: bool = False


@dataclass
class SimResult:
    events: list[DriftEvent] = field(default_factory=list)
    horizon_s: float = 0.0

    # --- derived metrics -----------------------------------------------------

    @property
    def total_events(self) -> int:
        return len(self.events)

    @property
    def permanently_lost(self) -> list[DriftEvent]:
        """Writes that exhausted retries and were never reconciled."""
        return [e for e in self.events if e.landed_at_s is None]

    @property
    def dead_lettered(self) -> int:
        """Poison writes: no amount of retrying or reconciling fixes these."""
        return sum(1 for e in self.events if e.poison)

    @property
    def reconcile_recoveries(self) -> int:
        """Writes the nightly sweep rescued after retries were exhausted."""
        return sum(1 for e in self.events if e.recovered_by_reconcile)

    @property
    def orphaned_vectors(self) -> int:
        """Deletes that never landed: the row is gone, the vector still serves."""
        return sum(1 for e in self.permanently_lost if e.kind == "delete")

    @property
    def missing_vectors(self) -> int:
        """Inserts that never landed: the product exists but is unsearchable."""
        return sum(1 for e in self.permanently_lost if e.kind == "insert")

    def stale_vector_seconds(self) -> float:
        """
        Total SKU-seconds during which the vector store disagreed with Postgres.

        For a landed event this is (landed_at - committed_at). For a lost event
        it is the remainder of the simulation horizon — it never converges.
        """
        total = 0.0
        for e in self.events:
            end = e.landed_at_s if e.landed_at_s is not None else self.horizon_s
            total += max(0.0, end - e.committed_at_s)
        return total

    def concurrently_stale(self, at_s: float) -> int:
        """How many SKUs are in a drifted state at a given instant."""
        n = 0
        for e in self.events:
            if e.committed_at_s > at_s:
                continue
            if e.landed_at_s is None or e.landed_at_s > at_s:
                n += 1
        return n


def _sample_lag(rng: random.Random) -> float:
    """CDC end-to-end lag: mostly fast, occasionally a long consumer-lag tail."""
    if rng.random() < CDC_TAIL_PROB:
        return rng.uniform(CDC_LAG_MEAN_S, CDC_LAG_TAIL_S)
    # Exponential-ish around the mean, floored at 1s.
    return max(1.0, rng.expovariate(1.0 / CDC_LAG_MEAN_S))


def simulate_disaggregated(
    hours: int,
    rng: random.Random,
    failure_rate: float = UPSERT_FAILURE_RATE,
    reconcile: bool = RECONCILE_ENABLED,
) -> SimResult:
    """Simulate architecture B: vectors in an external store, synced via CDC."""
    horizon_s = hours * 3600.0
    result = SimResult(horizon_s=horizon_s)

    per_hour = [
        ("update", UPDATES_PER_HOUR),
        ("delete", DELETES_PER_HOUR),
        ("insert", INSERTS_PER_HOUR),
    ]

    for hour in range(hours):
        hour_start = hour * 3600.0
        for kind, count in per_hour:
            for _ in range(count):
                committed = hour_start + rng.uniform(0, 3600.0)
                poison = rng.random() < POISON_RATE
                landed, attempts = _attempt_delivery(
                    committed, rng, failure_rate, poison
                )

                # The reconciliation job is the safety net you have to build.
                # It catches transiently-lost writes at the next nightly sweep.
                # It does NOT catch poison writes: a sweep re-reads the source
                # row and re-upserts it, hitting the same structural rejection.
                # Those need a human to look at the dead-letter queue.
                recovered = False
                if landed is None and reconcile and not poison:
                    sweeps = int(committed // (RECONCILE_EVERY_HOURS * 3600.0)) + 1
                    next_sweep = sweeps * RECONCILE_EVERY_HOURS * 3600.0
                    if next_sweep <= horizon_s:
                        landed = next_sweep
                        recovered = True

                result.events.append(
                    DriftEvent(kind, committed, landed, attempts, poison, recovered)
                )

    return result


def _attempt_delivery(
    committed_at_s: float,
    rng: random.Random,
    failure_rate: float,
    poison: bool = False,
) -> tuple[float | None, int]:
    """Deliver one write with retries. Returns (landed_at_s | None, attempts)."""
    if poison:
        # Structurally undeliverable: every retry produces the same rejection.
        return None, MAX_RETRIES + 1
    t = committed_at_s
    for attempt in range(1, MAX_RETRIES + 2):  # initial try + MAX_RETRIES
        t += _sample_lag(rng)
        if rng.random() >= failure_rate:
            return t, attempt
        if attempt <= MAX_RETRIES:
            t += RETRY_BACKOFF_S[min(attempt - 1, len(RETRY_BACKOFF_S) - 1)]
    return None, MAX_RETRIES + 1


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------


def _fmt_int(n: float) -> str:
    return f"{n:,.0f}"


def report(res: SimResult, hours: int, reconcile: bool) -> None:
    stale_s = res.stale_vector_seconds()
    stale_min = stale_s / 60.0

    # Average number of SKUs drifted at any instant, sampled hourly.
    samples = [res.concurrently_stale(h * 3600.0) for h in range(1, hours + 1)]
    avg_concurrent = sum(samples) / len(samples)
    peak_concurrent = max(samples)

    # Customer impact: probability a given search surfaces a drifted SKU.
    # P(a result is drifted) ≈ avg_concurrent / CATALOG_SIZE.
    p_result_drifted = avg_concurrent / CATALOG_SIZE
    p_search_affected = 1 - (1 - p_result_drifted) ** RESULTS_PER_SEARCH
    affected_searches = p_search_affected * SEARCHES_PER_HOUR * hours

    lost = res.permanently_lost

    print("=" * 74)
    print("  DUAL-WRITE DRIFT SIMULATION")
    print("=" * 74)
    print(f"  Catalog size              : {_fmt_int(CATALOG_SIZE)} SKUs")
    print(f"  Simulated horizon         : {hours} h")
    print(f"  Reconciliation job        : {'ON (nightly)' if reconcile else 'OFF'}")
    print(f"  Sync writes attempted     : {_fmt_int(res.total_events)}")
    print()

    print("-" * 74)
    print("  ARCHITECTURE A — pgvector (embedding co-located with the row)")
    print("-" * 74)
    print("  Stale vectors             : 0        (same transaction)")
    print("  Orphaned vectors          : 0        (same transaction)")
    print("  Missing vectors           : 0        (same transaction)")
    print("  Searches returning stale  : 0")
    print("  Sync pipeline to build    : none")
    print()

    print("-" * 74)
    print("  ARCHITECTURE B — external vector DB (outbox -> CDC -> upsert)")
    print("-" * 74)
    print(f"  Total drift               : {_fmt_int(stale_min)} SKU-minutes")
    print(f"  Avg SKUs drifted (instant): {_fmt_int(avg_concurrent)}")
    print(f"  Peak SKUs drifted         : {_fmt_int(peak_concurrent)}")
    print(f"  Permanently lost writes   : {_fmt_int(len(lost))}"
          f"  ({len(lost) / max(1, res.total_events):.3%} of writes)")
    print(f"    - orphaned vectors      : {_fmt_int(res.orphaned_vectors)}"
          "   (row deleted, vector still searchable)")
    print(f"    - missing vectors       : {_fmt_int(res.missing_vectors)}"
          "   (product live but unsearchable)")
    print(f"  Dead-letter queue depth   : {_fmt_int(res.dead_lettered)}"
          "   (poison writes — reconciliation cannot fix these)")
    print(f"  Rescued by reconciliation : {_fmt_int(res.reconcile_recoveries)}"
          f"   {'(sweep ON)' if reconcile else '(sweep OFF)'}")
    print()
    if reconcile and res.reconcile_recoveries == 0 and res.dead_lettered:
        print("  ^ Note: the nightly sweep rescued nothing, and that is the lesson.")
        print("    Retrying an independent transient failure 4x makes permanent")
        print("    loss vanishingly rare on its own — so the sweep has almost")
        print("    nothing left to catch. Every write in the DLQ above is poison,")
        print("    which re-reading the source row and re-upserting reproduces")
        print("    exactly. Reconciliation buys far less than teams assume; the")
        print("    DLQ still needs a human. Re-run with --failure-rate 0.4 to see")
        print("    the regime where the sweep does earn its keep.")
        print()
    print(f"  P(search hits drifted SKU): {p_search_affected:.4%}")
    print(f"  Affected searches / {hours}h  : {_fmt_int(affected_searches)}")
    print()

    print("-" * 74)
    print("  WHAT ARCHITECTURE B OBLIGES YOU TO BUILD AND STAFF")
    print("-" * 74)
    for item in (
        "transactional outbox table + writer",
        "CDC consumer (Debezium/Kafka) with idempotent upserts",
        "retry policy + dead-letter queue for exhausted writes",
        "nightly reconciliation sweep over the full catalog",
        "drift dashboard + alerting (the metrics printed above)",
        "runbook: 'Postgres and the vector store disagree'",
    ):
        print(f"    - {item}")
    print()

    print("=" * 74)
    print("  THE ARCHITECT'S POINT")
    print("=" * 74)
    print("  None of the six line items above appear on any vendor pricing page.")
    print("  They are the real cost of disaggregating vector state from the")
    print("  source of truth. A managed vector DB manages *their* index; the")
    print("  freshness contract with your catalog stays yours either way.")
    print()
    print("  Counter-argument to hold honestly: at a scale where the index no")
    print("  longer fits one node, or with hundreds of skewed tenants, you pay")
    print("  this cost anyway — because pgvector stops being an option. The")
    print("  decision is *when* the trigger fires, not whether drift is bad.")
    print("=" * 74)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--failure-rate", type=float, default=UPSERT_FAILURE_RATE,
                   help="transient upsert failure rate at the vector DB API")
    p.add_argument("--no-reconcile", action="store_true",
                   help="simulate without a nightly reconciliation sweep")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = random.Random(args.seed)
    reconcile = not args.no_reconcile
    res = simulate_disaggregated(
        args.hours, rng, failure_rate=args.failure_rate, reconcile=reconcile
    )
    report(res, args.hours, reconcile)


if __name__ == "__main__":
    main()
