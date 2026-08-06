"""
Week 2, Day 3 — The Sizing Ladder
=================================

Eight questions, asked in order, that determine a RAG retrieval architecture.
"Which vector database should I use?" is not one of them — it falls out of the
bottom.

    1. How many documents?
    2. x chunks per document                -> N   (a DECISION, not a fact)
    3. Embedding dimension?                 ->     (also a decision — MRL)
    4. Inserts vs updates per day?
    5. How selective are the filters?       -> can delete the problem entirely
    6. Expected QPS (peak)?
    7. Latency SLA at P99?
    8. What does a wrong answer cost?       -> recall SLO -> encoding -> cost

Two things this script does that a static checklist cannot:

  * It checks the FILTERED EXACT SEARCH path first. Latency and throughput fail
    separately — a filtered scan can meet a P99 SLA and still be unaffordable
    because each query burns whole core-seconds. Both are checked.

  * It runs a SENSITIVITY PASS: each answer is perturbed in turn and the
    architecture recomputed. The output names the question that actually
    decided the outcome — which is the one worth arguing about in a design
    review, and usually not the one people spend the meeting on.

Cost constants are kept consistent with quantization_cost_calc.py.
stdlib only.

Usage:
    python sizing_ladder.py --domain retail
    python sizing_ladder.py --domain insurance
    python sizing_ladder.py --domain fintech-txn
    python sizing_ladder.py --docs 800000 --chunks-per-doc 5 --dims 1536 \
        --qps 500 --p99-ms 150 --filter-selectivity 0.02 --miss-cost low
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Constants. Kept aligned with quantization_cost_calc.py so the two scripts
# cannot disagree with each other in an interview.
# ----------------------------------------------------------------------------

MANAGED_PG_RAM_GB_MONTH = 11.0
EC2_RAM_GB_MONTH = 5.5
PG_RAM_HEADROOM = 1.8
ANN_RAM_HEADROOM = 1.3
HNSW_GRAPH_BYTES_EXT = 136    # pgvector stores a full vector copy instead

ENGINEER_ANNUAL = 180_000
OPS_FTE_EXTERNAL = 0.20
OPS_FTE_PGVECTOR = 0.05

# Exact-scan throughput. A brute-force scan is memory-bandwidth bound, not
# compute bound, once the distance kernel is SIMD. This is the effective rate
# one core sustains streaming vectors out of RAM.
SCAN_GB_PER_SEC_PER_CORE = 6.0
MAX_PARALLEL_WORKERS = 4      # Postgres parallel seq scan, realistic setting

# Cores you are willing to hand to retrieval before it stops being sane.
CORE_BUDGET = 16

# The largest single instance you can actually rent. Past this, "buy a bigger
# box" stops being an available move and sharding is structural, not optional.
MAX_INSTANCE_RAM_GB = 768

# Miss cost -> recall SLO. This is the whole point of question 8.
RECALL_SLO = {"low": 0.90, "medium": 0.95, "high": 0.99}

# The in-Postgres mitigation ladder from section 7, as achievable factors.
LADDER_FACTOR = 4.9           # halfvec + MRL truncation, measured not assumed


@dataclass
class Encoding:
    name: str
    bytes_per_vector: float
    recall: float
    note: str


def encodings_for(dims: int) -> list[Encoding]:
    """Cheapest-first. Recall figures are post-rerank where rerank is required."""
    return [
        Encoding("binary + exact rerank", dims / 8, 0.92,
                 "needs full vectors on disk for the rerank stage"),
        Encoding("PQ-64", 64, 0.94, "needs full vectors on disk for rerank"),
        Encoding("PQ-96", 96, 0.96, "needs full vectors on disk for rerank"),
        Encoding("int8 scalar quantization", dims * 1.0, 0.97,
                 "4x smaller, no rerank stage needed — best quality per byte"),
        Encoding("float16 (halfvec)", dims * 2.0, 0.98, "free 2x"),
        Encoding("float32", dims * 4.0, 0.98, "baseline"),
    ]


# ----------------------------------------------------------------------------
# The ladder
# ----------------------------------------------------------------------------


@dataclass
class Answers:
    docs: int
    chunks_per_doc: float
    dims: int
    inserts_per_day: float        # fraction of corpus
    updates_per_day: float        # fraction of corpus
    filter_selectivity: float     # fraction surviving the WHERE clause
    qps: float                    # peak
    p99_ms: float
    miss_cost: str                # low | medium | high

    @property
    def n(self) -> int:
        return int(self.docs * self.chunks_per_doc)


@dataclass
class Verdict:
    architecture: str
    index: str
    encoding: str
    reason: str
    monthly_usd: float
    detail: dict


def exact_scan_ms(vectors: float, dims: int, workers: int) -> float:
    gb = vectors * dims * 4 / 1e9
    return gb / (SCAN_GB_PER_SEC_PER_CORE * workers) * 1000.0


def exact_core_seconds(vectors: float, dims: int) -> float:
    """Work per query in core-seconds — independent of how you parallelise it."""
    gb = vectors * dims * 4 / 1e9
    return gb / SCAN_GB_PER_SEC_PER_CORE


def decide(a: Answers) -> Verdict:
    n = a.n
    slo = RECALL_SLO[a.miss_cost]
    heap_gb = n * (a.dims * 4 + 40) / 1e9

    # --- Q5 first: does filtering make the ANN index unnecessary? -----------
    candidates = n * a.filter_selectivity
    filt_ms = exact_scan_ms(candidates, a.dims, MAX_PARALLEL_WORKERS)
    filt_cores = exact_core_seconds(candidates, a.dims) * a.qps

    if filt_ms <= a.p99_ms and filt_cores <= CORE_BUDGET:
        pg_ram = max(8.0, heap_gb * 0.15 * PG_RAM_HEADROOM)  # hot slice only
        cost = pg_ram * MANAGED_PG_RAM_GB_MONTH + heap_gb * 0.10
        cost += ENGINEER_ANNUAL / 12 * OPS_FTE_PGVECTOR
        if a.filter_selectivity < 0.01:
            which = "partition by the filter key, then exact search inside it"
        elif a.filter_selectivity >= 0.95:
            which = "exhaustive exact search, no index"
        else:
            which = "exact search behind a b-tree filter"
        return Verdict(
            architecture=f"Postgres only — {which}",
            index="b-tree / partition pruning on the filter column. NO ANN index.",
            encoding="float32 (recall 1.0 — this is exact search)",
            reason=(f"filtering leaves {candidates:,.0f} candidates; scanning them "
                    f"exactly costs {filt_ms:.1f} ms P99 and {filt_cores:.2f} cores "
                    f"at {a.qps:,.0f} QPS"),
            monthly_usd=cost,
            detail=dict(path="exact", kind="exact", candidates=candidates, filt_ms=filt_ms,
                        filt_cores=filt_cores),
        )

    # --- Unfiltered exact? (small corpora) ----------------------------------
    full_ms = exact_scan_ms(n, a.dims, MAX_PARALLEL_WORKERS)
    full_cores = exact_core_seconds(n, a.dims) * a.qps
    if full_ms <= a.p99_ms and full_cores <= CORE_BUDGET:
        pg_ram = max(8.0, heap_gb * PG_RAM_HEADROOM)
        cost = pg_ram * MANAGED_PG_RAM_GB_MONTH + ENGINEER_ANNUAL / 12 * OPS_FTE_PGVECTOR
        return Verdict(
            architecture="Postgres only — exhaustive exact search, no index",
            index="none",
            encoding="float32 (recall 1.0)",
            reason=(f"the whole corpus is {heap_gb:.2f} GB; scanning all of it is "
                    f"{full_ms:.1f} ms P99 and {full_cores:.2f} cores at "
                    f"{a.qps:,.0f} QPS — an index would buy speed you do not need "
                    f"and cost recall you cannot audit"),
            monthly_usd=cost,
            detail=dict(path="exact-full", kind="exact", full_ms=full_ms,
                        full_cores=full_cores),
        )

    # --- ANN required. Pick the cheapest encoding meeting the recall SLO. ---
    all_encodings = encodings_for(a.dims)
    choices = [e for e in all_encodings if e.recall >= slo]
    enc = choices[0] if choices else all_encodings[-1]
    slo_unreachable = not choices

    # pgvector path: HNSW stores a full copy of the vector, so index ~= heap.
    pg_index_gb = heap_gb
    pg_ram_gb = (heap_gb + pg_index_gb) * PG_RAM_HEADROOM
    pg_cost = pg_ram_gb * MANAGED_PG_RAM_GB_MONTH
    pg_cost += ENGINEER_ANNUAL / 12 * OPS_FTE_PGVECTOR

    pg_ladder_ram = pg_ram_gb / LADDER_FACTOR
    pg_ladder_cost = pg_ladder_ram * MANAGED_PG_RAM_GB_MONTH
    pg_ladder_cost += ENGINEER_ANNUAL / 12 * OPS_FTE_PGVECTOR

    # External path: quantized index in RAM, full vectors on disk for rerank.
    ext_ram_gb = n * (enc.bytes_per_vector + HNSW_GRAPH_BYTES_EXT) / 1e9 * ANN_RAM_HEADROOM
    ext_cost = ext_ram_gb * EC2_RAM_GB_MONTH + heap_gb * 0.10
    ext_cost += ENGINEER_ANNUAL / 12 * OPS_FTE_EXTERNAL
    ext_cost += max(8.0, n * 48 / 1e9 * PG_RAM_HEADROOM) * MANAGED_PG_RAM_GB_MONTH

    # CPU wall check — hit before the RAM wall on a shared OLTP instance.
    # Above ~1 core of steady retrieval work you are competing with OLTP for the
    # same scheduler, which is a checkout-latency problem, not a search problem.
    ann_cores = a.qps * 0.003  # ~3 ms of CPU per HNSW probe at these dims

    # Feasibility before cost. A cost comparison alone will happily recommend a
    # 3 TB Postgres instance, which is not a thing you can buy.
    pg_deployable = pg_ladder_ram <= MAX_INSTANCE_RAM_GB

    if pg_deployable and pg_ladder_cost <= ext_cost:
        cheaper, cost = "pgvector", pg_ladder_cost
        arch, kind = "pgvector + HNSW", "pgvector"
        if ann_cores > 1.0:
            arch += ", on a DEDICATED READ REPLICA (CPU wall, not RAM wall)"
            kind = "pgvector-replica"
        idx = f"HNSW in Postgres, halfvec + MRL truncation (~{LADDER_FACTOR:.1f}x)"
        # The pgvector path uses the ladder, not the cheapest-possible encoding.
        enc_line = (f"halfvec + MRL-truncated dims (recall ~0.97 vs SLO {slo:.2f}) — "
                    f"{pg_ram_gb:.1f} GB of RAM becomes {pg_ladder_ram:.1f} GB")
        reason = (f"pgvector after the section-7 ladder is ${pg_ladder_cost:,.0f}/mo "
                  f"vs ${ext_cost:,.0f}/mo for an external {enc.name} index — the "
                  f"external option's {OPS_FTE_EXTERNAL:.0%} FTE of ops "
                  f"(${ENGINEER_ANNUAL / 12 * OPS_FTE_EXTERNAL:,.0f}/mo) is not "
                  f"repaid at this size")
    else:
        cheaper, cost = "external", ext_cost
        kind = "external"
        arch = f"Postgres for truth + external ANN service ({enc.name})"
        idx = (f"{enc.name} index in a FAISS/usearch process, "
               f"{ext_ram_gb:.1f} GB resident")
        enc_line = (f"{enc.name} (recall {enc.recall:.2f} vs SLO {slo:.2f}) — "
                    f"{enc.note}")
        if not pg_deployable:
            reason = (f"pgvector is not deployable here at all: even after the "
                      f"{LADDER_FACTOR:.1f}x ladder it needs {pg_ladder_ram:,.0f} GB "
                      f"of RAM against a {MAX_INSTANCE_RAM_GB} GB largest-instance "
                      f"ceiling. Cost is not the argument — availability is. "
                      f"External {enc.name} fits in {ext_ram_gb:,.1f} GB at "
                      f"${ext_cost:,.0f}/mo")
        else:
            reason = (f"external {enc.name} is ${ext_cost:,.0f}/mo vs "
                      f"${pg_ladder_cost:,.0f}/mo for pgvector even after the "
                      f"ladder — {(1 - ext_cost / pg_ladder_cost):.0%} cheaper, and "
                      f"the ladder cannot close a "
                      f"{pg_ram_gb / ext_ram_gb:.0f}x memory gap")

    if slo_unreachable:
        enc_line += (f"\n             WARNING: no ANN encoding reaches recall "
                     f"{slo:.2f}. A '{a.miss_cost}' miss cost is an argument for "
                     f"exact search, not for a better index.")

    return Verdict(
        architecture=arch, index=idx, encoding=enc_line,
        reason=reason, monthly_usd=cost,
        detail=dict(path=cheaper, kind=kind, pg_ram_gb=pg_ram_gb,
                    pg_ladder_ram=pg_ladder_ram,
                    ext_ram_gb=ext_ram_gb, ann_cores=ann_cores,
                    pg_cost=pg_ladder_cost, ext_cost=ext_cost,
                    filt_ms=filt_ms, filt_cores=filt_cores, candidates=candidates),
    )


# ----------------------------------------------------------------------------
# Sensitivity — which answer actually decided this?
# ----------------------------------------------------------------------------

# Each question is perturbed in every direction that could plausibly matter.
# Q5 in particular has to be tested BOTH ways: less selective filters can force
# an index into existence, and more selective ones can delete the need for it.
PERTURBATIONS = [
    ("Q1 documents", [
        ("2x the documents", lambda a: _rep(a, docs=int(a.docs * 2)))]),
    ("Q2 chunks per document", [
        ("2x the chunks (smaller chunks)",
         lambda a: _rep(a, chunks_per_doc=a.chunks_per_doc * 2))]),
    ("Q3 embedding dimension", [
        ("half the dims (MRL truncation)",
         lambda a: _rep(a, dims=max(64, a.dims // 2)))]),
    ("Q4 update rate", [
        ("5x the update rate",
         lambda a: _rep(a, updates_per_day=a.updates_per_day * 5))]),
    ("Q5 filter selectivity", [
        ("20x less selective",
         lambda a: _rep(a, filter_selectivity=min(1.0, a.filter_selectivity * 20))),
        ("20x more selective",
         lambda a: _rep(a, filter_selectivity=a.filter_selectivity / 20))]),
    ("Q6 peak QPS", [
        ("5x the peak QPS", lambda a: _rep(a, qps=a.qps * 5)),
        ("1/5 the peak QPS", lambda a: _rep(a, qps=a.qps / 5))]),
    ("Q7 P99 SLA", [
        ("4x tighter P99 SLA", lambda a: _rep(a, p99_ms=a.p99_ms / 4)),
        ("4x looser P99 SLA", lambda a: _rep(a, p99_ms=a.p99_ms * 4))]),
    ("Q8 cost of a miss", [
        ("one step higher miss cost", lambda a: _rep(a, miss_cost=_bump(a.miss_cost)))]),
]


def _rep(a: Answers, **kw) -> Answers:
    d = dict(vars(a))
    d.update(kw)
    return Answers(**d)


def _bump(m: str) -> str:
    order = ["low", "medium", "high"]
    return order[min(len(order) - 1, order.index(m) + 1)]


def _differs(w: Verdict, base: Verdict) -> bool:
    """Compare the structural verdict, not its wording.

    Two exact-search answers phrased differently are the SAME architecture; a
    flag here would be a false positive and those are what make a sensitivity
    table stop being believed.
    """
    return w.detail["kind"] != base.detail["kind"]


def sensitivity(a: Answers, base: Verdict) -> list[tuple[str, str, bool, str, float]]:
    """For each question, report the variant that moved the answer furthest."""
    out = []
    for name, variants in PERTURBATIONS:
        best = None
        for label, fn in variants:
            w = decide(fn(a))
            changed = _differs(w, base)
            delta = w.monthly_usd - base.monthly_usd
            cand = (name, label, changed, w.architecture, delta)
            # An architecture change always beats a price change.
            if best is None or (changed and not best[2]) or \
                    (changed == best[2] and abs(delta) > abs(best[4])):
                best = cand
        out.append(best)
    return out


# ----------------------------------------------------------------------------
# Presets — the three domains carried through Week 2
# ----------------------------------------------------------------------------

DOMAIN_PRESETS = {
    "retail": dict(
        docs=800_000, chunks_per_doc=5, dims=1536,
        inserts_per_day=0.005, updates_per_day=0.025,
        filter_selectivity=0.02, qps=500, p99_ms=150, miss_cost="low",
        note="Category + brand filters cut the corpus to ~2%, which is fast "
             "enough per query but not at 500 QPS — this is the case where "
             "filtering does NOT save you and you genuinely need the index."),
    "insurance": dict(
        docs=12_000, chunks_per_doc=17, dims=1536,
        inserts_per_day=0.001, updates_per_day=0.0,
        filter_selectivity=0.05, qps=20, p99_ms=2000, miss_cost="high",
        note="A high miss cost and a generous SLA point the same direction: "
             "exact search. Recall 1.0, no index to rebuild, no graph decay, "
             "and a result you can reproduce in an audit three years later."),
    "fintech-txn": dict(
        docs=2_000_000_000, chunks_per_doc=1, dims=768,
        inserts_per_day=0.02, updates_per_day=0.0,
        filter_selectivity=0.00000025, qps=4000, p99_ms=50, miss_cost="high",
        note="The 2-billion-vector ANN problem disappears once you ask what a "
             "single query searches: one card's history, ~500 rows. Partition "
             "by business key and the hard problem was never there."),
    "fintech-docs": dict(
        docs=15_000, chunks_per_doc=10, dims=1536,
        inserts_per_day=0.002, updates_per_day=0.001,
        filter_selectivity=0.1, qps=30, p99_ms=1500, miss_cost="high",
        note="Policy and regulatory documents. Same shape as insurance — small "
             "corpus, high miss cost, human in the loop."),
}


# ----------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------

def wrap(text: str, indent: str = "    ", width: int = 76) -> None:
    line = ""
    for w in text.split():
        if len(line) + len(w) + 1 > width:
            print(f"{indent}{line}")
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        print(f"{indent}{line}")


def report(a: Answers, note: str | None) -> None:
    n = a.n
    heap_gb = n * (a.dims * 4 + 40) / 1e9
    v = decide(a)

    print("=" * 80)
    print("  THE SIZING LADDER")
    print("=" * 80)
    print(f"  Q1  documents ................ {a.docs:,}")
    print(f"  Q2  chunks per document ...... {a.chunks_per_doc:g}"
          f"   -> N = {n:,} vectors   [a DECISION]")
    print(f"  Q3  embedding dimension ...... {a.dims:,}"
          f"   -> {a.dims * 4:,} B/vector, heap {heap_gb:,.2f} GB   [a DECISION]")
    print(f"  Q4  inserts / updates per day  {a.inserts_per_day:.2%} in,"
          f" {a.updates_per_day:.2%} up")
    print(f"  Q5  filter selectivity ....... {a.filter_selectivity:.6g}"
          f"   -> {n * a.filter_selectivity:,.0f} candidates/query")
    print(f"  Q6  peak QPS ................. {a.qps:,.0f}")
    print(f"  Q7  P99 latency SLA .......... {a.p99_ms:,.0f} ms")
    print(f"  Q8  cost of a wrong answer ... {a.miss_cost.upper()}"
          f"   -> recall SLO {RECALL_SLO[a.miss_cost]:.2f}")
    print()

    print("-" * 80)
    print("  STEP 5 CHECK — CAN FILTERING DELETE THE PROBLEM?")
    print("-" * 80)
    cand = n * a.filter_selectivity
    ms = exact_scan_ms(cand, a.dims, MAX_PARALLEL_WORKERS)
    cores = exact_core_seconds(cand, a.dims) * a.qps
    lat_ok = "PASS" if ms <= a.p99_ms else "FAIL"
    thr_ok = "PASS" if cores <= CORE_BUDGET else "FAIL"
    print(f"  Candidates after filter : {cand:,.0f} vectors"
          f" ({cand * a.dims * 4 / 1e6:,.1f} MB)")
    print(f"  Exact scan latency      : {ms:,.2f} ms P99"
          f"  vs SLA {a.p99_ms:,.0f} ms   [{lat_ok}]")
    print(f"  Exact scan throughput   : {cores:,.2f} cores at {a.qps:,.0f} QPS"
          f"  vs budget {CORE_BUDGET}   [{thr_ok}]")
    if lat_ok == "PASS" and thr_ok == "FAIL":
        print()
        wrap("Note the asymmetry: latency passes and throughput fails. Each query "
             "is individually fast but burns core-seconds you cannot afford to "
             "spend hundreds of times a second. Checking only P99 here is the "
             "classic way to talk yourself into an architecture that collapses "
             "under load.", "  ")
    print()

    print("=" * 80)
    print("  ARCHITECTURE")
    print("=" * 80)
    print(f"  {v.architecture}")
    print()
    print(f"  Index    : {v.index}")
    print(f"  Encoding : {v.encoding}")
    print(f"  Cost     : ${v.monthly_usd:,.0f}/month all-in")
    print()
    print("  WHY")
    wrap(v.reason)
    print()

    if v.detail["path"] in ("exact", "exact-full"):
        wrap("No ANN index means: recall is 1.0 by construction, there is no "
             "recall-decay curve to monitor, no REINDEX window, no build "
             "pipeline, and a query re-run in three years returns exactly what "
             "it returned today. If you can afford exact search, every "
             "operational problem in this lesson stops existing.", "  ")
        print()

    print("-" * 80)
    print("  SENSITIVITY — WHICH ANSWER ACTUALLY DECIDED THIS?")
    print("-" * 80)
    print(f"  {'Question':<26}{'If...':<34}{'Verdict':>16}")
    print("  " + "-" * 76)
    deciders = []
    for name, label, changed, arch, delta in sensitivity(a, v):
        if abs(delta) < 0.5:
            delta = 0.0
        mark = "CHANGES" if changed else f"{delta:+,.0f}/mo"
        if changed:
            deciders.append((name, label, arch))
        print(f"  {name:<26}{label:<34}{mark:>16}")
    print()
    if deciders:
        print("  LOAD-BEARING ANSWERS:")
        for name, label, arch in deciders:
            wrap(f"{name} — with {label}, the architecture becomes: {arch}", "    ")
        print()
        wrap("These are the answers to verify with real data before building "
             "anything. The rest move the bill; these move the design.", "  ")
    else:
        print("  No single answer flips the architecture — the verdict is robust.")
        wrap("That is worth saying out loud in a design review. It means you can "
             "stop arguing about the inputs and start building, because being "
             "wrong about any one of them does not change what you build.", "  ")

    if note:
        print()
        print("  DOMAIN NOTE")
        wrap(note)

    print()
    print("=" * 80)
    print("  THE POINT")
    print("=" * 80)
    wrap("'Which vector database should I use?' is not on the ladder. It is the "
         "output. If it was your first question, you picked a technology and "
         "will now reverse-engineer requirements to justify it — and an "
         "interviewer can tell, because you will have no number to defend when "
         "they ask why.", "  ")
    print("=" * 80)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--docs", type=int)
    p.add_argument("--chunks-per-doc", type=float)
    p.add_argument("--dims", type=int)
    p.add_argument("--inserts-per-day", type=float)
    p.add_argument("--updates-per-day", type=float)
    p.add_argument("--filter-selectivity", type=float,
                   help="fraction of the corpus surviving a typical WHERE clause")
    p.add_argument("--qps", type=float, help="PEAK queries per second")
    p.add_argument("--p99-ms", type=float)
    p.add_argument("--miss-cost", choices=sorted(RECALL_SLO))
    args = p.parse_args()

    base = dict(DOMAIN_PRESETS["retail"])
    if args.domain:
        base = dict(DOMAIN_PRESETS[args.domain])
    note = base.pop("note", None)

    for k, v in base.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    a = Answers(docs=args.docs, chunks_per_doc=args.chunks_per_doc, dims=args.dims,
                inserts_per_day=args.inserts_per_day,
                updates_per_day=args.updates_per_day,
                filter_selectivity=args.filter_selectivity, qps=args.qps,
                p99_ms=args.p99_ms, miss_cost=args.miss_cost)
    report(a, note if args.domain else None)


if __name__ == "__main__":
    main()
