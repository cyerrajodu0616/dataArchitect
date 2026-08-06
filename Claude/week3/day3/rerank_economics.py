"""
Week 3, Day 3 — Reranking: The Ceiling, The Bill, and Where They Cross
======================================================================

A reranker is a second-stage model that re-scores the candidates the first
stage returned. It improves PRECISION. It cannot improve RECALL, ever, because
it only sees what it was handed.

That single fact settles most reranking arguments, and it is what this script
makes concrete:

    the best possible end-to-end quality is bounded by recall@N of stage one

So the design question is not "should we rerank" but "how large can N be
before latency or cost stops us, and is recall@N at that point good enough to
be worth reranking at all?" Two exact quantities and one measured one:

  * CEILING  — recall@N. Exact, given your measured recall curve. If
               recall@100 is 0.72, then 28% of queries CANNOT be answered no
               matter how good the reranker is. Fix retrieval first.

  * COST     — N forward passes per query for a cross-encoder, or N chunks of
               prompt for an LLM reranker. Exact, given throughput and price.
               Linear in N, which is why this is a budget conversation.

  * GAIN     — how much of the available headroom the reranker actually
               captures. NOT exact. Supplied as a parameter, because the only
               honest source is your own golden set (Day 4). The default here
               is illustrative and labelled as such.

The output is the intersection: the largest N your latency budget allows, what
that costs, and what the ceiling is there.

stdlib only.

Usage:
    python rerank_economics.py
    python rerank_economics.py --domain retail
    python rerank_economics.py --domain insurance --reranker llm
    python rerank_economics.py --recall-10 0.55 --recall-100 0.72
"""

from __future__ import annotations

import argparse
import math

# ----------------------------------------------------------------------------
# Throughput. Measure your own — these are Q1-2026 ballparks for planning.
# A "pair" is one (query, candidate) forward pass.
# ----------------------------------------------------------------------------

RERANKERS = {
    # name            pairs/sec   $/hour   note
    "minilm-gpu":  dict(rate=2000.0, hourly=0.80,
                        note="6-layer MiniLM cross-encoder on an L4-class GPU. "
                             "The default choice — cheap, fast, and most of the "
                             "quality of a bigger model."),
    "base-gpu":    dict(rate=500.0, hourly=0.80,
                        note="12-layer base-size cross-encoder, 512 tokens. "
                             "Better ordering, 4x the compute."),
    "minilm-cpu":  dict(rate=120.0, hourly=0.10,
                        note="Same model on CPU. Viable only at low QPS — but "
                             "at low QPS it is genuinely the right answer, and "
                             "it removes a GPU from your architecture."),
    "llm":         dict(rate=None, hourly=None,
                        note="Listwise LLM reranking: one call, all candidates "
                             "in the prompt. Best quality, priced per token, "
                             "and latency scales with prompt length."),
}

LLM_IN_USD_PER_MTOK = 0.25
LLM_TPS = 4000.0            # prompt-processing tokens/sec, affects latency
SECONDS_PER_MONTH = 2_592_000
PEAK_TO_AVERAGE_RATIO = 0.25

CANDIDATE_SWEEP = [10, 25, 50, 100, 200, 500, 1000]


DOMAIN_PRESETS = {
    "retail": dict(
        qps=500.0, p99_budget_ms=150.0, first_stage_ms=25.0, chunk_tokens=200,
        recall_10=0.71, recall_100=0.93, top_k=8,
        note="500 QPS is the binding constraint, not quality. A cross-encoder at "
             "N=100 needs 25 GPUs to keep up. This is the domain where reranking "
             "is an economics problem before it is a relevance problem — and "
             "where routing (rerank only the ambiguous classes) pays for itself."),
    "insurance": dict(
        qps=20.0, p99_budget_ms=2000.0, first_stage_ms=60.0, chunk_tokens=512,
        recall_10=0.62, recall_100=0.89, top_k=6,
        note="The gap between recall@10 (0.62) and recall@100 (0.89) is 27 points "
             "of headroom sitting unused — exactly the situation reranking exists "
             "for, and a cross-encoder captures it for $584/mo. Note what the LLM "
             "reranker run shows though: at 512-token chunks even N=25 blows the "
             "2-second budget. A generous latency budget does NOT imply an LLM "
             "reranker fits; prompt length, not QPS, is the binding constraint "
             "there. Run --reranker llm and read the fits? column."),
    "fintech-docs": dict(
        qps=30.0, p99_budget_ms=1500.0, first_stage_ms=40.0, chunk_tokens=400,
        recall_10=0.68, recall_100=0.91, top_k=6,
        note="Comfortable middle. Cross-encoder at N=100 on a single GPU with "
             "room to spare; the decision is which model, not whether."),
}


def recall_at(n: int, r10: float, r100: float) -> float:
    """
    Interpolate a recall@N curve through two measured points.

    Recall rises with N and saturates. Fitting r(N) = 1 - a*exp(-N/tau) through
    (10, r10) and (100, r100) gives a curve with the right shape. This is a
    PLANNING model — replace it with your measured curve as soon as you have
    one, because the shape near your operating point is what decides N.
    """
    if r100 <= r10:
        return r10
    # Solve 1 - a*e^(-10/tau) = r10 and 1 - a*e^(-100/tau) = r100.
    ratio = (1 - r10) / (1 - r100)
    tau = 90.0 / math.log(ratio)
    a = (1 - r10) * math.exp(10.0 / tau)
    return min(0.999, 1 - a * math.exp(-n / tau))


def rerank_latency_ms(n: int, kind: str, chunk_tokens: int) -> float:
    if kind == "llm":
        return n * chunk_tokens / LLM_TPS * 1000.0
    return n / RERANKERS[kind]["rate"] * 1000.0


def rerank_cost_month(n: int, kind: str, qps: float, chunk_tokens: int) -> float:
    avg_qps = qps * PEAK_TO_AVERAGE_RATIO
    if kind == "llm":
        queries = avg_qps * SECONDS_PER_MONTH
        return queries * n * chunk_tokens / 1e6 * LLM_IN_USD_PER_MTOK
    # Size the fleet on PEAK, bill it for the whole month — you cannot spin a
    # GPU up per request.
    rate = RERANKERS[kind]["rate"]
    machines = math.ceil(qps * n / rate)
    return machines * RERANKERS[kind]["hourly"] * 730.0


def report(args: argparse.Namespace, note: str | None) -> None:
    kind = args.reranker
    print("=" * 94)
    print("  RERANKING — CEILING, BILL, AND WHERE THEY CROSS")
    print("=" * 94)
    if args.domain:
        print(f"  Domain          : {args.domain.upper()}")
    print(f"  Reranker        : {kind}")
    print(f"  Traffic         : {args.qps:,.0f} QPS peak"
          f"  ({args.qps * PEAK_TO_AVERAGE_RATIO:,.0f} avg)")
    print(f"  Latency budget  : {args.p99_budget_ms:,.0f} ms P99"
          f"  (first stage already spends {args.first_stage_ms:,.0f} ms)")
    print(f"  Measured recall : recall@10 = {args.recall_10:.2f},"
          f"  recall@100 = {args.recall_100:.2f}")
    print(f"  Final k         : {args.top_k}")
    print()
    print(f"  {RERANKERS[kind]['note']}")
    print()

    print("-" * 94)
    print(f"  {'N':>6}{'recall@N':>11}{'headroom':>11}{'rerank ms':>12}"
          f"{'total ms':>11}{'fits?':>8}{'$/month':>12}{'$ per +0.01':>14}")
    print("  " + "-" * 90)

    affordable = []
    for n in CANDIDATE_SWEEP:
        r = recall_at(n, args.recall_10, args.recall_100)
        ms = rerank_latency_ms(n, kind, args.chunk_tokens)
        total = ms + args.first_stage_ms
        fits = total <= args.p99_budget_ms
        cost = rerank_cost_month(n, kind, args.qps, args.chunk_tokens)
        # Headroom = how much quality is theoretically available by reranking
        # this candidate set instead of just taking the first-stage top-k.
        base = recall_at(args.top_k, args.recall_10, args.recall_100)
        headroom = r - base
        flag = "yes" if fits else "NO"
        if fits:
            affordable.append((n, r, total, cost, headroom))
        # What each 0.01 of ceiling actually costs at this N.
        unit = cost / (headroom * 100) if headroom > 0.001 else float("inf")
        unit_s = f"{unit:,.0f}" if unit != float("inf") else "-"
        print(f"  {n:>6}{r:>11.3f}{headroom:>+11.3f}{ms:>12.1f}{total:>11.1f}"
              f"{flag:>8}{cost:>12,.0f}{unit_s:>14}")
    print()

    base = recall_at(args.top_k, args.recall_10, args.recall_100)
    print("-" * 94)
    print("  READING IT")
    print("-" * 94)
    print(f"  CEILING is recall@N. No reranker can exceed it, because it only ever")
    print(f"  sees the candidates stage one handed it. At N=100 the ceiling is")
    print(f"  {recall_at(100, args.recall_10, args.recall_100):.3f} — so"
          f" {(1 - recall_at(100, args.recall_10, args.recall_100)) * 100:.0f}% of"
          f" queries are unanswerable regardless of reranker quality.")
    print()
    print(f"  HEADROOM is recall@N minus recall@{args.top_k}, i.e. what reranking")
    print(f"  could recover versus just taking the first-stage top-{args.top_k}")
    print(f"  ({base:.3f}). That difference is the entire prize. If it is small,")
    print(f"  a reranker is a latency tax with no upside.")
    print()

    if not affordable:
        print("=" * 94)
        print("  NOTHING FITS")
        print("=" * 94)
        print(f"  Even N=10 costs {rerank_latency_ms(10, kind, args.chunk_tokens):.0f}"
              f" ms against a {args.p99_budget_ms:.0f} ms budget with"
              f" {args.first_stage_ms:.0f} ms already spent.")
        print("  Options: a smaller reranker, rerank only some query classes, or")
        print("  accept first-stage ordering and spend the effort on retrieval.")
    else:
        max_n, max_r, max_total, max_cost, max_head = affordable[-1]

        # Pick on MARGINAL EFFICIENCY, not on what fits. The largest N the
        # latency budget allows is almost never the one worth buying, because
        # cost is linear in N while the recall curve saturates.
        def unit_cost(row):
            _, _, _, c, h = row
            return c / (h * 100) if h > 0.001 else float("inf")

        best = min(affordable, key=unit_cost)
        n, r, total, cost, headroom = best

        print("=" * 94)
        print("  RECOMMENDATION")
        print("=" * 94)
        print(f"  Candidates N         : {n}")
        print(f"  Ceiling there        : recall@{n} = {r:.3f}")
        print(f"  Headroom to capture  : {headroom:+.3f} over first-stage top-"
              f"{args.top_k}")
        print(f"  Latency              : {total:.0f} ms of a {args.p99_budget_ms:.0f}"
              f" ms budget")
        print(f"  Cost                 : ${cost:,.0f}/month"
              f"  (${unit_cost(best):,.0f} per +0.01 of ceiling)")
        print()
        print("  Chosen on COST PER UNIT OF CEILING, not on what fits the latency")
        print("  budget. Cost is linear in N; the recall curve saturates. Those two")
        print("  facts guarantee an efficiency optimum strictly below the limit.")
        print()

        if max_n != n:
            extra_q = max_r - r
            extra_c = max_cost - cost
            print(f"  The largest N that fits is {max_n}: recall@{max_n} = {max_r:.3f}"
                  f" at ${max_cost:,.0f}/mo.")
            print(f"  That is ${extra_c:,.0f}/month more for {extra_q:+.3f} of"
                  f" ceiling — ${extra_c / max(extra_q * 100, 1e-9):,.0f}")
            print(f"  per +0.01, versus ${unit_cost(best):,.0f} at N={n}.")
            print()
            print(f"  'We set N to whatever fit the latency budget' is a real and")
            print(f"  common decision procedure. It is how you end up paying"
                  f" {extra_c / max(cost, 1):.1f}x")
            print(f"  for the last few points of a curve that had already flattened.")
            print()

        # Where the curve itself flattens, independent of price.
        knee = None
        for cand in CANDIDATE_SWEEP:
            gain = recall_at(cand * 2, args.recall_10, args.recall_100) - \
                recall_at(cand, args.recall_10, args.recall_100)
            if gain < 0.01:
                knee = cand
                break
        if knee:
            print(f"  For reference, the recall curve itself flattens at N={knee}"
                  f" — doubling")
            print(f"  from there adds under 0.01. Past that point you are buying"
                  f" nothing at")
            print(f"  any price.")
        print()

        if headroom < 0.05:
            print("  WARNING: headroom is under 0.05. Reranking cannot buy you much")
            print("  here because the first stage already puts the answer in the top")
            print(f"  {args.top_k}. Spend the effort on the queries where it does not.")
        if args.recall_100 < 0.90:
            print(f"  WARNING: recall@100 is only {args.recall_100:.2f}, so even a")
            print(f"  perfect reranker leaves {(1 - args.recall_100) * 100:.0f}% of"
                  f" queries unanswerable. Fix")
            print("  first-stage recall first — chunking, hybrid weighting, query")
            print("  expansion. Reranking a candidate set that does not contain the")
            print("  answer only reorders wrong answers.")

    if note:
        print()
        print("  DOMAIN NOTE")
        line = ""
        for w in note.split():
            if len(line) + len(w) + 1 > 86:
                print(f"    {line}")
                line = w
            else:
                line = f"{line} {w}".strip()
        print(f"    {line}")

    print()
    print("=" * 94)
    print("  THE ORDER TO DO THINGS IN")
    print("=" * 94)
    for line in [
        "1. Measure recall@N of stage one. This is the ceiling and it costs nothing",
        "   to compute — you already have the golden set from Week 2 Day 5.",
        "2. If recall@100 is below ~0.90, STOP. Fix retrieval. A reranker applied",
        "   to a candidate set that does not contain the answer changes the order",
        "   of wrong answers.",
        "3. If recall@10 is already close to recall@100, STOP. There is no headroom",
        "   and a reranker is pure latency.",
        "4. Only if there is a real gap between recall@k and recall@N is reranking",
        "   the right tool — and then N is set by your latency budget, not by a",
        "   number from a paper.",
    ]:
        print(f"  {line}")
    print("=" * 94)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--reranker", choices=sorted(RERANKERS), default="minilm-gpu")
    p.add_argument("--qps", type=float)
    p.add_argument("--p99-budget-ms", type=float)
    p.add_argument("--first-stage-ms", type=float)
    p.add_argument("--chunk-tokens", type=int)
    p.add_argument("--recall-10", type=float, help="measured recall@10 of stage one")
    p.add_argument("--recall-100", type=float, help="measured recall@100 of stage one")
    p.add_argument("--top-k", type=int)
    args = p.parse_args()

    base = dict(DOMAIN_PRESETS["retail"])
    if args.domain:
        base = dict(DOMAIN_PRESETS[args.domain])
    note = base.pop("note", None)
    for k, v in base.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    report(args, note if args.domain else None)


if __name__ == "__main__":
    main()
