"""
Week 3, Day 1 — Chunk Size: The Joint Quality/Cost Sweep
========================================================

Week 2 Day 3 section 10 claimed chunks-per-document is a 2-4x infrastructure
lever. That is true and it is only half the story, because chunk size moves TWO
costs in OPPOSITE directions:

    smaller chunks -> more vectors      -> index cost UP
    smaller chunks -> fewer tokens/query -> generation cost DOWN

and for almost every real RAG system the second number is one to two orders of
magnitude larger than the first. This script prices both.

It also refuses to invent an nDCG. Instead it computes two things that follow
from geometry alone, so they are defensible under questioning:

  * P(answer span survives) — the probability that a contiguous answer of L
    tokens lands entirely inside one chunk. For chunk size C and overlap O with
    stride S = C - O, a span [p, p+L) sits inside chunk i iff
    iS <= p <= iS + C - L, so the good positions occupy min(S, C-L) of every S:

        P = 1                    if O >= L      <- overlap guarantees it
        P = (C - L) / (C - O)    if O <  L < C
        P = 0                    if L >= C

    That yields a sharp, checkable rule: **overlap at least as long as a typical
    answer span guarantees the span is never split.** Not "overlap preserves
    context" — an actual threshold.

  * Signal density L/C — the fraction of a retrieved chunk that is the answer.
    Everything else is tokens you pay for twice: once in the index, once in the
    prompt.

Small chunks are dense but fragment answers. Large chunks preserve answers but
dilute them. Overlap buys back containment at a storage multiplier of
C/(C-O). That is the whole tradeoff, and it is arithmetic.

Assumption stated plainly: answer spans are modelled as uniformly placed. Real
documents have structure, which is exactly the argument for structure-aware
chunking in section 5 of the lesson. Treat these as a planning model, then
measure on a golden set.

stdlib only.

Usage:
    python chunk_cost_sweep.py
    python chunk_cost_sweep.py --domain insurance
    python chunk_cost_sweep.py --domain retail --fixed-context
    python chunk_cost_sweep.py --answer-span 250 --overlap-pct 0.25
"""

from __future__ import annotations

import argparse

# ----------------------------------------------------------------------------
# PRICES — Q1-2026 ballparks. Refresh before quoting.
# ----------------------------------------------------------------------------

MANAGED_PG_RAM_GB_MONTH = 11.0
PG_RAM_HEADROOM = 1.8
SSD_GB_MONTH = 0.10

EMBED_USD_PER_MTOK = 0.02       # text-embedding-3-small class
LLM_IN_USD_PER_MTOK = 0.25      # small/fast model, cached-miss input
LLM_OUT_USD_PER_MTOK = 1.25
LLM_OUT_TOKENS = 350            # a grounded answer with citations

# Week 2 Day 2 established this: sizing on peak and billing on peak is the
# single most common way these estimates come out 4x too high.
PEAK_TO_AVERAGE_RATIO = 0.25
SECONDS_PER_MONTH = 2_592_000

CHUNK_SIZES = [128, 200, 256, 400, 512, 800, 1024, 1600]

# Re-embedding the corpus. Chunking changes are not free to roll out — you pay
# the ingest pass again every time you change the strategy.
REEMBED_PER_YEAR = 2


DOMAIN_PRESETS = {
    "retail": dict(
        corpus_tokens=3_200_000_000,   # 800k SKUs x ~4k tokens (specs, reviews, Q&A)
        dims=1536, answer_span=40, top_k=8, rag_qps=20.0,
        note="Answer spans are short — a spec value, a return window, a "
             "compatibility note. Short spans mean small chunks stay intact, so "
             "retail can afford the precision of small chunks. Note the 500 QPS "
             "from Week 2 is PRODUCT SEARCH, not RAG; the agent path is ~20 QPS "
             "peak and that distinction is worth 25x on this bill."),
    "insurance": dict(
        corpus_tokens=168_000_000,     # 12k filed forms x ~14k tokens
        dims=1536, answer_span=300, top_k=6, rag_qps=1.0,
        note="An answer is a full clause WITH its qualifiers — the exclusion "
             "that modifies the coverage grant, the definition the term depends "
             "on. 300-token spans mean anything under ~512 tokens fragments the "
             "answer, and a fragmented exclusion is a compliance incident, not a "
             "relevance miss."),
    "fintech-docs": dict(
        corpus_tokens=120_000_000,     # 15k regulatory/dispute docs x ~8k tokens
        dims=1536, answer_span=200, top_k=6, rag_qps=3.0,
        note="Regulatory clauses and dispute-resolution procedure. Between the "
             "other two: spans long enough to punish small chunks, corpus small "
             "enough that index cost never drives the decision."),
}


def containment(chunk: int, overlap: int, span: int) -> float:
    """P(a contiguous answer span of `span` tokens lands inside one chunk)."""
    if span >= chunk:
        return 0.0
    if overlap >= span:
        return 1.0
    stride = chunk - overlap
    return max(0.0, (chunk - span) / stride) if stride > 0 else 1.0


def sweep(args: argparse.Namespace) -> list[dict]:
    rows = []
    avg_qps = args.rag_qps * PEAK_TO_AVERAGE_RATIO
    queries_per_month = avg_qps * SECONDS_PER_MONTH

    for c in CHUNK_SIZES:
        overlap = int(c * args.overlap_pct)
        stride = c - overlap
        if stride <= 0:
            continue

        # Storage multiplier from overlap: every token is stored C/(C-O) times.
        n_chunks = args.corpus_tokens / stride
        stored_tokens = n_chunks * c

        heap_gb = n_chunks * (args.dims * 4 + 40) / 1e9
        index_gb = heap_gb                     # pgvector HNSW copies the vector
        ram_gb = max(8.0, (heap_gb + index_gb) * PG_RAM_HEADROOM)
        infra = ram_gb * MANAGED_PG_RAM_GB_MONTH + heap_gb * SSD_GB_MONTH

        # Ingest: you embed every stored token, overlap included.
        embed_monthly = (stored_tokens * REEMBED_PER_YEAR / 12) / 1e6 * EMBED_USD_PER_MTOK

        # Retrieval -> prompt. Two policies:
        #   fixed-k       : always retrieve k chunks (context shrinks with C)
        #   fixed-context : retrieve enough chunks to fill the same budget
        if args.fixed_context:
            k = max(1, round(args.context_budget / c))
        else:
            k = args.top_k
        ctx_tokens = k * c

        llm_in = queries_per_month * ctx_tokens / 1e6 * LLM_IN_USD_PER_MTOK
        llm_out = queries_per_month * LLM_OUT_TOKENS / 1e6 * LLM_OUT_USD_PER_MTOK
        generation = llm_in + llm_out

        p_intact = containment(c, overlap, args.answer_span)
        density = min(1.0, args.answer_span / c)

        rows.append(dict(
            chunk=c, overlap=overlap, k=k, ctx=ctx_tokens,
            n_chunks=n_chunks, store_mult=c / stride,
            index_gb=index_gb, infra=infra, embed=embed_monthly,
            generation=generation, total=infra + embed_monthly + generation,
            p_intact=p_intact, density=density,
        ))
    return rows


def report(args: argparse.Namespace, note: str | None) -> None:
    rows = sweep(args)
    avg_qps = args.rag_qps * PEAK_TO_AVERAGE_RATIO

    print("=" * 92)
    print("  CHUNK SIZE — JOINT QUALITY AND COST SWEEP")
    print("=" * 92)
    if args.domain:
        print(f"  Domain          : {args.domain.upper()}")
    print(f"  Corpus          : {args.corpus_tokens:,} tokens")
    print(f"  Answer span     : {args.answer_span} tokens (typical contiguous answer)")
    print(f"  Overlap         : {args.overlap_pct:.0%} of chunk size")
    print(f"  Retrieval       : {'fixed context budget ' + str(args.context_budget) + ' tok'
                                 if args.fixed_context else f'fixed k = {args.top_k}'}")
    print(f"  RAG traffic     : {args.rag_qps:,.1f} QPS peak"
          f" -> {avg_qps:,.2f} avg ({PEAK_TO_AVERAGE_RATIO:.0%} peak-to-average)")
    print(f"                    {avg_qps * SECONDS_PER_MONTH:,.0f} queries/month")
    print()

    print("-" * 92)
    print(f"  {'Chunk':>6}{'Ovl':>6}{'k':>4}{'Ctx tok':>9}{'Chunks':>13}"
          f"{'Index GB':>10}{'Infra $':>10}{'Gen $/mo':>12}{'Total $':>11}")
    print("  " + "-" * 88)
    for r in rows:
        print(f"  {r['chunk']:>6}{r['overlap']:>6}{r['k']:>4}{r['ctx']:>9,}"
              f"{r['n_chunks']:>13,.0f}{r['index_gb']:>10,.1f}{r['infra']:>10,.0f}"
              f"{r['generation']:>12,.0f}{r['total']:>11,.0f}")
    print()

    print("-" * 92)
    print("  WHAT THE CHUNK SIZE DOES TO RETRIEVAL QUALITY (geometry, not vibes)")
    print("-" * 92)
    print(f"  {'Chunk':>6}{'Ovl':>6}{'P(span intact)':>17}{'Signal density':>17}"
          f"{'Store mult':>13}{'Verdict':>22}")
    print("  " + "-" * 88)
    for r in rows:
        if r["p_intact"] == 0.0:
            verdict = "span never fits"
        elif r["p_intact"] >= 0.999:
            verdict = "overlap >= span: safe"
        elif r["p_intact"] < 0.8:
            verdict = "fragments answers"
        else:
            verdict = "acceptable"
        print(f"  {r['chunk']:>6}{r['overlap']:>6}{r['p_intact']:>16.1%}"
              f"{r['density']:>17.1%}{r['store_mult']:>12.2f}x{verdict:>22}")
    print()

    # ---- the headline: which cost actually moves -----------------------------
    small, large = rows[0], rows[-1]
    print("=" * 92)
    print("  WHERE THE MONEY ACTUALLY IS")
    print("=" * 92)
    ratio_small = small["generation"] / small["infra"] if small["infra"] else 0
    ratio_large = large["generation"] / large["infra"] if large["infra"] else 0
    print(f"  At {small['chunk']:,}-token chunks : infra ${small['infra']:,.0f}"
          f"  generation ${small['generation']:,.0f}"
          f"   ({ratio_small:,.1f}x the infra)")
    print(f"  At {large['chunk']:,}-token chunks : infra ${large['infra']:,.0f}"
          f"  generation ${large['generation']:,.0f}"
          f"   ({ratio_large:,.1f}x the infra)")
    print()
    if not args.fixed_context:
        cheapest = min(rows, key=lambda r: r["total"])
        print("  Halving the chunk size DOUBLES the index and HALVES the prompt, so the")
        print("  two costs move in OPPOSITE directions and total cost is U-shaped:")
        print(f"    index      ~ 1/C   ->  ${rows[0]['infra']:,.0f} at {rows[0]['chunk']}"
              f" falling to ${rows[-1]['infra']:,.0f} at {rows[-1]['chunk']}")
        print(f"    generation ~   C   ->  ${rows[0]['generation']:,.0f} at "
              f"{rows[0]['chunk']} rising to ${rows[-1]['generation']:,.0f} at "
              f"{rows[-1]['chunk']}")
        print(f"  Minimum at {cheapest['chunk']} tokens, ${cheapest['total']:,.0f}/mo."
              f"  An index-only cost model would have")
        print(f"  told you to make chunks as LARGE as possible — here that is "
              f"${rows[-1]['total']:,.0f}/mo,")
        print(f"  {rows[-1]['total'] / cheapest['total']:.1f}x the true optimum.")
    else:
        print("  Under a FIXED CONTEXT BUDGET, generation cost is flat by construction")
        print("  and only the index moves. This is the honest comparison when smaller")
        print("  chunks force you to retrieve more of them to cover the same material.")
    print()

    # ---- recommendation ------------------------------------------------------
    # Two ways to buy containment, and they are priced very differently:
    #   raise the CHUNK SIZE -> more tokens in every prompt -> generation cost
    #   raise the OVERLAP    -> more copies of every token  -> index cost
    # Since generation usually dominates, overlap is normally the cheaper buy.
    # Sweep both rather than assuming.
    grid = []
    for c in CHUNK_SIZES:
        for opct in [0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            saved = args.overlap_pct
            args.overlap_pct = opct
            for r in sweep(args):
                if r["chunk"] == c:
                    r["overlap_pct"] = opct
                    grid.append(r)
            args.overlap_pct = saved

    viable = [r for r in grid if r["p_intact"] >= args.containment_slo]
    fixed_ovl_viable = [r for r in rows if r["p_intact"] >= args.containment_slo]

    print("=" * 92)
    print("  RECOMMENDATION")
    print("=" * 92)
    if not viable:
        need = args.answer_span
        print(f"  NOTHING in the grid holds containment >= {args.containment_slo:.0%}.")
        print(f"  With {need}-token answer spans you need chunks well above {need}")
        print(f"  tokens, or overlap >= {need} tokens. Raise --context-budget or")
        print(f"  reconsider whether the answer really is one contiguous span.")
    else:
        best = min(viable, key=lambda r: r["total"])
        print(f"  Chunk size      : {best['chunk']} tokens, overlap {best['overlap']}"
              f" ({best['overlap_pct']:.0%})")
        print(f"  Chunks          : {best['n_chunks']:,.0f}"
              f"  ({best['store_mult']:.2f}x storage from overlap)")
        print(f"  P(span intact)  : {best['p_intact']:.1%}"
              f"   (SLO {args.containment_slo:.0%})")
        print(f"  Signal density  : {best['density']:.1%} — the rest of every retrieved")
        print(f"                    chunk is tokens you pay for in the prompt anyway")
        print(f"  Cost            : ${best['total']:,.0f}/mo"
              f"  (infra ${best['infra']:,.0f} + embed ${best['embed']:,.0f}"
              f" + generation ${best['generation']:,.0f})")

        if fixed_ovl_viable:
            naive = min(fixed_ovl_viable, key=lambda r: r["total"])
            if naive["chunk"] != best["chunk"] or naive["overlap"] != best["overlap"]:
                print()
                print(f"  Holding overlap at {args.overlap_pct:.0%} and buying containment")
                print(f"  with CHUNK SIZE instead would mean {naive['chunk']}-token chunks")
                print(f"  at ${naive['total']:,.0f}/mo — "
                      f"{naive['total'] / best['total']:.1f}x more expensive for the same")
                print(f"  guarantee, because every extra token rides along in every prompt.")
        print()
        rule = int(args.answer_span)
        print(f"  THE RULE THIS FALLS OUT OF:")
        print(f"    Buy containment with OVERLAP, not with chunk size. Overlap >= the")
        print(f"    typical answer span ({rule} tokens) guarantees no answer is ever")
        print(f"    split, and it is paid for once in storage. Chunk size is paid for")
        print(f"    again in every single prompt that retrieves the chunk.")
        print(f"    Then take the SMALLEST chunk that clears the bar — signal density")
        print(f"    falls as 1/C and you pay for the noise twice.")

    if note:
        print()
        print("  DOMAIN NOTE")
        line = ""
        for w in note.split():
            if len(line) + len(w) + 1 > 84:
                print(f"    {line}")
                line = w
            else:
                line = f"{line} {w}".strip()
        print(f"    {line}")

    print()
    print("=" * 92)
    print("  THE TRAP")
    print("=" * 92)
    print("  'Smaller chunks are more precise' is true and incomplete. Precision rises")
    print("  as 1/C while the probability the answer survives intact FALLS. Below the")
    print("  answer-span length the chunk cannot contain the answer at all, and no")
    print("  amount of reranking recovers a fact that was cut in half at ingest.")
    print("=" * 92)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--corpus-tokens", type=int)
    p.add_argument("--dims", type=int)
    p.add_argument("--answer-span", type=int,
                   help="typical contiguous answer length, in tokens")
    p.add_argument("--overlap-pct", type=float, default=0.15)
    p.add_argument("--top-k", type=int)
    p.add_argument("--rag-qps", type=float, help="PEAK RAG queries per second")
    p.add_argument("--fixed-context", action="store_true",
                   help="retrieve enough chunks to fill a constant token budget "
                        "instead of a constant k")
    p.add_argument("--context-budget", type=int, default=6400)
    p.add_argument("--containment-slo", type=float, default=0.95)
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
