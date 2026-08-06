"""
Week 3, Day 4 — RAG Evaluation, With the Statistics Nobody Runs
===============================================================

Days 1-3 produced three decisions that all need measuring:

    chunk size and overlap   (Day 1)
    hybrid arm weighting     (Day 2)
    reranker and candidate N (Day 3)

Every one of them gets "evaluated" the same way in practice: run 40 golden
queries against A and B, compare mean nDCG, ship the winner. This script exists
because that procedure is usually incapable of distinguishing the two systems,
and nobody checks.

WHAT IT DOES

  1. Retrieval metrics done properly — recall@k, precision@k, MRR, nDCG@k —
     computed per query, then aggregated per QUERY CLASS as well as overall.
     (Week 2 Day 4: aggregates hide class-level regressions.)

  2. A PAIRED BOOTSTRAP between two systems. Paired, because both systems ran
     the same queries and the per-query correlation is large — an unpaired test
     throws that information away and needs several times the sample size.

  3. A POWER SWEEP — the part that is almost never run. Given the observed
     per-query variance, what is the smallest true improvement you could
     reliably detect at n = 20, 50, 100, 500 queries?

     This is the number that tells you whether your eval set can support the
     decision you are about to make with it. A 40-query golden set typically
     cannot resolve anything smaller than ~0.08 nDCG, which is larger than most
     real chunking or reranking gains.

THE POINT

  "B scored 0.03 higher" is not a result. It is a measurement whose error bars
  you have not computed. Half the tuning work in RAG systems is spent chasing
  differences that are indistinguishable from resampling noise, and the fix is
  cheap: a bootstrap is twenty lines, and a power sweep tells you how many
  queries to label BEFORE you spend a month labelling the wrong number.

stdlib only. Sample run data is generated deterministically from a fixed seed
so the numbers are reproducible; replace load_runs() with your own.

Usage:
    python rag_eval_harness.py
    python rag_eval_harness.py --true-delta 0.03      # a realistic small gain
    python rag_eval_harness.py --n-queries 500
    python rag_eval_harness.py --alpha 0.01
"""

from __future__ import annotations

import argparse
import math
import random

# ----------------------------------------------------------------------------
# Query classes carried forward from Week 2 Day 4. Chunking and reranking
# affect these differently, which is the entire reason to keep them separate.
# ----------------------------------------------------------------------------

CLASSES = {
    "EXACT_ID":         dict(share=0.22, base=0.91, spread=0.10,
                             note="SKU / form number / reason code. Nearly "
                                  "solved, and nearly immune to reranking."),
    "NATURAL_LANGUAGE": dict(share=0.41, base=0.63, spread=0.26,
                             note="Questions and descriptions. Highest variance "
                                  "and where all the headroom lives."),
    "NAVIGATIONAL":     dict(share=0.19, base=0.78, spread=0.15,
                             note="Brand / category browse. Ordering is a "
                                  "business-rules problem, not a relevance one."),
    "AMBIGUOUS":        dict(share=0.18, base=0.48, spread=0.30,
                             note="Short, underspecified. Worst scores, widest "
                                  "spread — and the class most likely to produce "
                                  "a false 'improvement'."),
}


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def logit(p: float) -> float:
    p = min(0.995, max(0.005, p))
    return math.log(p / (1 - p))


def inv_logit(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


def load_runs(n: int, true_delta: float, seed: int) -> list[dict]:
    """
    Generate a paired evaluation run. Replace this with your real data.

    The important structural property, which real data also has: system A and
    system B scores are strongly CORRELATED per query, because a hard query is
    hard for both. That correlation is what makes a paired test far more
    powerful than an unpaired one, and it is why using an unpaired t-test on
    RAG evals is both common and wasteful.
    """
    rng = random.Random(seed)
    rows = []
    names = list(CLASSES)
    weights = [CLASSES[c]["share"] for c in names]

    # The shape that matters. A retrieval change does not nudge every query a
    # little — it leaves MOST queries untouched and flips a minority hard,
    # because a query either surfaces the right chunk or it does not. So the
    # per-query difference is a spike at zero plus a wide tail, and its standard
    # deviation is an order of magnitude larger than a "small noise" model
    # suggests. Getting this wrong makes an eval set look far more powerful than
    # it is, which is exactly the mistake this script exists to catch.
    P_CHANGE = 0.35
    FLIP_SD = 1.6          # in logit units
    LOGIT_SLOPE = 0.21     # ~ p(1-p) at the corpus mean; converts logit -> score

    for i in range(n):
        cls = rng.choices(names, weights=weights)[0]
        cfg = CLASSES[cls]
        difficulty = rng.gauss(0, 1)
        a = clamp(cfg["base"] + difficulty * cfg["spread"])

        # EXACT_ID barely moves — Day 3: a reranker cannot help where there is
        # no headroom.
        scale = 0.15 if cls == "EXACT_ID" else 1.0
        # Shift in LOGIT space, not score space. Scores are bounded in [0,1],
        # so adding a constant and clamping introduces a systematic negative
        # bias — a query at 0.95 loses most of a positive shift to the clamp
        # while absorbing all of a negative one. That artifact is large enough
        # to flip the sign of the population effect, which would invalidate the
        # whole simulation. A logit shift is bounded by construction and still
        # produces the real ceiling effect: the same shift moves a score near
        # 0.5 far more than one near 0.95.
        if rng.random() < P_CHANGE:
            logit_shift = (true_delta * scale / P_CHANGE) / LOGIT_SLOPE \
                + rng.gauss(0, FLIP_SD)
        else:
            logit_shift = 0.0
        rows.append(dict(qid=i, cls=cls, a=a,
                         b=inv_logit(logit(a) + logit_shift)))
    return rows


def population_effect(true_delta: float, seed: int, n: int = 200_000) -> float:
    """
    The actual mean difference this data-generating process produces.

    --true-delta is an input to the process, not the population effect, because
    the ceiling effect shrinks gains on already-good queries. Reporting the
    parameter as if it were the truth would be exactly the kind of unexamined
    claim this script is about.
    """
    rows = load_runs(n, true_delta, seed + 991)
    return sum(r["b"] - r["a"] for r in rows) / len(rows)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

def dcg(relevances: list[float]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg_at_k(retrieved_rel: list[float], ideal_rel: list[float], k: int) -> float:
    idcg = dcg(sorted(ideal_rel, reverse=True)[:k])
    return dcg(retrieved_rel[:k]) / idcg if idcg else 0.0


def recall_at_k(retrieved_ids: list[int], relevant_ids: set[int], k: int) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(retrieved_ids[:k]) & relevant_ids) / len(relevant_ids)


def mrr(retrieved_ids: list[int], relevant_ids: set[int]) -> float:
    for i, d in enumerate(retrieved_ids, 1):
        if d in relevant_ids:
            return 1.0 / i
    return 0.0


def demo_metrics() -> None:
    """One worked example, so the definitions are concrete rather than named."""
    retrieved = [11, 4, 7, 2, 9]
    relevant = {4, 9, 15}
    grades = {4: 3.0, 9: 2.0, 15: 3.0}          # graded relevance, 0-3
    rel_seq = [grades.get(d, 0.0) for d in retrieved]
    ideal = list(grades.values())

    print("-" * 92)
    print("  THE METRICS, ON ONE QUERY")
    print("-" * 92)
    print(f"  retrieved (ranked) : {retrieved}")
    print(f"  relevant           : {sorted(relevant)}  graded {grades}")
    print(f"  relevance sequence : {rel_seq}")
    print()
    print(f"  recall@5    = {recall_at_k(retrieved, relevant, 5):.3f}"
          f"   found 2 of 3 relevant documents")
    print(f"  precision@5 = {len(set(retrieved[:5]) & relevant) / 5:.3f}"
          f"   2 of the 5 returned were relevant")
    print(f"  MRR         = {mrr(retrieved, relevant):.3f}"
          f"   first relevant hit was at position 2")
    print(f"  nDCG@5      = {ndcg_at_k(rel_seq, ideal, 5):.3f}"
          f"   position-weighted, graded, normalised")
    print()
    for line in [
        "Which to report, and why it matters that you say so:",
        "",
        "  recall@N     the CEILING for anything downstream. Day 3's whole",
        "               argument. Report it for the candidate set, not for k.",
        "  nDCG@k       the one to optimise for a user-facing ranking. Graded and",
        "               position-weighted, so promoting the best answer from rank",
        "               3 to rank 1 shows up. recall@k would not move at all.",
        "  MRR          right when there is exactly ONE correct answer — an exact",
        "               ID lookup, a policy clause. Wrong for 'show me options'.",
        "  precision@k  matters when the prompt budget is tight, because every",
        "               irrelevant chunk is paid for twice (Day 1).",
        "",
        "A team reporting only nDCG@5 cannot tell a retrieval failure from a",
        "ranking failure, and those have completely different fixes.",
    ]:
        print(f"  {line}")
    print()


# ----------------------------------------------------------------------------
# Paired bootstrap
# ----------------------------------------------------------------------------

def paired_bootstrap(diffs: list[float], iters: int, seed: int,
                     alpha: float) -> tuple[float, float, float, float]:
    """Return (mean, lo, hi, p_two_sided) for the mean paired difference."""
    rng = random.Random(seed)
    n = len(diffs)
    observed = sum(diffs) / n
    means = []
    for _ in range(iters):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        means.append(s)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[min(iters - 1, int((1 - alpha / 2) * iters))]
    # Two-sided p: how often does a recentred bootstrap reach the observed?
    centred = [m - observed for m in means]
    p = sum(1 for c in centred if abs(c) >= abs(observed)) / iters
    return observed, lo, hi, p


def stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


# ----------------------------------------------------------------------------

def report(args: argparse.Namespace) -> None:
    rows = load_runs(args.n_queries, args.true_delta, args.seed)

    print("=" * 92)
    print("  RAG EVALUATION HARNESS")
    print("=" * 92)
    realized = sum(r["b"] - r["a"] for r in rows) / len(rows)
    truth = population_effect(args.true_delta, args.seed)
    print(f"  Golden queries  : {args.n_queries}")
    print(f"  TRUE effect     : {truth:+.4f} nDCG  (measured over 200,000"
          f" simulated queries)")
    print(f"  This sample says: {realized:+.4f} nDCG")
    print(f"  SIMULATED data — knowing the truth is the only way to check"
          f" whether the")
    print(f"  statistics would have told it to us.")
    print(f"  Confidence      : {(1 - args.alpha) * 100:.0f}%")
    print(f"  Bootstrap iters : {args.iters:,}")
    print()

    demo_metrics()

    # ---- per class ---------------------------------------------------------
    print("=" * 92)
    print("  SYSTEM A vs SYSTEM B, BY QUERY CLASS")
    print("=" * 92)
    print(f"  {'class':<20}{'n':>5}{'A':>8}{'B':>8}{'delta':>9}"
          f"{'95% CI':>20}{'verdict':>18}")
    print("  " + "-" * 88)

    misleading = []
    for cls in CLASSES:
        sub = [r for r in rows if r["cls"] == cls]
        if not sub:
            continue
        diffs = [r["b"] - r["a"] for r in sub]
        a_mean = sum(r["a"] for r in sub) / len(sub)
        b_mean = sum(r["b"] for r in sub) / len(sub)
        d, lo, hi, p = paired_bootstrap(diffs, args.iters, args.seed + 1, args.alpha)
        sig = lo > 0 or hi < 0
        verdict = ("significant" if sig else "NOT significant")
        if not sig and abs(d) > 0.02:
            misleading.append((cls, d, lo, hi, len(sub)))
        print(f"  {cls:<20}{len(sub):>5}{a_mean:>8.3f}{b_mean:>8.3f}{d:>+9.3f}"
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}{verdict:>18}")

    diffs_all = [r["b"] - r["a"] for r in rows]
    a_all = sum(r["a"] for r in rows) / len(rows)
    b_all = sum(r["b"] for r in rows) / len(rows)
    d, lo, hi, p = paired_bootstrap(diffs_all, args.iters, args.seed + 1, args.alpha)
    print("  " + "-" * 88)
    print(f"  {'OVERALL':<20}{len(rows):>5}{a_all:>8.3f}{b_all:>8.3f}{d:>+9.3f}"
          f"{f'[{lo:+.3f}, {hi:+.3f}]':>20}"
          f"{('significant' if lo > 0 or hi < 0 else 'NOT significant'):>18}")
    print(f"  bootstrap p (two-sided) = {p:.4f}")
    print()

    if misleading:
        print("-" * 92)
        print("  CLASSES WHERE THE POINT ESTIMATE LOOKS REAL AND IS NOT")
        print("-" * 92)
        for cls, dd, l, h, n in misleading:
            print(f"  {cls:<20}{dd:+.3f} with CI [{l:+.3f}, {h:+.3f}] on n={n}")
            print(f"  {'':<20}the interval spans zero — this class cannot be called")
        print()
        print("  These are the rows that get screenshotted into a decision doc as")
        print("  'B improves NATURAL_LANGUAGE by 4 points'. Per-class slicing is")
        print("  necessary (Week 2 Day 4) and it multiplies your noise problem,")
        print("  because each slice has a fraction of the sample.")
        print()

    # ---- the payoff --------------------------------------------------------
    sd_all = stdev(diffs_all)
    mde_here = 2.80 * sd_all / math.sqrt(len(rows))
    print("=" * 92)
    print("  WHAT JUST HAPPENED")
    print("=" * 92)
    print(f"  Truth               : {truth:+.4f}   (B really is"
          f" {'better' if truth > 0 else 'worse'})")
    print(f"  This eval measured  : {d:+.4f}   CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  Smallest detectable : {mde_here:+.4f}   at n={len(rows)}")
    print()
    wrong_sign = (truth > 0) != (d > 0)
    if wrong_sign:
        for line in [
            "THE POINT ESTIMATE HAS THE WRONG SIGN.",
            "",
            "Not because anything was implemented incorrectly — the sampling did",
            f"exactly what sampling does. The true effect ({truth:+.3f}) is well",
            f"below what {len(rows)} queries can resolve ({mde_here:.3f}), so the",
            "measurement is dominated by which queries happened to be drawn.",
            "",
            "Note what each procedure would have done here:",
            "",
            "  compare the means, ship the winner  ->  ships A. Rejects a real",
            "                                          improvement, and records a",
            "                                          number with the wrong sign",
            "                                          in the decision doc.",
            "",
            "  bootstrap first                     ->  reports 'not significant',",
            "                                          which is CORRECT and is the",
            "                                          only honest answer available",
            "                                          from this much data.",
            "",
            "The bootstrap did not find the truth. Nothing could, from 50 queries.",
            "What it did was stop you from asserting a falsehood — and that is the",
            "entire value of the twenty lines it takes to run one.",
        ]:
            print(f"  {line}")
    elif abs(d) < mde_here:
        for line in [
            "The sign happens to be right, and it is still not a finding: the",
            f"observed {d:+.3f} is inside the {mde_here:.3f} noise floor for this",
            "sample size. Being right by luck is indistinguishable, from inside",
            "the experiment, from being right on purpose.",
        ]:
            print(f"  {line}")
    else:
        print("  The effect is large enough for this sample size to resolve. This is")
        print("  what it looks like when the eval set is adequate for the decision.")
    print()

    # ---- power sweep -------------------------------------------------------
    print("=" * 92)
    print("  POWER — WHAT COULD THIS EVAL SET ACTUALLY DETECT?")
    print("=" * 92)
    sd = stdev(diffs_all)
    print(f"  Observed per-query SD of the paired difference: {sd:.4f}")
    print()
    print("  Minimum detectable effect (MDE) at 80% power, two-sided:")
    print(f"    MDE = (z_alpha/2 + z_beta) * SD / sqrt(n) = 2.80 * {sd:.4f} / sqrt(n)")
    print()
    print(f"  {'queries':>10}{'MDE (nDCG)':>16}   what that means")
    print("  " + "-" * 88)
    for n in [20, 40, 50, 100, 250, 500, 1000]:
        mde = 2.80 * sd / math.sqrt(n)
        if mde > 0.08:
            meaning = "only huge changes are visible"
        elif mde > 0.04:
            meaning = "typical reranking gains are NOT resolvable"
        elif mde > 0.02:
            meaning = "resolves reranking, not chunk tuning"
        else:
            meaning = "resolves realistic tuning deltas"
        star = "  <-- your eval set" if n == args.n_queries else ""
        print(f"  {n:>10}{mde:>16.4f}   {meaning}{star}")
    print()

    need_for = {}
    for target in (0.05, 0.03, 0.02):
        need_for[target] = math.ceil((2.80 * sd / target) ** 2)
    print("  To detect a TRUE improvement of:")
    for target, n in need_for.items():
        print(f"    {target:+.2f} nDCG  ->  you need about {n:,} golden queries")
    print()
    for line in [
        "And per class, multiply again: to resolve the same effect inside",
        f"NATURAL_LANGUAGE at {CLASSES['NATURAL_LANGUAGE']['share']:.0%} of traffic you need",
        f"roughly {math.ceil(need_for[0.03] / CLASSES['NATURAL_LANGUAGE']['share']):,}"
        " total queries, because only that share lands in the slice.",
        "",
        "This is the calculation to run BEFORE labelling, not after. Labelling",
        "is the expensive part of evaluation and the only part whose cost you",
        "control up front.",
    ]:
        print(f"  {line}")
    print()

    print("=" * 92)
    print("  WHAT TO DO WHEN YOU CANNOT AFFORD THE QUERIES")
    print("=" * 92)
    for line in [
        "Labelling 3,000 queries is not always realistic. Four honest options,",
        "roughly in order of value:",
        "",
        "  1. REDUCE VARIANCE INSTEAD OF ADDING SAMPLES. The paired design is",
        "     already doing this. Go further: fix the query set across runs, fix",
        "     seeds, and hold everything constant except the one change. Variance",
        "     you remove is worth the same as sample you add, and it is free.",
        "",
        "  2. MEASURE A LOWER-VARIANCE QUANTITY. recall@N has far less per-query",
        "     variance than nDCG@5, because it is not sensitive to ordering. If",
        "     the change is a retrieval change, measure retrieval.",
        "",
        "  3. ACCEPT A DIRECTIONAL READ, AND SAY SO. 'B is probably better, CI",
        "     [-0.01, +0.07], we shipped it because it is also simpler' is an",
        "     honest engineering decision. 'B is 3% better' is a false claim.",
        "",
        "  4. USE AN LLM JUDGE TO EXPAND THE SET — carefully. It buys sample size",
        "     at the cost of label quality, and judge bias is systematic rather",
        "     than random, so it does NOT average out with more samples. Calibrate",
        "     against a human-labelled subset and report the agreement rate.",
        "",
        "The failure to avoid is running the same 40 queries a hundred times",
        "while tuning. That is not evaluation, it is optimisation against the",
        "sample — and the gains it finds will not survive contact with traffic.",
    ]:
        print(f"  {line}")
    print("=" * 92)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-queries", type=int, default=50)
    p.add_argument("--true-delta", type=float, default=0.08,
                   help="strength of the improvement fed INTO the simulation. "
                        "Not the population effect — the ceiling effect shrinks "
                        "it. The script measures and reports the real one.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--seed", type=int, default=17)
    report(p.parse_args())


if __name__ == "__main__":
    main()
