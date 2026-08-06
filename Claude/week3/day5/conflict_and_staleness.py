"""
Week 3, Day 5 — Conflicting Sources, Quantified
================================================

Week 2 covered staleness as an INFRASTRUCTURE problem: dual-write drift, index
lag, recall decay. This is the retrieval-layer version of the same problem, and
it is nastier, because nothing is broken:

    every chunk is correctly stored, correctly embedded, correctly retrieved,
    and the answer is still wrong

The mechanism: a document that has been revised produces several near-identical
chunks — v1, v2, v3 of the same clause. They are near-identical, so they sit
almost on top of each other in embedding space, so a top-k retrieval that finds
one tends to find them all. The model receives three versions of the same rule,
is told nothing about which is current, and picks one. Often the wrong one, and
always without saying it had a choice.

WHAT THIS SCRIPT COMPUTES

  1. P(the context contains two or more versions of the same clause) as a
     function of version count and k. This is high, and it is high for a
     structural reason rather than a tuning reason: similarity is exactly what
     makes them collide.

  2. What each mitigation is actually worth, in order:
       - effective-date filter      (a WHERE clause; near-total fix)
       - version-aware dedup        (post-retrieval; keeps the newest per clause)
       - "cite the effective date"  (prompt-level; detects, does not prevent)
       - nothing                    (the default, and the baseline here)

  3. The interaction nobody expects: **reranking makes this worse.** A
     cross-encoder scores each candidate against the query independently, and
     all versions match the query about equally well, so a reranker will
     happily promote several versions into the top-k it hands to the model.
     Precision improves and version-conflict gets more likely.

Modelled, not measured: version similarity is treated as a parameter. Set it
from your own corpus by embedding two consecutive versions of a real clause.
The conclusions are not sensitive to the exact value — they follow from the
versions being similar at all.

stdlib only.

Usage:
    python conflict_and_staleness.py
    python conflict_and_staleness.py --domain insurance
    python conflict_and_staleness.py --versions 6 --k 8
    python conflict_and_staleness.py --rerank
"""

from __future__ import annotations

import argparse
import random

DOMAIN_PRESETS = {
    "insurance": dict(
        versions=4, k=6, clause_share=0.25, distinct_clauses=40,
        note="Filed forms are amended and re-filed by state. Four versions of a "
             "suicide exclusion is normal, they differ by a single sentence, and "
             "the difference is the entire legal content. This is the domain "
             "where a version conflict is a regulatory finding rather than a bad "
             "search result."),
    "fintech-docs": dict(
        versions=3, k=6, clause_share=0.30, distinct_clauses=60,
        note="Network rules are revised on a published schedule. Old reason-code "
             "tables stay in the corpus because disputes are adjudicated under "
             "the rules in force at transaction time — so you cannot simply "
             "delete them, which is exactly what makes this hard."),
    "retail": dict(
        versions=2, k=8, clause_share=0.15, distinct_clauses=200,
        note="Policy pages (returns, shipping, warranty) get revised; product "
             "copy gets duplicated across variants. Lower stakes, same mechanism "
             "— and the fix is cheaper because retail can simply delete the old "
             "version, an option regulated domains do not have."),
}


def simulate(args: argparse.Namespace, strategy: str, trials: int,
             seed: int) -> dict:
    """
    One trial = one query that targets a versioned clause.

    Retrieval model: the k slots are filled by scoring candidates. Versions of
    the TARGET clause all score near the top because they are near-identical to
    each other and to what the query is asking about. Other clauses compete for
    the remaining slots.
    """
    rng = random.Random(seed)
    conflicts = 0
    wrong_version_alone = 0
    slots_wasted = 0

    for _ in range(trials):
        # Score every candidate. Versions of the target clause get a high base
        # score plus small independent noise — that noise is the ONLY thing
        # separating them, which is the whole problem.
        target = [("v%d" % v, 1.0 - 0.02 * v + rng.gauss(0, args.version_noise))
                  for v in range(args.versions)]
        # Other clauses genuinely compete — some are more relevant to the query
        # than some versions of the target. A model where versions always sweep
        # the top-k would overstate the problem and prove nothing.
        others = [("other", rng.uniform(0.0, 1.15))
                  for _ in range(args.distinct_clauses)]

        cands = target + others

        if strategy == "date_filter":
            # Only the currently-effective version survives the WHERE clause.
            cands = [c for c in cands if c[0] in ("v0", "other")]
        elif strategy == "rerank":
            # A cross-encoder scores each candidate against the query. All
            # versions answer the query about equally well, so it tightens the
            # scores rather than separating them — and pushes them UP.
            cands = [(n, (s + 0.15 if n.startswith("v") else s * 0.9))
                     for n, s in cands]

        cands.sort(key=lambda c: -c[1])
        topk = cands[:args.k]

        if strategy == "dedup":
            # VERSION-AWARE dedup: keep the NEWEST version of the clause, not
            # the highest-scoring one. Keeping the top-scoring version is the
            # obvious implementation and it is wrong — retrieval score says
            # nothing about which version is in force.
            kept, used_version = [], False
            for n, sc in cands:
                if n.startswith("v"):
                    if used_version or n != "v0":
                        continue
                    used_version = True
                kept.append((n, sc))
                if len(kept) == args.k:
                    break
            topk = kept

        versions_in_context = [n for n, _ in topk if n.startswith("v")]
        if len(versions_in_context) >= 2:
            conflicts += 1
            slots_wasted += len(versions_in_context) - 1
        if versions_in_context and "v0" not in versions_in_context:
            wrong_version_alone += 1

    return dict(
        conflict_rate=conflicts / trials,
        wrong_only_rate=wrong_version_alone / trials,
        avg_slots_wasted=slots_wasted / trials,
    )


def report(args: argparse.Namespace, note: str | None) -> None:
    print("=" * 90)
    print("  CONFLICTING SOURCES — WHEN EVERY CHUNK IS CORRECT AND THE ANSWER IS NOT")
    print("=" * 90)
    if args.domain:
        print(f"  Domain            : {args.domain.upper()}")
    print(f"  Versions per clause: {args.versions}"
          f"   (v0 = currently effective)")
    print(f"  Top-k to the model : {args.k}")
    print(f"  Other clauses      : {args.distinct_clauses}")
    print(f"  Trials             : {args.trials:,}")
    print()

    strategies = [
        ("none", "nothing (the default)"),
        ("rerank", "cross-encoder rerank, version-blind"),
        ("dedup", "version-aware dedup (keep newest)"),
        ("date_filter", "effective-date WHERE clause"),
    ]

    print("-" * 90)
    print(f"  {'strategy':<36}{'conflict':>11}{'wrong only':>13}{'slots lost':>13}")
    print("  " + "-" * 86)
    results = {}
    for name, label in strategies:
        r = simulate(args, name, args.trials, args.seed)
        results[name] = r
        print(f"  {label:<36}{r['conflict_rate']:>10.1%}"
              f"{r['wrong_only_rate']:>13.1%}{r['avg_slots_wasted']:>13.2f}")
    print()
    print("  conflict   = context contained 2+ versions of the same clause")
    print("  wrong only = context contained a version, but NOT the effective one")
    print("  slots lost = top-k slots consumed by redundant versions")
    print()

    base = results["none"]["conflict_rate"]
    rer = results["rerank"]["conflict_rate"]

    print("=" * 90)
    print("  THREE THINGS TO TAKE FROM THAT TABLE")
    print("=" * 90)
    print(f"  1. THE BASELINE IS NOT RARE. {base:.0%} of queries about a versioned")
    print(f"     clause put two or more versions in front of the model. Nothing is")
    print(f"     broken; similarity is doing exactly what it is supposed to do, and")
    print(f"     near-identical text is near-identical in embedding space.")
    print()
    if rer >= base:
        print(f"  2. RERANKING MAKES IT WORSE: {base:.0%} -> {rer:.0%}. A cross-encoder")
        print(f"     scores each candidate against the QUERY, independently. Every")
        print(f"     version answers the query about equally well, so the reranker")
        print(f"     promotes all of them. Day 3's precision win and this failure")
        print(f"     mode are the same mechanism seen from two sides.")
    else:
        print(f"  2. Reranking moved conflicts {base:.0%} -> {rer:.0%} at these settings.")
    print()
    print(f"  3. THE FIX IS A WHERE CLAUSE, NOT A MODEL. The effective-date filter")
    print(f"     takes conflicts to {results['date_filter']['conflict_rate']:.0%} and")
    print(f"     costs one indexed predicate. Dedup gets to"
          f" {results['dedup']['conflict_rate']:.0%} and is the")
    print(f"     fallback when 'current' is not expressible as a date — but it is")
    print(f"     strictly worse, because it fires after you have already spent")
    print(f"     retrieval slots on redundant versions.")
    print()
    print(f"     Reclaimed slots: {results['none']['avg_slots_wasted']:.2f}"
          f" -> {results['date_filter']['avg_slots_wasted']:.2f} per query. Those slots")
    print(f"     were being paid for in prompt tokens AND crowding out the other")
    print(f"     documents the answer needed.")
    print()

    print("-" * 90)
    print("  WHY THE PROMPT-LEVEL FIX IS NOT A FIX")
    print("-" * 90)
    for line in [
        "'Instruct the model to cite the effective date and prefer the newest'",
        "is the cheapest thing to try and it belongs in the prompt anyway. But",
        "note what it can and cannot do:",
        "",
        "  it CAN     surface the ambiguity, so a human sees there was a choice",
        "  it CANNOT  add the current version if retrieval never returned it",
        "",
        f"In this run, {results['none']['wrong_only_rate']:.0%} of queries put a version"
        " in the context WITHOUT the",
        "effective one. For those, no instruction helps — the right text is not",
        "there to prefer. That is the difference between a detection control and",
        "a prevention control, and audit committees ask which one you have.",
    ]:
        print(f"  {line}")
    print()

    print("=" * 90)
    print("  THE SCHEMA THAT PREVENTS IT")
    print("=" * 90)
    for line in [
        "This is the bitemporal model from Week 2 Day 1, now earning its keep:",
        "",
        "  ALTER TABLE chunk ADD COLUMN clause_key      text;   -- identity",
        "  ALTER TABLE chunk ADD COLUMN doc_version     int;",
        "  ALTER TABLE chunk ADD COLUMN effective_from  date;",
        "  ALTER TABLE chunk ADD COLUMN effective_to    date;   -- NULL = current",
        "  ALTER TABLE chunk ADD COLUMN jurisdiction    text;",
        "",
        "  CREATE INDEX ON chunk (clause_key, effective_from DESC);",
        "",
        "  -- and every retrieval carries the AS-OF date, which is NOT today():",
        "  WHERE effective_from <= :as_of",
        "    AND (effective_to IS NULL OR effective_to > :as_of)",
        "    AND jurisdiction IN (:state, 'ALL')",
        "",
        "The as-of date is the load-bearing detail. For a claim filed in 2023 the",
        "question is what the policy said THEN, not now. A system that always",
        "filters to current is wrong in a different way — quietly, and in the",
        "direction that loses litigation.",
        "",
        "Week 2 Day 3 established that filter selectivity decides your whole",
        "architecture. This is the same predicate doing a second job: it is a",
        "correctness control first and a performance lever second.",
    ]:
        print(f"  {line}")

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
    print("=" * 90)
    print("  WHAT MAKES THIS FAILURE MODE DIFFERENT")
    print("=" * 90)
    for line in [
        "Every other failure in this lesson announces itself. Retrieval misses",
        "produce visibly unhelpful answers; hallucinations contradict the source;",
        "stale indexes produce complaints.",
        "",
        "A version conflict produces a fluent, well-cited, plausible answer drawn",
        "from a real document that was true last year. Groundedness scores it as",
        "GROUNDED, because the claim IS supported by a retrieved chunk. Your",
        "Day 4 metrics will not catch it.",
        "",
        "The eval that catches it has to be built on purpose: questions whose",
        "answer CHANGED between versions, scored against the version effective on",
        "a stated as-of date. If your golden set has no such questions, you have",
        "no evidence about this failure mode at all.",
    ]:
        print(f"  {line}")
    print("=" * 90)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--versions", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--distinct-clauses", type=int)
    p.add_argument("--version-noise", type=float, default=0.06,
                   help="how much retrieval scores differ between versions of "
                        "the same clause — small, because they are near-identical")
    p.add_argument("--trials", type=int, default=20000)
    p.add_argument("--seed", type=int, default=11)
    args = p.parse_args()

    base = dict(DOMAIN_PRESETS["insurance"])
    if args.domain:
        base = dict(DOMAIN_PRESETS[args.domain])
    note = base.pop("note", None)
    base.pop("clause_share", None)
    for k, v in base.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)

    report(args, note if args.domain else None)


if __name__ == "__main__":
    main()
