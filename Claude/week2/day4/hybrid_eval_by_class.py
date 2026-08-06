"""
Week 2, Day 4 — The Hybrid Evaluation Trap
==========================================

Reproduces the mistake that makes a hybrid search rollout look like a win while
it quietly damages your most valuable traffic.

You measure nDCG@10 on a held-out query set. It goes up. You ship. Weeks later
search-attributed revenue is flat or down.

The cause: aggregate metrics average over query classes of very different SIZE
and very different VALUE. Blanket hybrid helps the big, low-intent classes and
hurts the small, high-intent one — because blending dense results into a query
that already had a perfect exact match can only dilute rank 1.

This script computes three things from the same per-class measurements:
    1. traffic-weighted nDCG  (the number people report)     -> improves
    2. revenue-weighted nDCG  (the number that matters)      -> regresses
    3. the routed variant, where EXACT_IDENTIFIER bypasses fusion

stdlib only.

Usage:
    python hybrid_eval_by_class.py
    python hybrid_eval_by_class.py --show-ndcg-derivation
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

# ----------------------------------------------------------------------------
# Per-class measurements. Replace with your own golden-set numbers.
#
# conversion_index: conversion rate of this class relative to the average query.
# Exact-identifier searchers already know what they want; they convert several
# times higher than someone typing a descriptive phrase.
# ----------------------------------------------------------------------------


@dataclass
class ClassResult:
    name: str
    traffic_share: float
    conversion_index: float
    ndcg_baseline: float   # dense-only
    ndcg_blanket: float    # hybrid with one global alpha
    ndcg_routed: float     # hybrid with per-class routing (Day 4 router)


CLASSES = [
    #                            share  conv   base  blanket routed
    ClassResult("NATURAL_LANGUAGE",     0.25, 1.0,  0.58, 0.71, 0.71),
    ClassResult("ATTRIBUTE_CONSTRAINED", 0.30, 1.4,  0.62, 0.70, 0.72),
    ClassResult("MISSPELLED",           0.10, 0.8,  0.44, 0.63, 0.66),
    ClassResult("BRAND_MODEL",          0.27, 2.1,  0.73, 0.70, 0.79),
    ClassResult("EXACT_IDENTIFIER",     0.08, 4.3,  0.94, 0.71, 0.97),
]


def weighted(classes: list[ClassResult], metric: str, by_value: bool) -> float:
    """Weight per-class nDCG by traffic share, optionally scaled by conversion."""
    num = den = 0.0
    for c in classes:
        w = c.traffic_share * (c.conversion_index if by_value else 1.0)
        num += w * getattr(c, metric)
        den += w
    return num / den


def _delta(a: float, b: float) -> str:
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.3f}"


def _verdict(d: float) -> str:
    return "IMPROVED" if d > 0.005 else ("REGRESSED" if d < -0.005 else "flat")


def report(args: argparse.Namespace) -> None:
    print("=" * 92)
    print("  HYBRID SEARCH EVALUATION — AGGREGATE vs SEGMENTED")
    print("=" * 92)
    print()

    print("  PER-CLASS nDCG@10")
    print("  " + "-" * 88)
    print(f"  {'Class':<24}{'Traffic':>9}{'ConvIdx':>9}{'Dense':>8}"
          f"{'Blanket':>9}{'Delta':>9}{'Routed':>9}{'Delta':>9}")
    print("  " + "-" * 88)
    for c in CLASSES:
        db = c.ndcg_blanket - c.ndcg_baseline
        dr = c.ndcg_routed - c.ndcg_baseline
        flag = "  <-- collapses" if db < -0.1 else ""
        print(f"  {c.name:<24}{c.traffic_share:>8.0%}{c.conversion_index:>8.1f}x"
              f"{c.ndcg_baseline:>8.2f}{c.ndcg_blanket:>9.2f}{db:>9.2f}"
              f"{c.ndcg_routed:>9.2f}{dr:>9.2f}{flag}")
    print()

    # --- the two aggregates ------------------------------------------------
    tb = weighted(CLASSES, "ndcg_baseline", by_value=False)
    tk = weighted(CLASSES, "ndcg_blanket", by_value=False)
    tr = weighted(CLASSES, "ndcg_routed", by_value=False)

    rb = weighted(CLASSES, "ndcg_baseline", by_value=True)
    rk = weighted(CLASSES, "ndcg_blanket", by_value=True)
    rr = weighted(CLASSES, "ndcg_routed", by_value=True)

    print("  " + "-" * 88)
    print("  AGGREGATES")
    print("  " + "-" * 88)
    print(f"  {'Weighting':<28}{'Dense':>10}{'Blanket':>11}{'Delta':>10}"
          f"{'Routed':>10}{'Delta':>10}")
    print(f"  {'traffic-weighted':<28}{tb:>10.3f}{tk:>11.3f}{_delta(tb, tk):>10}"
          f"{tr:>10.3f}{_delta(tb, tr):>10}")
    print(f"  {'revenue-weighted':<28}{rb:>10.3f}{rk:>11.3f}{_delta(rb, rk):>10}"
          f"{rr:>10.3f}{_delta(rb, rr):>10}")
    print()

    print("  " + "-" * 88)
    print("  WHAT EACH NUMBER SAYS")
    print("  " + "-" * 88)
    print(f"  Blanket hybrid, traffic-weighted : {_verdict(tk - tb):<10}"
          f"({_delta(tb, tk)})  <- the number that gets reported")
    print(f"  Blanket hybrid, revenue-weighted : {_verdict(rk - rb):<10}"
          f"({_delta(rb, rk)})  <- the number that matters")
    print(f"  Routed hybrid,  revenue-weighted : {_verdict(rr - rb):<10}"
          f"({_delta(rb, rr)})")
    print()

    # --- ship / no-ship ----------------------------------------------------
    print("=" * 92)
    print("  SHIP DECISION")
    print("=" * 92)

    regressions = [c for c in CLASSES if c.ndcg_blanket - c.ndcg_baseline < -0.02]
    if regressions:
        print("  BLANKET HYBRID: DO NOT SHIP")
        print()
        for c in regressions:
            lost = (c.ndcg_baseline - c.ndcg_blanket)
            print(f"    {c.name} regressed {lost:.2f} nDCG"
                  f" — {c.traffic_share:.0%} of traffic at {c.conversion_index:.1f}x"
                  " conversion")
        print()
        print(f"  Traffic-weighted nDCG improves by {_delta(tb, tk)}, which is why this")
        print("  ships in most organisations. Revenue-weighted nDCG moves by"
              f" {_delta(rb, rk)}.")
        print("  The mean rose because the large low-intent classes improved. The")
        print("  small high-value class collapsed and the average absorbed it.")
    else:
        print("  BLANKET HYBRID: no per-class regression detected.")

    print()
    print("  ROUTED HYBRID: SHIP")
    print(f"    every class improves or holds; revenue-weighted {_delta(rb, rr)}")
    print("    EXACT_IDENTIFIER short-circuits to a btree lookup instead of being")
    print("    fused, so a perfect match stays at rank 1 (0.94 -> 0.97).")
    print()
    print("  RULE: never evaluate retrieval on an aggregate alone. Segment by query")
    print("  class and weight by business value per class, not by traffic share.")
    print("  A change that lifts the mean while regressing your highest-intent")
    print("  segment is a bad change wearing a good metric.")
    print("=" * 92)

    if args.show_ndcg_derivation:
        _derivation()


def _derivation() -> None:
    print()
    print("=" * 92)
    print("  APPENDIX — WHY EXACT_IDENTIFIER COLLAPSES UNDER BLANKET FUSION")
    print("=" * 92)
    print("""
  Query: "SKU-88213". There is exactly one relevant document.

  Lexical arm  : SKU-88213 at rank 1  (exact token match)
  Dense arm    : SKU-88214 at rank 1, SKU-88213 at rank 3
                 (the embedding cannot distinguish adjacent identifier strings)

  Blanket RRF with equal weights, k=60:
      SKU-88213 : 0.5/(60+1) + 0.5/(60+3)  = 0.008197 + 0.007937 = 0.016134
      SKU-88214 : 0.0        + 0.5/(60+1)  =            0.008197 = 0.008197

  The right answer still wins here — but only just, and only because the
  lexical arm found it at rank 1. Add one more near-miss SKU that both arms
  rank highly, or let the lexical arm miss because tsvector split the token,
  and rank 1 flips. nDCG@10 for this class falls from ~0.94 to ~0.71.

  Routed: the classifier recognises the identifier pattern, does a btree
  lookup on sku_norm, and returns the row. No fusion, no dilution, no risk.
""")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show-ndcg-derivation", action="store_true",
                   help="print the worked RRF example for the identifier class")
    report(p.parse_args())


if __name__ == "__main__":
    main()
