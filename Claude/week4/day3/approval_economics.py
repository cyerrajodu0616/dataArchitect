"""
Week 4, Day 3 — Where To Put The Approval Gate, Derived
========================================================

"Add a human approval step" is easy to say and almost always specified by
intuition: someone picks a round number, usually $500, and it never gets
revisited.

It is an arithmetic problem, and the arithmetic has a surprise in it.

THE OBVIOUS HALF

  Reviewing an action costs money  : a person's minutes.
  Not reviewing costs money        : P(the agent is wrong) x what wrong costs.

  So review an action when   P(wrong) x cost_if_wrong  >  cost_to_review.
  Everything below that line, let the agent act alone.

THE HALF THAT CHANGES THE ANSWER

  Reviewer attention is FINITE. Send a human 40 approvals an hour and they stop
  reading them — this is well documented in every domain that has ever built an
  alert queue, and it is why "review everything" is not the safe option people
  assume.

  So the catch rate is not a constant. It falls as the queue grows:

      catch_rate = base_rate x decay(items per hour / comfortable capacity)

  Which means total expected loss is U-SHAPED in the threshold:

      threshold too HIGH  ->  bad actions go through unreviewed
      threshold too LOW   ->  the queue floods, reviewers rubber-stamp, and you
                              pay full price for a review that caught nothing

  There is a minimum in the middle, and it is usually nowhere near $500.

WHAT THIS SCRIPT DOES

  1. Sweeps the approval threshold and prices each one: review cost, expected
     loss from missed errors, and the total.
  2. Finds the minimum, and reports the naive answer next to it.
  3. Shows what "review everything" actually costs, including the
     rubber-stamping penalty.
  4. Prices the LATENCY of a pause separately, because a queue that takes two
     days to clear is a product decision, not just a cost.

stdlib only.

Usage:
    python approval_economics.py
    python approval_economics.py --domain insurance
    python approval_economics.py --error-rate 0.05 --review-minutes 12
    python approval_economics.py --no-fatigue        # the naive model, for contrast
"""

from __future__ import annotations

import argparse
import math

REVIEWER_HOURLY = 45.0          # loaded cost of the person doing the reviewing
WORK_HOURS_PER_MONTH = 160.0


DOMAIN_PRESETS = {
    "retail": dict(
        actions_per_month=120_000, review_minutes=3.0, error_rate=0.04,
        value_mean=42.0, value_spread=2.2, reviewers=2,
        note="Refund approvals. Huge volume, small values, cheap mistakes. The "
             "arithmetic says review almost nothing — and that is the correct, "
             "unintuitive answer. A $42 refund is not worth three minutes of a "
             "person's time at a 4% error rate."),
    "insurance": dict(
        actions_per_month=2_400, review_minutes=25.0, error_rate=0.07,
        value_mean=18_000.0, value_spread=1.6, reviewers=3,
        note="Underwriting decisions. Low volume, high value, and a regulatory "
             "floor UNDER the economics: some decisions must be reviewed "
             "whether or not the maths says so. Note what that means — the "
             "threshold is the lower of the economic answer and the compliance "
             "answer, and here compliance binds first."),
    "fintech": dict(
        actions_per_month=45_000, review_minutes=8.0, error_rate=0.05,
        value_mean=260.0, value_spread=2.6, reviewers=4,
        note="Dispute resolutions. Middle volume, middle value, and a hard "
             "regulatory clock — a queue that takes 3 days to clear is not a "
             "cost problem, it is a missed deadline and an automatic loss."),
}


def lognormal_tail_fraction(threshold: float, mean: float, spread: float) -> float:
    """
    Fraction of actions whose value exceeds `threshold`.

    Action values are heavily right-skewed — many small, a few very large — so
    a lognormal is a much better default than a normal. Using a normal here
    would understate the tail, which is precisely where the money is.
    """
    if threshold <= 0:
        return 1.0
    mu = math.log(mean) - 0.5 * math.log(spread) ** 2
    sigma = math.log(spread)
    z = (math.log(threshold) - mu) / sigma
    return 0.5 * math.erfc(z / math.sqrt(2))


def mean_value_above(threshold: float, mean: float, spread: float) -> float:
    """Average value of actions above the threshold (they are bigger than mean)."""
    if threshold <= 0:
        return mean                      # reviewing everything: the plain mean
    frac = lognormal_tail_fraction(threshold, mean, spread)
    if frac < 1e-9:
        return threshold
    mu = math.log(mean) - 0.5 * math.log(spread) ** 2
    sigma = math.log(spread)
    z = (math.log(threshold) - mu) / sigma
    # E[X | X > t] for a lognormal
    upper = 0.5 * math.erfc((z - sigma) / math.sqrt(2))
    return mean * upper / frac


def catch_rate(items_per_hour: float, reviewers: int, fatigue: bool) -> float:
    """
    How much a reviewer actually catches, as a function of queue pressure.

    BASE_CATCH is what a careful reviewer catches with time to think. As the
    per-reviewer rate climbs past what one person can genuinely consider, catch
    rate decays. This is the rubber-stamping effect and it is the whole reason
    "review everything" is not free.
    """
    BASE_CATCH = 0.92
    COMFORTABLE_PER_HOUR = 6.0
    if not fatigue:
        return BASE_CATCH
    load = items_per_hour / max(1, reviewers) / COMFORTABLE_PER_HOUR
    return BASE_CATCH * math.exp(-0.55 * max(0.0, load - 1.0))


def evaluate(threshold: float, a) -> dict:
    frac = lognormal_tail_fraction(threshold, a.value_mean, a.value_spread)
    reviewed = a.actions_per_month * frac
    avg_val_reviewed = mean_value_above(threshold, a.value_mean, a.value_spread)

    # THE HARD CONSTRAINT. You cannot review more than the team can physically
    # get through. Anything routed to review beyond capacity is not "reviewed
    # late". This model prices it as unreviewed, matching a deliberately unsafe
    # fail-open timeout policy. Production should fail closed: expire, reject,
    # or escalate it. Either way, the model must not let you buy review capacity
    # you have not staffed.
    capacity = a.reviewers * WORK_HOURS_PER_MONTH * (60.0 / a.review_minutes)
    routed = reviewed
    actually_reviewed = min(routed, capacity)
    overflow = routed - actually_reviewed

    per_hour = actually_reviewed / WORK_HOURS_PER_MONTH
    rate = catch_rate(per_hour, a.reviewers, not a.no_fatigue)

    # You only pay for reviews that happen.
    review_cost = actually_reviewed * (a.review_minutes / 60.0) * REVIEWER_HOURLY

    # Losses come from three places now.
    #   below the threshold : never routed, every error costs full value
    #   overflow            : routed but never actually seen — same as unreviewed
    #   actually reviewed   : only `rate` of errors are caught
    unrouted = a.actions_per_month - routed
    avg_val_below = (a.value_mean * a.actions_per_month
                     - avg_val_reviewed * routed) / max(unrouted, 1e-9)
    loss_below = unrouted * a.error_rate * max(0.0, avg_val_below)
    loss_overflow = overflow * a.error_rate * avg_val_reviewed
    loss_above = actually_reviewed * a.error_rate * avg_val_reviewed * (1 - rate)

    backlog_days = (routed / capacity) * 30.0 if capacity else 999.0

    return dict(
        threshold=threshold, reviewed=actually_reviewed, routed=routed,
        overflow=overflow, frac=frac, capacity=capacity,
        review_cost=review_cost,
        loss_below=loss_below + loss_overflow, loss_above=loss_above,
        total=review_cost + loss_below + loss_overflow + loss_above,
        catch=rate, per_hour=per_hour, backlog_days=backlog_days,
        avg_val_reviewed=avg_val_reviewed,
    )


def report(a, note):
    print("=" * 92)
    print("  WHERE TO PUT THE APPROVAL GATE")
    print("=" * 92)
    if a.domain:
        print(f"  Domain          : {a.domain.upper()}")
    print(f"  Actions / month : {a.actions_per_month:,}")
    print(f"  Agent error rate: {a.error_rate:.1%}")
    print(f"  Review takes    : {a.review_minutes:.0f} min"
          f"  (${a.review_minutes / 60 * REVIEWER_HOURLY:.2f} of a reviewer's time)")
    print(f"  Reviewers       : {a.reviewers}")
    print(f"  Action value    : ${a.value_mean:,.0f} mean, lognormal"
          f" (spread {a.value_spread})")
    print(f"  Fatigue model   : {'OFF (naive)' if a.no_fatigue else 'ON'}")
    print()

    # The naive threshold: review when expected loss exceeds review cost.
    review_cost_each = a.review_minutes / 60.0 * REVIEWER_HOURLY
    naive = review_cost_each / a.error_rate
    print(f"  The textbook answer, ignoring reviewer capacity:")
    print(f"    review when  P(wrong) x value > cost to review")
    print(f"    ${review_cost_each:.2f} / {a.error_rate:.2f}"
          f" = review anything above ${naive:,.0f}")
    print()

    thresholds = [0.0]
    t = max(1.0, a.value_mean / 50)
    while t < a.value_mean * 60:
        thresholds.append(t)
        t *= 1.9

    print("-" * 92)
    print(f"  {'threshold':>12}{'routed':>10}{'reviewed':>10}{'/hour':>8}"
          f"{'catch':>7}{'review $':>11}{'missed $':>12}{'TOTAL $':>12}{'queue':>8}")
    print("  " + "-" * 88)
    rows = [evaluate(t, a) for t in thresholds]
    best = min(rows, key=lambda r: r["total"])
    for r in rows:
        flag = "  <-- best" if r is best else ""
        label = "review ALL" if r["threshold"] == 0 else f"${r['threshold']:,.0f}"
        over = "!" if r["overflow"] > 1 else " "
        print(f"  {label:>12}{r['routed']:>10,.0f}{r['reviewed']:>9,.0f}{over}"
              f"{r['per_hour']:>8.1f}{r['catch']:>7.0%}{r['review_cost']:>11,.0f}"
              f"{r['loss_below'] + r['loss_above']:>12,.0f}"
              f"{r['total']:>12,.0f}{r['backlog_days']:>7.1f}d{flag}")
    print()

    all_row = rows[0]
    none_row = evaluate(a.value_mean * 1e6, a)

    print("=" * 92)
    print("  RESULT")
    print("=" * 92)
    label = "review everything" if best["threshold"] == 0 else f"${best['threshold']:,.0f}"
    print(f"  Cheapest gate    : {label}")
    print(f"  Reviewed         : {best['reviewed']:,.0f}/month"
          f"  ({best['frac']:.1%} of actions)")
    print(f"  Catch rate there : {best['catch']:.0%}"
          f"   ({best['per_hour']:.1f} items/hour across {a.reviewers} reviewers)")
    print(f"  Total cost       : ${best['total']:,.0f}/month")
    print(f"  Queue clears in  : {best['backlog_days']:.1f} days")
    print()
    print(f"  vs review NOTHING    : ${none_row['total']:,.0f}/month"
          f"   ({none_row['total'] / max(best['total'], 1):.1f}x)")
    print(f"  vs review EVERYTHING : ${all_row['total']:,.0f}/month"
          f"   ({all_row['total'] / max(best['total'], 1):.1f}x)")
    print()

    if not a.no_fatigue and all_row["catch"] < 0.5:
        print(f"  READ THE 'review ALL' ROW AGAIN. Catch rate is"
              f" {all_row['catch']:.0%}.")
        print(f"  At {all_row['per_hour']:.0f} items/hour per reviewer nobody is")
        print(f"  reading them. You pay ${all_row['review_cost']:,.0f}/month for a")
        print(f"  control that stops {all_row['catch']:.0%} of what it looks at.")
        print()
        print("  This is the finding worth carrying: 'review everything' is not")
        print("  the safe default. It is an expensive way to CONVERT a real")
        print("  control into a theatrical one. Under the modeled fail-open policy,")
        print("  the unreviewed overflow can still look approved in a dashboard.")
        print()

    if best["routed"] > best["capacity"] * 0.9 and best["overflow"] < 1:
        print(f"  THE OPTIMUM IS SET BY STAFFING, NOT BY VALUE. At"
              f" ${best['threshold']:,.0f} the")
        print(f"  queue is {best['routed']:,.0f}/month against a capacity of"
              f" {best['capacity']:,.0f}. Lower the threshold")
        print(f"  by one step and the queue overflows — rows marked ! route more")
        print(f"  than the team can physically read. This model prices excess as")
        print(f"  fail-open to expose the loss; production should expire fail-closed.")
        print()
        print("  So the real question is not 'what is the right threshold'. It is")
        print("  'how many reviewers do we have', and the threshold is whatever")
        print("  keeps the queue inside that. Staffing IS the control.")
        print()

    if abs(naive - best["threshold"]) / max(naive, 1) > 0.3 and best["threshold"] > 0:
        direction = "LOWER" if naive < best["threshold"] else "HIGHER"
        print(f"  Note the textbook threshold (${naive:,.0f}) is {direction} than the")
        print(f"  real optimum (${best['threshold']:,.0f}) — it routes"
              f" {'MORE' if naive < best['threshold'] else 'LESS'} to review than")
        print(f"  the team can absorb. The gap is entirely reviewer capacity: the")
        print(f"  formula that ignores it is optimising for a reviewer who never")
        print(f"  gets tired and never has a queue.")
        print()

    if best["backlog_days"] > 2:
        print(f"  LATENCY WARNING: the queue takes {best['backlog_days']:.1f} days to")
        print(f"  clear. That is a product decision, not just a cost — the customer")
        print(f"  is waiting, and in fintech or claims a regulatory clock runs.")
        print(f"  Either staff up, raise the threshold, or make the pause async.")
        print()

    if note:
        print("  DOMAIN NOTE")
        line = ""
        for w in note.split():
            if len(line) + len(w) + 1 > 84:
                print(f"    {line}"); line = w
            else:
                line = f"{line} {w}".strip()
        print(f"    {line}")
        print()

    print("=" * 92)
    print("  WHAT THIS MODEL DOES NOT PRICE")
    print("=" * 92)
    for line in [
        "Be straight about the limits, because an interviewer will probe them.",
        "",
        "  REGULATORY FLOORS. Some decisions must be reviewed regardless of",
        "  expected value. The real threshold is the LOWER of the economic",
        "  answer and the compliance answer, and in insurance compliance",
        "  usually binds first. The economics then tell you what that",
        "  requirement costs — which is still worth knowing.",
        "",
        "  REPUTATION AND CORRELATED FAILURE. This prices one error at a time.",
        "  A bug that makes the agent wrong the same way 4,000 times is not",
        "  4,000 independent small losses; it is one incident. Circuit breakers",
        "  belong in Day 4, not in the threshold.",
        "",
        "  WHAT REVIEW IS FOR. Some approval gates exist to create an",
        "  accountable human, not to catch errors. That is a legitimate reason",
        "  and this model cannot see it. Say so rather than pretending the",
        "  maths settles it.",
    ]:
        print(f"  {line}")
    print("=" * 92)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--actions-per-month", type=int)
    p.add_argument("--review-minutes", type=float)
    p.add_argument("--error-rate", type=float)
    p.add_argument("--value-mean", type=float)
    p.add_argument("--value-spread", type=float)
    p.add_argument("--reviewers", type=int)
    p.add_argument("--no-fatigue", action="store_true",
                   help="assume reviewers never get tired — the naive model")
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
