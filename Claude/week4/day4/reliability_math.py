"""
Week 4, Day 4 — Why Reliable Steps Make an Unreliable Agent
============================================================

Start with the arithmetic that surprises people, because everything else in
this file follows from it.

    a node that works 99% of the time is a GOOD node
    six of them in a row work 0.99^6 = 94.1% of the time
    twelve of them work 88.6%

You did not do anything wrong. Reliability MULTIPLIES along a path, so adding
steps makes an agent less reliable even when every step is excellent. An agent
with twenty tool calls and a 99% per-call success rate fails one run in five.

That is the problem. Retries are the obvious fix, and they introduce a worse
problem, which is most of this script.

THE THREE KINDS OF FAILURE, AND WHY ONLY ONE IS EASY

  TRANSIENT   the network blipped, the API returned 503, you were rate limited.
              Retry. It will probably work.

  PERMANENT   the input is invalid, the policy does not exist, you lack
              permission. Retrying is pure cost — it will fail identically
              every time, and each attempt is billed.

  AMBIGUOUS   you sent the request and never saw the response. Did the refund
              go out or not? THIS is the hard one, and it is the one that turns
              a retry into a second refund.

WHAT THIS SCRIPT COMPUTES

  1. End-to-end success vs step count, and what retries buy back.
  2. The cost of retrying the wrong things — a permanent failure retried 4
     times is 5 paid attempts and zero chance of success.
  3. An IDEMPOTENCY simulation: the same workload with and without keys,
     counting duplicate side effects. This is the number that matters, because
     a duplicate refund is not a slower success, it is money gone.
  4. Retry storms: why fixed-delay retries synchronise and jitter fixes it.

stdlib only.

Usage:
    python reliability_math.py
    python reliability_math.py --domain insurance
    python reliability_math.py --steps 12 --step-success 0.985
    python reliability_math.py --no-idempotency
"""

from __future__ import annotations

import argparse
import random

LLM_CALL_COST = 0.004          # a mid-size call with tools, in dollars
SECONDS_PER_ATTEMPT = 1.8


DOMAIN_PRESETS = {
    "retail": dict(
        steps=6, step_success=0.99, runs_per_month=1_300_000,
        side_effect_value=42.0, transient_share=0.75,
        note="Short graphs, enormous volume. At 6 steps and 99% per step you "
             "fail 5.9% of runs — that is 76,000 failed conversations a month "
             "before you retry anything. Volume is what makes a small "
             "per-step failure rate a staffing problem."),
    "insurance": dict(
        steps=14, step_success=0.985, runs_per_month=2_400,
        side_effect_value=18_000.0, transient_share=0.55,
        note="Long graphs, tiny volume, huge side effects. 14 steps at 98.5% "
             "is 81% end-to-end — one run in five needs intervention. And a "
             "duplicate side effect here is a policy bound twice, which is a "
             "rescission and a regulatory conversation, not a refund."),
    "fintech": dict(
        steps=9, step_success=0.992, runs_per_month=45_000,
        side_effect_value=260.0, transient_share=0.70,
        note="Middle everything, but with a hard deadline: a run that fails "
             "and needs manual restart may miss a network window. Here the "
             "cost of a failed run is not the retry, it is the automatic loss "
             "when the clock runs out."),
}


def end_to_end(steps: int, p: float, retries: int, transient_share: float) -> float:
    """
    Probability the whole graph completes.

    With `retries` attempts, a step fails permanently only if every attempt
    fails. Retries only help TRANSIENT failures — a permanent failure fails
    identically every time, which is why transient_share is in the formula and
    why it caps what retrying can buy you.
    """
    fail = 1 - p
    perm = fail * (1 - transient_share)
    trans = fail * transient_share
    step_fail = perm + trans ** (retries + 1)
    return (1 - step_fail) ** steps


def report(a, note):
    print("=" * 92)
    print("  RELIABILITY MULTIPLIES. THAT IS THE WHOLE PROBLEM.")
    print("=" * 92)
    if a.domain:
        print(f"  Domain           : {a.domain.upper()}")
    print(f"  Steps in the graph: {a.steps}")
    print(f"  Per-step success  : {a.step_success:.1%}")
    print(f"  Of failures       : {a.transient_share:.0%} transient (retryable),"
          f" {1 - a.transient_share:.0%} permanent")
    print(f"  Runs / month      : {a.runs_per_month:,}")
    print()

    print("-" * 92)
    print("  1. WHAT STEP COUNT COSTS YOU, BEFORE ANY RETRIES")
    print("-" * 92)
    print(f"  {'steps':>7}{'end-to-end':>13}{'failed runs/mo':>18}")
    print("  " + "-" * 88)
    for n in [1, 3, 6, 9, 12, 20, 30]:
        e = a.step_success ** n
        mark = "  <-- yours" if n == a.steps else ""
        print(f"  {n:>7}{e:>13.1%}{a.runs_per_month * (1 - e):>18,.0f}{mark}")
    print()
    print(f"  Every step you add is a multiplication, not an addition. This is the")
    print(f"  argument for FEWER, BIGGER nodes — and it runs directly against the")
    print(f"  instinct to decompose an agent into many small clean steps.")
    print()

    print("-" * 92)
    print("  2. WHAT RETRIES BUY BACK — AND WHAT THEY CANNOT")
    print("-" * 92)
    print(f"  {'retries':>9}{'end-to-end':>13}{'failed/mo':>13}"
          f"{'extra calls/mo':>17}{'extra $/mo':>13}")
    print("  " + "-" * 88)
    base = None
    for r in range(0, 5):
        e = end_to_end(a.steps, a.step_success, r, a.transient_share)
        if base is None:
            base = e
        # Expected extra attempts: each step that fails gets retried.
        extra_per_run = a.steps * (1 - a.step_success) * min(r, 3)
        extra_calls = a.runs_per_month * extra_per_run
        print(f"  {r:>9}{e:>13.1%}{a.runs_per_month * (1 - e):>13,.0f}"
              f"{extra_calls:>17,.0f}{extra_calls * LLM_CALL_COST:>13,.0f}")
    ceiling = end_to_end(a.steps, a.step_success, 50, a.transient_share)
    print()
    print(f"  CEILING: even with infinite retries you reach {ceiling:.1%}, because")
    print(f"  {1 - a.transient_share:.0%} of failures are PERMANENT and retrying them just")
    print(f"  buys the same error again at full price.")
    print()
    print(f"  So the first question is not 'how many retries' — it is 'can I tell")
    print(f"  transient from permanent?' A retry policy that cannot distinguish")
    print(f"  them is paying {1 - a.transient_share:.0%} of its retry budget for nothing.")
    print()

    # ---- idempotency -------------------------------------------------------
    print("=" * 92)
    print("  3. THE AMBIGUOUS FAILURE — AND WHY IT IS NOT A RELIABILITY PROBLEM")
    print("=" * 92)
    print("  You call the refund API. The connection drops before the response.")
    print("  Did the refund go out?")
    print()
    print("  Retry and you might issue it twice. Do not retry and you might have")
    print("  issued nothing. Neither choice is safe without an idempotency protocol.")
    print("  Safety requires a stable logical key plus receiver-side unique enforcement")
    print("  and atomic storage of the request fingerprint, side effect, and result.")
    print()

    rng = random.Random(a.seed)
    AMBIGUOUS_SHARE = 0.18       # of transient failures, this share are ambiguous
    trials = 200_000
    dupes = {True: 0, False: 0}
    for use_keys in (False, True):
        d = 0
        for _ in range(trials):
            if rng.random() < (1 - a.step_success) * a.transient_share * AMBIGUOUS_SHARE:
                # The call actually succeeded; we just never saw the response.
                # Retrying repeats the side effect UNLESS a key dedupes it.
                if not use_keys:
                    d += 1
        dupes[use_keys] = d

    dup_rate = dupes[False] / trials
    dup_per_month = a.runs_per_month * dup_rate
    print(f"  {'':<28}{'duplicate side effects':>26}{'$/month':>14}")
    print("  " + "-" * 88)
    print(f"  {'without idempotency keys':<28}{dup_per_month:>26,.0f}"
          f"{dup_per_month * a.side_effect_value:>14,.0f}")
    print(f"  {'with enforced idempotency':<28}{0:>26,.0f}{0:>14,.0f}")
    print()
    print(f"  {dup_rate:.3%} of runs hit an ambiguous failure. That sounds small.")
    print(f"  At {a.runs_per_month:,} runs a month and ${a.side_effect_value:,.0f} per")
    print(f"  side effect it is ${dup_per_month * a.side_effect_value:,.0f} a month.")
    print(f"  A UUID header carries the key; receiver-side atomic enforcement is the fix.")
    print()
    for line in [
        "Say this precisely, because it is the distinction that matters:",
        "",
        "  a duplicate side effect is NOT a reliability problem.",
        "  it is a CORRECTNESS problem created by the reliability fix.",
        "",
        "Retries make availability better and correctness worse, and an enforced",
        "idempotency protocol lets you have both. Which is why 'add retries' is not a",
        "complete answer to 'how do you handle failures' — the complete answer",
        "is 'retries plus idempotency keys, and here is which operations need",
        "them'.",
    ]:
        print(f"  {line}")
    print()

    # ---- retry storms ------------------------------------------------------
    print("=" * 92)
    print("  4. RETRY STORMS — WHY FIXED DELAYS ARE WORSE THAN NO RETRIES")
    print("=" * 92)
    concurrent = max(50, int(a.runs_per_month / 30 / 24 / 60))
    print(f"  Suppose a dependency has a 5-second outage while ~{concurrent:,} runs")
    print(f"  are in flight. All of them fail at once.")
    print()
    print(f"  {'strategy':<34}{'peak retry rate':>20}{'vs steady state':>18}")
    print("  " + "-" * 88)
    steady = concurrent / 60.0
    for name, peak in [
        ("fixed 1s delay", concurrent / 1.0),
        ("exponential backoff, no jitter", concurrent / 1.0),
        ("exponential + full jitter", concurrent / 30.0),
    ]:
        print(f"  {name:<34}{peak:>17,.0f}/s{peak / max(steady, 1e-9):>17.0f}x")
    print()
    for line in [
        "Exponential backoff ALONE does not help here. Every client failed at",
        "the same instant, so every client waits the same 1s, then the same 2s,",
        "then the same 4s — they stay synchronised and hit the recovering",
        "dependency together, which knocks it over again.",
        "",
        "Jitter is what breaks the synchronisation. It is one line",
        "(sleep(random.uniform(0, backoff)) instead of sleep(backoff)) and it is",
        "the difference between a dependency that recovers and one that",
        "oscillates for an hour.",
        "",
        "And the backstop above both: a CIRCUIT BREAKER. After N consecutive",
        "failures, stop calling entirely for a cooling-off period. Retries",
        "assume the dependency is basically healthy; a breaker is what you have",
        "for when that assumption is false.",
    ]:
        print(f"  {line}")
    print()

    print("=" * 92)
    print("  WHAT TO ACTUALLY BUILD, IN ORDER")
    print("=" * 92)
    for line in [
        "1. FEWER NODES. Reliability multiplies, so the cheapest reliability",
        "   work is deleting a step. Do this before tuning retry policy.",
        "",
        "2. CLASSIFY THE FAILURE. Transient, permanent, ambiguous. A retry",
        f"   policy that cannot tell them apart wastes {1 - a.transient_share:.0%} of its budget",
        "   and cannot be made safe.",
        "",
        "3. IDEMPOTENCY KEYS on every operation with a side effect. Before",
        "   retries, not after — retries without keys make things worse.",
        "",
        "4. EXPONENTIAL BACKOFF WITH JITTER. Backoff without jitter keeps",
        "   clients synchronised and is barely better than a fixed delay.",
        "",
        "5. A CIRCUIT BREAKER for when the dependency is genuinely down, and",
        "   for the correlated-failure case Day 3 could not price: an agent",
        "   that is wrong the same way 4,000 times is one incident, not 4,000",
        "   independent losses, and a breaker is what stops it at 40.",
        "",
        "6. A DEAD LETTER PATH. Some runs will not succeed. They need somewhere",
        "   to go that a human looks at, with the state attached so the run can",
        "   be resumed rather than restarted.",
    ]:
        print(f"  {line}")

    if note:
        print()
        print("  DOMAIN NOTE")
        line = ""
        for w in note.split():
            if len(line) + len(w) + 1 > 84:
                print(f"    {line}"); line = w
            else:
                line = f"{line} {w}".strip()
        print(f"    {line}")
    print("=" * 92)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--steps", type=int)
    p.add_argument("--step-success", type=float)
    p.add_argument("--runs-per-month", type=int)
    p.add_argument("--side-effect-value", type=float)
    p.add_argument("--transient-share", type=float)
    p.add_argument("--seed", type=int, default=7)
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
