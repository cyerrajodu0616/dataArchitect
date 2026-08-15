"""
Week 4, Day 5 — Sizing Agent Checkpoints, and the Quadratic Nobody Expects
==========================================================================

A checkpoint is a saved copy of the agent's state, written after every node so
the run can survive a crash or a three-day pause. Day 2 and Day 3 both landed
on the same requirement; this is what it costs.

The naive estimate is:

    checkpoints = runs x steps
    storage     = runs x steps x state_size

That is wrong in a way that matters, because STATE GROWS AS THE RUN PROCEEDS.
The message history accumulates. Tool outputs pile up. Checkpoint 12 is much
bigger than checkpoint 1.

If you write the WHOLE state at every step and the state grows linearly, total
bytes per run is 1 + 2 + 3 + ... + N, which is:

    N(N+1)/2   ->   QUADRATIC in step count, not linear

A 20-step agent does not store twice what a 10-step agent stores. It stores
about four times as much. That is the difference between a storage line item
and a storage problem, and it is invisible in the naive formula.

WHAT THIS SCRIPT COMPUTES

  1. Checkpoint write rate, and how it compares to the OLTP write load the
     same database is already carrying.
  2. Storage under the naive linear model vs the real quadratic one.
  3. What the three fixes are worth: retention, delta checkpoints, and
     trimming what goes in the state at all.
  4. Whether Postgres is the right home, using the same reasoning as ADR-002 —
     and when it stops being.

stdlib only.

Usage:
    python checkpoint_sizing.py
    python checkpoint_sizing.py --domain insurance
    python checkpoint_sizing.py --steps 20 --growth-kb 6
    python checkpoint_sizing.py --retain-days 7
"""

from __future__ import annotations

import argparse

PG_GB_MONTH = 0.10              # gp3-class storage
PG_RAM_GB_MONTH = 11.0
REDIS_GB_MONTH = 45.0           # managed, in-memory
SECONDS_PER_MONTH = 2_592_000


DOMAIN_PRESETS = {
    "retail": dict(
        steps=6, runs_per_month=1_300_000, base_kb=4.0, growth_kb=3.0,
        retain_days=3, oltp_writes_per_sec=2_400,
        note="Short runs, enormous volume. Checkpoints are a WRITE RATE problem "
             "here, not a storage one — and the write rate lands on the same "
             "Postgres instance that is taking checkout traffic. Retention can "
             "be brutal because a finished conversation is worth nothing after "
             "the customer leaves."),
    "insurance": dict(
        steps=14, runs_per_month=2_400, base_kb=25.0, growth_kb=18.0,
        retain_days=2555, oltp_writes_per_sec=40,
        note="The opposite shape: tiny volume, long runs, big state, and a "
             "SEVEN-YEAR retention requirement because the checkpoint history "
             "is the audit trail of how an underwriting decision was reached. "
             "You cannot delete these, so the quadratic growth is the whole "
             "problem and delta checkpointing is not optional."),
    "fintech": dict(
        steps=9, runs_per_month=45_000, base_kb=12.0, growth_kb=7.0,
        retain_days=180, oltp_writes_per_sec=4_000,
        note="Middle volume, middle state, and a retention period set by "
             "dispute-window rules rather than by engineering. Note the OLTP "
             "write rate: 4,000/s of transaction traffic on the same instance "
             "means checkpoint writes need to be measured against it before "
             "anyone assumes they are free."),
}


def storage_per_run(steps: int, base_kb: float, growth_kb: float,
                    delta: bool) -> float:
    """
    Bytes written per run, in KB.

    FULL checkpoints: state at step i is base + i*growth, and you write all of
    it. Sum over i = 1..N gives base*N + growth*N(N+1)/2 — quadratic.

    DELTA checkpoints: you write only what changed, which is roughly `growth`
    each step plus the initial state. Linear.
    """
    if delta:
        return base_kb + growth_kb * steps
    return base_kb * steps + growth_kb * steps * (steps + 1) / 2


def report(a, note):
    print("=" * 92)
    print("  AGENT CHECKPOINT SIZING")
    print("=" * 92)
    if a.domain:
        print(f"  Domain            : {a.domain.upper()}")
    print(f"  Steps per run     : {a.steps}")
    print(f"  Runs per month    : {a.runs_per_month:,}")
    print(f"  State at step 1   : {a.base_kb:.0f} KB")
    print(f"  Growth per step   : +{a.growth_kb:.0f} KB"
          f"   (state at the last step: {a.base_kb + a.growth_kb * a.steps:,.0f} KB)")
    print(f"  Retention         : {a.retain_days:,} days")
    print()

    # ---- 1. write rate -----------------------------------------------------
    writes_per_month = a.runs_per_month * a.steps
    writes_per_sec = writes_per_month / SECONDS_PER_MONTH
    print("-" * 92)
    print("  1. WRITE RATE — MEASURED AGAINST WHAT THE DATABASE ALREADY DOES")
    print("-" * 92)
    print(f"  Checkpoint writes : {writes_per_month:,.0f}/month"
          f"  = {writes_per_sec:,.1f}/sec average")
    print(f"  Peak (4x average) : {writes_per_sec * 4:,.1f}/sec")
    print(f"  Existing OLTP     : {a.oltp_writes_per_sec:,}/sec")
    share = writes_per_sec * 4 / max(a.oltp_writes_per_sec, 1)
    print(f"  Checkpoints add   : {share:.1%} of current write load")
    print()
    if share < 0.05:
        print("  That is noise. Checkpointing into the existing Postgres instance")
        print("  is free at this rate, and the ADR-002 argument (one system, one")
        print("  backup, one failure domain) carries with nothing to trade off.")
    elif share < 0.30:
        print("  Noticeable but affordable. Worth watching, not worth a separate")
        print("  system. Check WAL volume and replication lag after rollout.")
    else:
        print("  SIGNIFICANT. Checkpoints would materially change the write")
        print("  profile of an instance that is also serving OLTP. Either put")
        print("  them on their own instance or move hot state to Redis with")
        print("  Postgres as the durable tier.")
    print()

    # ---- 2. the quadratic --------------------------------------------------
    naive_kb = a.runs_per_month * a.steps * a.base_kb
    real_kb = a.runs_per_month * storage_per_run(a.steps, a.base_kb,
                                                 a.growth_kb, delta=False)
    delta_kb = a.runs_per_month * storage_per_run(a.steps, a.base_kb,
                                                  a.growth_kb, delta=True)

    print("-" * 92)
    print("  2. STORAGE PER MONTH — WHY THE NAIVE ESTIMATE IS WRONG")
    print("-" * 92)
    print(f"  {'model':<44}{'GB/month':>14}{'vs naive':>14}")
    print("  " + "-" * 88)
    print(f"  {'naive: runs x steps x initial state':<44}{naive_kb / 1e6:>14,.1f}"
          f"{'1.0x':>14}")
    print(f"  {'real: full checkpoints, state grows':<44}{real_kb / 1e6:>14,.1f}"
          f"{real_kb / max(naive_kb, 1):>13.1f}x")
    print(f"  {'with delta checkpoints':<44}{delta_kb / 1e6:>14,.1f}"
          f"{delta_kb / max(naive_kb, 1):>13.1f}x")
    print()
    print(f"  The naive number is off by {real_kb / max(naive_kb, 1):.1f}x, and the")
    print(f"  error grows with step count because the full-checkpoint model is")
    print(f"  QUADRATIC: sum of 1..N is N(N+1)/2.")
    print()
    print(f"  {'steps':>8}{'GB/mo (full)':>16}{'GB/mo (delta)':>16}{'ratio':>10}")
    print("  " + "-" * 88)
    for n in [4, 6, 10, 14, 20, 30]:
        f_kb = a.runs_per_month * storage_per_run(n, a.base_kb, a.growth_kb, False)
        d_kb = a.runs_per_month * storage_per_run(n, a.base_kb, a.growth_kb, True)
        mark = "  <-- yours" if n == a.steps else ""
        print(f"  {n:>8}{f_kb / 1e6:>16,.1f}{d_kb / 1e6:>16,.1f}"
              f"{f_kb / max(d_kb, 1e-9):>9.1f}x{mark}")
    print()
    print("  Doubling the step count roughly QUADRUPLES full-checkpoint storage.")
    print("  A 20-step agent does not store twice a 10-step agent; it stores")
    print("  about four times as much. That is the line item that surprises")
    print("  people six months in.")
    print()

    # ---- 3. the three fixes ------------------------------------------------
    retained_months = a.retain_days / 30.0
    print("=" * 92)
    print("  3. THE THREE LEVERS, PRICED")
    print("=" * 92)
    steady_full = real_kb / 1e6 * retained_months
    steady_delta = delta_kb / 1e6 * retained_months
    print(f"  Steady-state storage at {a.retain_days:,} days retention:")
    print(f"    full checkpoints  : {steady_full:>12,.1f} GB"
          f"   ${steady_full * PG_GB_MONTH:>10,.0f}/month")
    print(f"    delta checkpoints : {steady_delta:>12,.1f} GB"
          f"   ${steady_delta * PG_GB_MONTH:>10,.0f}/month")
    print()
    print(f"  {'lever':<38}{'GB':>14}{'$/month':>13}   (and what it costs you)")
    print("  " + "-" * 88)

    levers = [
        ("do nothing", steady_full, "the baseline"),
        ("delta checkpoints", steady_delta,
         "you can no longer read one checkpoint standalone; "
         "you replay from the last full one"),
        ("trim state (drop raw tool output)", steady_full * 0.35,
         "the biggest single win, and it is free — most state is tool "
         "responses nobody re-reads"),
        ("retain 7 days instead", real_kb / 1e6 * (7 / 30.0),
         "only legal if nothing needs the history"),
    ]
    for name, gb, cost in levers:
        print(f"  {name:<38}{gb:>14,.1f}{gb * PG_GB_MONTH:>13,.0f}")
        line = ""
        for w in cost.split():
            if len(line) + len(w) + 1 > 70:
                print(f"      {line}"); line = w
            else:
                line = f"{line} {w}".strip()
        print(f"      {line}")
    print()
    print("  Take the levers in that order. TRIMMING THE STATE is first because")
    print("  it is free and it shrinks everything downstream — write rate, WAL,")
    print("  replication, backup, and the prompt if any of that state is being")
    print("  passed to the model. Delta checkpointing is second. Retention is")
    print("  last because it is usually not an engineering decision at all.")
    print()

    if a.retain_days > 365:
        print(f"  NOTE: {a.retain_days:,} days is a COMPLIANCE retention, not an")
        print(f"  engineering choice. You cannot delete your way out of this one,")
        print(f"  which is exactly why the quadratic matters here more than")
        print(f"  anywhere else — delta checkpointing is not an optimisation, it")
        print(f"  is the difference between {steady_full:,.0f} GB and"
              f" {steady_delta:,.0f} GB you must keep for seven years.")
        print()

    # ---- 4. where it lives -------------------------------------------------
    print("=" * 92)
    print("  4. DOES IT BELONG IN POSTGRES?")
    print("=" * 92)
    hot_gb = delta_kb / 1e6 * (1 / 30.0)    # roughly a day of in-flight state
    print(f"  {'option':<40}{'$/month':>13}   note")
    print("  " + "-" * 88)
    print(f"  {'Postgres (the ADR-002 instance)':<40}"
          f"{steady_delta * PG_GB_MONTH:>13,.0f}   already there, transactional")
    print(f"  {'Redis hot + Postgres durable':<40}"
          f"{hot_gb * REDIS_GB_MONTH + steady_delta * PG_GB_MONTH:>13,.0f}"
          f"   two systems, two failure modes")
    print()
    for line in [
        "The ADR-002 argument applies again, and more strongly:",
        "",
        "  ONE SYSTEM. The checkpoint and the business record it refers to live",
        "  in the same database, so 'refund issued' and 'agent state says refund",
        "  issued' can be written in ONE TRANSACTION. Split them and you have",
        "  re-created the dual-write problem from Week 2 Day 2 — with the same",
        "  drift, and now the drift is between an action and the record of it.",
        "",
        "  That transactional point is the strongest argument in this lesson and",
        "  it is usually left out of the Redis-vs-Postgres discussion entirely.",
        "",
        "  ONE BACKUP, ONE RESTORE. A restore that recovers the business data",
        "  but not the agent state leaves in-flight runs pointing at rows that",
        "  moved. For insurance, where the checkpoint history IS the audit",
        "  trail, they are the same artifact and must restore together.",
        "",
        "WHEN IT STOPS BEING POSTGRES:",
        "",
        "  - checkpoint writes are a large share of a shared OLTP instance's",
        "    write load (see section 1 for yours)",
        "  - state is large AND hot AND read every step, where Redis latency",
        "    genuinely pays -- then keep Postgres as the durable tier and treat",
        "    Redis as a cache you can lose",
        "  - runs are so short that nothing needs to survive at all, in which",
        "    case do not checkpoint. An in-memory saver is a legitimate choice",
        "    and nobody suggests it.",
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
    print()
    print("=" * 92)
    print("  THE SCHEMA")
    print("=" * 92)
    for line in [
        "CREATE TABLE agent_checkpoint (",
        "  thread_id   uuid    NOT NULL,        -- the run. Day 3's resume key.",
        "  step        int     NOT NULL,",
        "  node        text    NOT NULL,        -- where to resume",
        "  state       jsonb   NOT NULL,        -- full, or a delta",
        "  is_full     boolean NOT NULL,        -- delta chains need an anchor",
        "  created_at  timestamptz NOT NULL DEFAULT now(),",
        "  PRIMARY KEY (thread_id, step)",
        ");",
        "",
        "-- resume = SELECT ... ORDER BY step DESC LIMIT 1. That is the whole",
        "-- 'magic' of a checkpointer.",
        "CREATE INDEX ON agent_checkpoint (created_at);   -- for retention sweeps",
        "",
        "Partition by created_at if retention is short: dropping a partition is",
        "instant, while DELETE at this row count generates WAL, bloats the table",
        "and gives the autovacuum work you did not budget for.",
    ]:
        print(f"  {line}")
    print("=" * 92)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=sorted(DOMAIN_PRESETS))
    p.add_argument("--steps", type=int)
    p.add_argument("--runs-per-month", type=int)
    p.add_argument("--base-kb", type=float)
    p.add_argument("--growth-kb", type=float)
    p.add_argument("--retain-days", type=int)
    p.add_argument("--oltp-writes-per-sec", type=int)
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
