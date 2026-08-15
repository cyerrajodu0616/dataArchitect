"""
Week 4, Day 2 — The Same Job, Four Ways, Then the Requirements Change
======================================================================

Framework comparisons are usually written as feature tables, which is the least
useful form because every framework can do everything if you try hard enough.
The question that actually decides it is:

    when the requirements change, which one needs an EDIT and which one needs
    a REWRITE?

So this script builds the same refund agent four ways, then applies four
requirement changes that really do land on teams, and reports the damage.

THE FOUR APPROACHES, IN PLAIN WORDS

  1. RAW           You call the model yourself in a loop. No framework.
                   Like writing SQL by hand instead of using an ORM.

  2. CHAIN         A fixed pipeline of steps. LangChain's classic form.
                   Like a shell script: step, step, step, done.

  3. GRAPH         A state machine. Nodes and edges, edges can go backwards.
                   LangGraph. Day 1 built one of these in 40 lines.

  4. CREW          Several specialised agents that talk to each other.
                   CrewAI, AutoGen. Like a team with job titles.

WHAT THIS SCRIPT IS NOT

It is not a benchmark. It does not call any model. It is a structural argument:
given how each approach represents control flow, which changes are cheap?
That is knowable from the shape alone, which is why it is worth reasoning about
before you have written anything.

stdlib only.

Usage:
    python framework_chooser.py
    python framework_chooser.py --show-code
    python framework_chooser.py --decide          # the chooser, on your answers
    python framework_chooser.py --decide --loops 2 --human-approval --long-running
"""

from __future__ import annotations

import argparse

# ============================================================================
# The four implementations, reduced to their control flow so the SHAPE is
# visible. Each "step" is a stand-in for an LLM call or a tool call.
# ============================================================================

CODE = {
    "raw": '''
# RAW — you own the loop
def handle(request):
    state = {"request": request}
    for _ in range(MAX_TURNS):                 # your loop, your rules
        action = call_model(state)             # ask the model what to do next
        if action.done:
            return action.answer
        state[action.tool] = run_tool(action)  # you dispatch the tool
    return escalate(state)
''',
    "chain": '''
# CHAIN — a fixed pipeline
pipeline = classify | lookup_order | check_policy | decide
result = pipeline.invoke({"request": request})
# there is no way to express "go back to classify"
''',
    "graph": '''
# GRAPH — nodes and edges, edges may point backwards
g.add_node("classify", classify)
g.add_node("ask_clarify", ask_clarify)
g.add_conditional_edges("classify", route)     # route() picks the next node
g.add_edge("ask_clarify", "classify")          # <-- backwards
g.add_edge("approve", END)
result = g.compile(checkpointer=pg).invoke(state, thread_id=case_id)
''',
    "crew": '''
# CREW — specialists with roles, talking to each other
classifier = Agent(role="Intent classifier", goal="...")
policy     = Agent(role="Policy expert",     goal="...")
crew = Crew(agents=[classifier, policy], process=Process.sequential)
result = crew.kickoff({"request": request})
# who talks to whom, and when it stops, is largely the framework's decision
''',
}

APPROACHES = ["raw", "chain", "graph", "crew"]

LABEL = {
    "raw": "RAW (you call the API)",
    "chain": "CHAIN (LangChain)",
    "graph": "GRAPH (LangGraph)",
    "crew": "CREW (CrewAI / AutoGen)",
}

# ============================================================================
# Four requirement changes that actually happen, and what each costs.
#   "edit"    = a small, local change
#   "bolt-on" = works, but you are now hand-rolling what a framework provides
#   "rewrite" = the shape of the code has to change
# ============================================================================

CHANGES = [
    dict(
        name="1. Ask a clarifying question, then continue",
        why="The first real requirement anyone hits. Needs a backward edge.",
        cost=dict(
            raw=("bolt-on", "Your loop already re-runs, so this is a branch "
                            "inside it. Cheap now; this is also the first step "
                            "toward writing your own graph engine."),
            chain=("rewrite", "A pipeline has no backward edge. You either wrap "
                              "the whole chain in a while loop — which IS a "
                              "graph, badly — or you change approach."),
            graph=("edit", "One node, one conditional edge. This is the case "
                           "the abstraction exists for."),
            crew=("edit", "Agents already converse; add a clarifying turn."),
        ),
    ),
    dict(
        name="2. Pause for human approval on refunds over $500",
        why="Not just 'ask a question' — STOP, possibly for hours, then resume "
            "at exactly the right point with the same state.",
        cost=dict(
            raw=("rewrite", "Your loop lives in a process. To pause for hours "
                            "you must serialise the state, exit, and rebuild "
                            "the loop position on resume. That is a "
                            "checkpointer, and you are now writing one."),
            chain=("rewrite", "Same problem, and no place to put the pause."),
            graph=("edit", "interrupt_before=['approve'] plus a checkpointer. "
                           "The framework already saves state per node."),
            crew=("bolt-on", "Human-in-the-loop exists but resumption is "
                             "weaker; check how the framework persists a "
                             "half-finished conversation before relying on it."),
        ),
    ),
    dict(
        name="3. Survive a crash mid-run and resume",
        why="A pod restarts. An underwriting case is 3 days into a 2-week "
            "process. Does the work survive?",
        cost=dict(
            raw=("rewrite", "Needs durable state after every step. Doable — "
                            "it is a table and a state column — but it is the "
                            "single biggest chunk of work on this list."),
            chain=("rewrite", "Nothing is persisted between steps."),
            graph=("edit", "Checkpointing is the core feature. Point it at "
                           "Postgres and resume by thread_id."),
            crew=("bolt-on", "Varies by framework and by version. Verify "
                             "rather than assume."),
        ),
    ),
    dict(
        name="4. Add a fraud-check specialist that runs in parallel",
        why="A second opinion on the same case, merged before deciding.",
        cost=dict(
            raw=("bolt-on", "Threads plus a merge. You now own the "
                            "concurrency and the merge semantics."),
            chain=("rewrite", "Sequential by construction."),
            graph=("edit", "Fan out to two nodes, fan in with a reducer. The "
                           "reducer is why the merge is deterministic."),
            crew=("edit", "This is what crews are FOR — the one row where "
                          "crew is the natural answer."),
        ),
    ),
]

SCORE = {"edit": 0, "bolt-on": 1, "rewrite": 3}


def show_code() -> None:
    print("=" * 84)
    print("  THE SAME JOB, FOUR SHAPES")
    print("=" * 84)
    for a in APPROACHES:
        print(f"\n  --- {LABEL[a]} " + "-" * (62 - len(LABEL[a])))
        for line in CODE[a].strip("\n").splitlines():
            print(f"  {line}")
    print()
    print("  Read those four again before the table below. Every conclusion")
    print("  that follows is visible in the SHAPE of the code, not in any")
    print("  benchmark — which is why you can reason about it up front.")
    print()


def comparison() -> None:
    print("=" * 84)
    print("  WHAT HAPPENS WHEN THE REQUIREMENTS CHANGE")
    print("=" * 84)
    print("  edit    = small local change")
    print("  bolt-on = works, but you are now hand-rolling framework features")
    print("  rewrite = the shape of the code has to change")
    print()

    totals = {a: 0 for a in APPROACHES}
    for ch in CHANGES:
        print("-" * 84)
        print(f"  {ch['name']}")
        wrapped, line = [], ""
        for w in ch["why"].split():
            if len(line) + len(w) + 1 > 74:
                wrapped.append(line); line = w
            else:
                line = f"{line} {w}".strip()
        wrapped.append(line)
        for w in wrapped:
            print(f"     {w}")
        print()
        for a in APPROACHES:
            verdict, note = ch["cost"][a]
            totals[a] += SCORE[verdict]
            print(f"     {LABEL[a]:<26}{verdict.upper():<10}", end="")
            first = True
            line = ""
            for w in note.split():
                if len(line) + len(w) + 1 > 44:
                    print(line if first else f"     {'':<36}{line}")
                    first = False
                    line = w
                else:
                    line = f"{line} {w}".strip()
            print(line if first else f"     {'':<36}{line}")
        print()

    print("=" * 84)
    print("  DAMAGE TOTAL  (lower is better; 0 = every change was a small edit)")
    print("=" * 84)
    for a in sorted(APPROACHES, key=lambda x: totals[x]):
        bar = "#" * (totals[a] * 4) or "-"
        print(f"  {LABEL[a]:<26}{totals[a]:>3}   {bar}")
    print()
    for line in [
        "Read this the right way round. It is NOT 'graph wins, use LangGraph'.",
        "",
        "It says: IF those four changes are in your future, the graph shape",
        "absorbs them and the others do not. If none of them are — if the job",
        "really is one model call and a tool — then RAW scores 0 on the only",
        "list that matters, which is the empty one, and it has no dependency,",
        "no version churn and no abstraction to debug through.",
        "",
        "The mistake is not picking the wrong framework. It is picking one",
        "before you know which of these four changes are coming.",
    ]:
        print(f"  {line}")
    print()


def honest_notes() -> None:
    print("=" * 84)
    print("  THE PART FEATURE TABLES LEAVE OUT")
    print("=" * 84)
    for title, body in [
        ("RAW is underrated",
         "A tool-calling loop is about 40 lines. You own it, you can read it, "
         "and it never breaks on a minor version bump. For a single-purpose "
         "agent with two tools this is very often the correct answer, and "
         "saying so in an interview reads as judgement rather than "
         "inexperience."),
        ("CHAIN is not obsolete",
         "For a genuinely fixed pipeline — summarise, extract, format — a "
         "chain is clearer than a graph because it cannot express things you "
         "do not want. Constraint is a feature. Reach for a graph when you "
         "need a backward edge, not because it is newer."),
        ("CREW is the one to be most careful with",
         "Multi-agent systems are appealing because they mirror how a team "
         "works, and that intuition is exactly what makes them hard to debug. "
         "Control flow becomes emergent: who speaks next, when it stops, and "
         "how disagreement resolves are all decided by the framework and by "
         "the models. That is fine for exploration and difficult to defend "
         "when a regulator asks why a decision was made. Prefer explicit "
         "edges when the answer has to be auditable."),
        ("The cost nobody prices",
         "Every framework is a dependency with a release cadence, a set of "
         "breaking changes, and abstractions you will eventually have to read "
         "the source of. For a small team that is a real, recurring tax. "
         "Week 2 priced 0.2 FTE for running an ANN service; a fast-moving "
         "agent framework is not free either."),
    ]:
        print(f"\n  {title.upper()}")
        line = ""
        for w in body.split():
            if len(line) + len(w) + 1 > 74:
                print(f"    {line}"); line = w
            else:
                line = f"{line} {w}".strip()
        print(f"    {line}")
    print()


# ============================================================================
# The chooser
# ============================================================================

def decide(args) -> None:
    print("=" * 84)
    print("  THE CHOOSER")
    print("=" * 84)
    print(f"  backward edges (retry / clarify / re-plan) : {args.loops}")
    print(f"  pauses for a human                         : {args.human_approval}")
    print(f"  must survive a crash / runs for days       : {args.long_running}")
    print(f"  genuinely independent specialists          : {args.specialists}")
    print(f"  answer must be auditable                   : {args.auditable}")
    print(f"  team size                                  : {args.team}")
    print()

    reasons = []
    if args.loops == 0 and not args.human_approval and not args.long_running \
            and args.specialists <= 1:
        pick = "RAW — call the API in your own loop"
        reasons = [
            "Nothing here needs a framework. No backward edges, no pause, no",
            "durability requirement, no second specialist.",
            "",
            "A tool-calling loop is ~40 lines you own outright. Adding a",
            "framework buys you abstractions for problems you do not have, and",
            "charges you a dependency with a release cadence for the privilege.",
        ]
    elif args.long_running or args.human_approval:
        pick = "GRAPH — LangGraph with a Postgres checkpointer"
        reasons = [
            "Pausing for a human and surviving a crash are the same technical",
            "requirement: durable state, resumable at the exact node.",
            "",
            "That is the one thing that is genuinely painful to hand-roll —",
            "it is most of the work in the RAW column — and it is the core",
            "feature of the graph frameworks. Day 5 sizes the Postgres side.",
        ]
        if args.auditable:
            reasons += [
                "",
                "Auditability reinforces this: explicit edges mean the path is a",
                "recorded fact, not an emergent property of a conversation.",
            ]
    elif args.specialists >= 2 and not args.auditable:
        pick = "CREW — but scope it tightly"
        reasons = [
            "Genuinely independent specialists with no audit requirement is the",
            "case crews are built for.",
            "",
            "Scope it: pin who may speak, cap the turns, and make the stopping",
            "rule explicit. Emergent control flow is the appeal and it is also",
            "the thing that makes an incident hard to explain afterwards.",
        ]
    elif args.loops >= 1:
        pick = "GRAPH — LangGraph"
        reasons = [
            f"{args.loops} backward edge(s) means you have a state machine",
            "whether or not you use a framework. You can hand-roll it — Day 1's",
            "engine is 40 lines — but past two loops you are maintaining a graph",
            "engine as a side project.",
        ]
    else:
        pick = "CHAIN — a fixed pipeline"
        reasons = [
            "Fixed order, no loops, no pause. A chain says exactly that and",
            "nothing more. Constraint is a feature: it cannot express the",
            "behaviours you do not want.",
        ]

    print(f"  RECOMMENDATION: {pick}")
    print()
    for r in reasons:
        print(f"  {r}")
    print()

    if args.team <= 3 and pick.startswith("CREW"):
        print("  CAUTION: a small team plus emergent multi-agent control flow is")
        print("  the hardest combination to operate on this page. Debugging 'why")
        print("  did they decide that' costs more than the framework saved.")
        print()

    print("-" * 84)
    print("  WHAT WOULD CHANGE THIS ANSWER")
    print("-" * 84)
    for line in [
        "  add one backward edge          -> RAW and CHAIN start losing",
        "  add a human pause              -> GRAPH, essentially regardless",
        "  drop the durability need       -> RAW becomes competitive again",
        "  add an audit requirement       -> move away from CREW",
        "",
        "  Notice how few of these are about the frameworks. They are about",
        "  your workflow. That is the point: the framework is an OUTPUT of the",
        "  requirements, the same way Week 2 §10 made the vector database an",
        "  output of the sizing questions rather than the first decision.",
    ]:
        print(f"  {line}")
    print("=" * 84)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--show-code", action="store_true")
    p.add_argument("--decide", action="store_true")
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--human-approval", action="store_true")
    p.add_argument("--long-running", action="store_true")
    p.add_argument("--specialists", type=int, default=1)
    p.add_argument("--auditable", action="store_true")
    p.add_argument("--team", type=int, default=4)
    args = p.parse_args()

    if args.decide:
        decide(args)
        return

    show_code()
    comparison()
    honest_notes()
    print("  Try the chooser on your own workflow:")
    print("    python framework_chooser.py --decide --loops 0 --specialists 1")
    print("    python framework_chooser.py --decide --loops 2 --human-approval")
    print("    python framework_chooser.py --decide --specialists 3 --team 3")
    print("=" * 84)


if __name__ == "__main__":
    main()
