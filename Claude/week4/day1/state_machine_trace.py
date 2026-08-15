"""
Week 4, Day 1 — A State Machine You Can Read In One Sitting
============================================================

This is a complete, working graph engine in about 40 lines, plus a refund agent
built on it. Nothing is hidden. If you read this file you will know what
LangGraph does, because LangGraph does this — with checkpointing, streaming and
a nicer API on top.

THE WHOLE IDEA, IN THE WORDS OF THINGS YOU ALREADY KNOW

    state   is a dict. Think of it as ONE ROW in a table that you keep updating.
    a node  is a function that takes the row and returns some columns to change.
            Think: an UPDATE statement.
    an edge is what runs next. A plain edge is "always go to X". A conditional
            edge looks at the row and decides. Think: a WHERE clause / CASE.
    the run  is: start at the entry node, apply nodes, follow edges, stop at END.

That is genuinely all of it. The interesting part is not the machinery, it is
that the edges are allowed to point BACKWARDS.

WHY BACKWARDS EDGES ARE THE WHOLE POINT

A chain (LangChain's classic form, or an Airflow DAG) goes one direction:
A -> B -> C -> done. It cannot re-do a step. So when step B produces something
ambiguous, a chain has exactly two options: guess, or fail.

A graph can send the run back to an earlier node with more information in the
state. That single capability is what makes "ask the user a clarifying question
and then continue" expressible at all.

WHAT THIS SCRIPT SHOWS

  1. The same refund request through a CHAIN. It guesses, and guesses wrong.
  2. The same request through a GRAPH. It loops back, asks, and gets it right.
  3. A full state trace: every node, every field that changed, in order.
  4. The beginner trap: a cycle with no exit condition, and the guard that
     catches it.

stdlib only. Run it and read the output next to the code.

Usage:
    python state_machine_trace.py
    python state_machine_trace.py --request "where is my order"
    python state_machine_trace.py --runaway        # the infinite-loop demo
    python state_machine_trace.py --quiet          # results only, no trace
"""

from __future__ import annotations

import argparse

END = "__end__"


# ============================================================================
# THE ENGINE. This is the entire thing.
# ============================================================================

class Graph:
    def __init__(self, max_steps: int = 25):
        self.nodes = {}          # name -> function(state) -> dict of changes
        self.edges = {}          # name -> next name  (always go here)
        self.branches = {}       # name -> function(state) -> next name
        self.entry = None
        self.max_steps = max_steps

    def node(self, name, fn):
        self.nodes[name] = fn
        return self

    def edge(self, src, dst):
        """Plain edge: after src, always go to dst."""
        self.edges[src] = dst
        return self

    def branch(self, src, chooser):
        """Conditional edge: after src, ASK the state where to go."""
        self.branches[src] = chooser
        return self

    def start(self, name):
        self.entry = name
        return self

    def run(self, state: dict, trace: bool = True) -> dict:
        current = self.entry
        steps = 0
        path = []

        while current != END:
            steps += 1
            if steps > self.max_steps:
                # THE GUARD. A graph with a cycle and no exit will spin
                # forever. Every real engine has this; do not remove it.
                state["_halted"] = f"step limit {self.max_steps} exceeded"
                if trace:
                    print(f"\n  !! HALTED: {state['_halted']}")
                    print(f"     path was: {' -> '.join(path)}")
                break

            before = dict(state)
            changes = self.nodes[current](state) or {}
            state.update(changes)
            path.append(current)

            if trace:
                _print_step(steps, current, before, state)

            if current in self.branches:
                nxt = self.branches[current](state)
                if trace:
                    print(f"        branch chose: {nxt}")
            else:
                nxt = self.edges.get(current, END)
            current = nxt

        state["_path"] = path
        state["_steps"] = steps
        return state


def _print_step(n, name, before, after):
    changed = {k: v for k, v in after.items()
               if k not in before or before[k] != v}
    changed.pop("_halted", None)
    print(f"  step {n}: {name}")
    if not changed:
        print("        (no change to state)")
    def short(x):
        t = repr(x) if isinstance(x, str) else str(x)
        return t if len(t) < 46 else t[:43] + '..."'

    for k, v in changed.items():
        if k in before:
            print(f"        {k}: {short(before[k])} -> {short(v)}")
        else:
            print(f"        {k} = {short(v)}   (new)")


# ============================================================================
# THE AGENT. A retail refund request.
# ============================================================================

# Deliberately tiny "understanding" so you can see exactly why it gets stuck.
CLEAR_INTENTS = {
    "refund": ["refund", "money back", "return this"],
    "track": ["where is", "tracking", "delivered yet"],
    "cancel": ["cancel"],
}

ORDERS = {
    "A-1001": dict(item="laptop sleeve", days_since_delivery=9, price=24.00),
    "A-1002": dict(item="usb c charger", days_since_delivery=41, price=39.00),
}


def classify(state):
    """Work out what the customer wants."""
    text = state["request"].lower()
    for intent, phrases in CLEAR_INTENTS.items():
        if any(p in text for p in phrases):
            return {"intent": intent, "confident": True}
    return {"intent": "unknown", "confident": False}


def ask_clarify(state):
    """
    The node a chain cannot have, because after it you must go BACK.

    In a real system this pauses and waits for the user. Here we simulate the
    reply so the script runs end to end.
    """
    asked = state.get("clarify_count", 0) + 1
    # The customer actually wanted TRACKING. The chain never finds this out.
    canned = {1: "no i just want to know where is my parcel"}
    reply = canned.get(asked, state["request"])
    return {
        "clarify_count": asked,
        "question_asked": "Sorry — did you want a refund, tracking, or a cancellation?",
        "request": state["request"] + " | user said: " + reply,
    }


def lookup_order(state):
    oid = state.get("order_id", "A-1001")
    return {"order": ORDERS[oid], "order_id": oid}


def check_policy(state):
    """30-day return window."""
    days = state["order"]["days_since_delivery"]
    return {
        "within_window": days <= 30,
        "days_since_delivery": days,
    }


def approve(state):
    return {"decision": "APPROVED",
            "reason": f"within the 30-day window ({state['days_since_delivery']} days)"}


def escalate(state):
    return {"decision": "ESCALATED_TO_HUMAN",
            "reason": f"outside the 30-day window ({state['days_since_delivery']} days)"}


def handle_track(state):
    return {"decision": "SENT_TRACKING", "reason": "tracking link emailed"}


def give_up(state):
    return {"decision": "HANDED_TO_AGENT",
            "reason": f"still unclear after {state.get('clarify_count', 0)} questions"}


# ---------------------------------------------------------------------------
# The branching rules. Each is just: look at the state, return a node name.
# ---------------------------------------------------------------------------

def after_classify(state):
    if state["intent"] == "refund":
        return "lookup_order"
    if state["intent"] == "track":
        return "handle_track"
    # Not confident. THIS is the edge a chain cannot express.
    if state.get("clarify_count", 0) >= 2:
        return "give_up"          # the exit condition that makes the loop safe
    return "ask_clarify"


def after_policy(state):
    return "approve" if state["within_window"] else "escalate"


def build_graph(max_steps=25, runaway=False):
    g = Graph(max_steps=max_steps)
    g.node("classify", classify)
    g.node("ask_clarify", ask_clarify)
    g.node("lookup_order", lookup_order)
    g.node("check_policy", check_policy)
    g.node("approve", approve)
    g.node("escalate", escalate)
    g.node("handle_track", handle_track)
    g.node("give_up", give_up)

    g.start("classify")
    g.branch("classify", (lambda s: "ask_clarify") if runaway else after_classify)
    g.edge("ask_clarify", "classify")        # <-- THE BACKWARD EDGE
    g.edge("lookup_order", "check_policy")
    g.branch("check_policy", after_policy)
    g.edge("approve", END)
    g.edge("escalate", END)
    g.edge("handle_track", END)
    g.edge("give_up", END)
    return g


# ============================================================================
# The chain version, for comparison.
# ============================================================================

def run_chain(request: str, trace: bool) -> dict:
    """
    A chain: classify -> lookup -> policy -> decide. One direction only.

    When classify is not confident there is nowhere to go but forward, so the
    chain has to guess. Watch what that costs.
    """
    state = {"request": request}
    steps = [("classify", classify), ("lookup_order", lookup_order),
             ("check_policy", check_policy)]
    for i, (name, fn) in enumerate(steps, 1):
        before = dict(state)
        state.update(fn(state) or {})
        if trace:
            _print_step(i, name, before, state)

    if not state["confident"]:
        # The only options a chain has. Neither is good.
        state["decision"] = "APPROVED"
        state["reason"] = ("GUESSED that an unclear request meant refund, "
                           "because there was nowhere else to go")
        state["guessed"] = True
    elif state["intent"] == "refund":
        state.update(approve(state) if state["within_window"] else escalate(state))
    else:
        state.update(handle_track(state))
    if trace:
        _print_step(4, "decide", {}, {"decision": state["decision"]})
    return state


# ============================================================================

def banner(title):
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", default="hi, this isn't what I expected")
    p.add_argument("--runaway", action="store_true",
                   help="build the graph with a cycle that has no exit")
    p.add_argument("--quiet", action="store_true", help="skip the state trace")
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()
    # The runaway demo is about seeing the loop, not scrolling. Keep it short.
    if args.max_steps is None:
        args.max_steps = 6 if args.runaway else 25
    trace = not args.quiet

    print("=" * 78)
    print("  ONE REQUEST, TWO ARCHITECTURES")
    print("=" * 78)
    print(f"  Customer says: \"{args.request}\"")
    print()
    print("  Notice it doesn't clearly say 'refund'. That ambiguity is the whole")
    print("  experiment — it is where a chain and a graph stop being equivalent.")

    if args.runaway:
        banner("THE BEGINNER TRAP — A CYCLE WITH NO EXIT")
        print("  Built with: branch('classify') -> always 'ask_clarify'")
        print("  and edge('ask_clarify') -> 'classify'. Nothing ever leaves.")
        print()
        g = build_graph(max_steps=args.max_steps, runaway=True)
        s = g.run({"request": args.request}, trace=trace)
        print()
        print("  The guard stopped it. Without a step limit this process runs")
        print("  until something kills it, and every loop costs an LLM call.")
        print()
        print("  THE RULE: every cycle needs an exit condition that is guaranteed")
        print("  to fire. Here it is 'give up after 2 clarifying questions'. A")
        print("  step limit is a BACKSTOP, not the exit condition — if the limit")
        print("  is what stops you, you have a bug, not a design.")
        return

    # ---- chain -------------------------------------------------------------
    banner("1. AS A CHAIN  (one direction, no going back)")
    print("  classify -> lookup_order -> check_policy -> decide")
    print()
    chain = run_chain(args.request, trace)
    print()
    print(f"  RESULT: {chain['decision']}")
    print(f"  WHY   : {chain['reason']}")
    if chain.get("guessed"):
        print()
        print("  Look at what happened. The request was ambiguous, the chain had")
        print("  no way to ask, so it guessed — and it guessed 'refund' on a")
        print("  message that never said refund. In production that is money out")
        print("  the door on a customer who wanted a tracking number.")

    # ---- graph -------------------------------------------------------------
    banner("2. AS A GRAPH  (edges can point backwards)")
    print("  classify --unclear--> ask_clarify --> classify   (the loop)")
    print("           --refund--> lookup_order -> check_policy -+-> approve")
    print("                                                     +-> escalate")
    print("           --track--->  handle_track")
    print("           --still unclear after 2 tries--> give_up")
    print()
    g = build_graph(max_steps=args.max_steps)
    graph = g.run({"request": args.request}, trace=trace)
    print()
    print(f"  RESULT: {graph['decision']}")
    print(f"  WHY   : {graph['reason']}")
    print(f"  PATH  : {' -> '.join(graph['_path'])}")
    print(f"  STEPS : {graph['_steps']}")

    # ---- the comparison ----------------------------------------------------
    banner("WHAT ACTUALLY DIFFERED")
    print(f"  chain decided : {chain['decision']}")
    print(f"  graph decided : {graph['decision']}")
    print()
    if chain.get("guessed"):
        print("  Same nodes. Same logic inside each node. Same model, if there")
        print("  were one. The ONLY difference is that the graph had an edge")
        print("  pointing backwards, so 'I am not sure, let me ask' was a thing")
        print("  it could express.")
        print()
        print("  This is the entire argument for state machines over chains, and")
        print("  it is worth being precise about it in an interview: a graph is")
        print("  not smarter than a chain. It has a larger vocabulary of")
        print("  behaviours. Retry, clarify, re-plan, and escalate are all just")
        print("  'go back to an earlier node with more in the state'.")
    print()
    print("  The three pieces, one more time:")
    print("    state  = a dict you keep updating          (a row)")
    print("    node   = function(state) -> changes        (an UPDATE)")
    print("    edge   = what runs next, maybe conditional (a WHERE / CASE)")
    print()
    print("  Try: python state_machine_trace.py --request 'where is my order'")
    print("       python state_machine_trace.py --runaway")
    print("=" * 78)


if __name__ == "__main__":
    main()
