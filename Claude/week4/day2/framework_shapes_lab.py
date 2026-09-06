"""Run one refund workflow through four control-flow shapes.

This is a structural lab, not a framework or model benchmark. The crew version is
an explicit coordinator simulation; it does not import or measure CrewAI/AutoGen.
"""

from __future__ import annotations

import argparse
import operator
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

CLEAR_INTENTS = {
    "refund": ["refund", "money back", "return this"],
    "track": ["where is", "tracking", "parcel"],
}
ORDERS = {
    "A-1001": {"days_since_delivery": 9},
    "A-1002": {"days_since_delivery": 41},
}


@dataclass(frozen=True)
class RunResult:
    decision: str
    path: tuple[str, ...]
    state: dict[str, Any]
    routing_owner: str


def _intent(messages: list[str]) -> str:
    text = " ".join(messages).lower()
    for intent, phrases in CLEAR_INTENTS.items():
        if any(phrase in text for phrase in phrases):
            return intent
    return "unknown"


def _refund_decision(order_id: str, require_approval: bool) -> str:
    if ORDERS[order_id]["days_since_delivery"] > 30:
        return "ESCALATED_TO_HUMAN"
    return "PENDING_APPROVAL" if require_approval else "APPROVED"


def run_raw(request: str, order_id: str = "A-1001", require_approval: bool = False) -> RunResult:
    messages, path = [request], []
    for turn in range(2):
        path.append("classify")
        intent = _intent(messages)
        if intent != "unknown":
            break
        path.append("ask_clarify")
        messages.append("user clarification: where is my parcel" if turn == 0 else "still unclear")
    if intent == "track":
        path.append("handle_track"); decision = "SENT_TRACKING"
    elif intent == "refund":
        path.extend(("lookup_order", "check_policy"))
        decision = _refund_decision(order_id, require_approval)
        path.append("approval_gate" if decision == "PENDING_APPROVAL" else "decide")
    else:
        path.append("give_up"); decision = "HANDED_TO_AGENT"
    return RunResult(decision, tuple(path), {"messages": messages, "intent": intent}, "application loop")


def run_chain(request: str, order_id: str = "A-1001", require_approval: bool = False) -> RunResult:
    intent = _intent([request])
    path = ["classify", "lookup_order", "check_policy", "decide"]
    if intent == "track":
        decision = "SENT_TRACKING"
    elif intent == "refund":
        decision = _refund_decision(order_id, require_approval)
    else:
        decision = "APPROVED_GUESSED_REFUND"
    return RunResult(decision, tuple(path), {"messages": [request], "intent": intent}, "fixed pipeline")


class GraphState(TypedDict, total=False):
    messages: Annotated[list[str], operator.add]
    order_id: str
    require_approval: bool
    intent: str
    clarify_count: int
    decision: str
    path: Annotated[list[str], operator.add]


def _build_graph():
    def classify(state: GraphState) -> GraphState:
        return {"intent": _intent(state["messages"]), "path": ["classify"]}

    def clarify(state: GraphState) -> GraphState:
        return {"messages": ["user clarification: where is my parcel"], "clarify_count": state.get("clarify_count", 0) + 1, "path": ["ask_clarify"]}

    def route(state: GraphState) -> Literal["ask_clarify", "refund", "track", "give_up"]:
        if state["intent"] == "refund": return "refund"
        if state["intent"] == "track": return "track"
        return "give_up" if state.get("clarify_count", 0) >= 2 else "ask_clarify"

    def refund(state: GraphState) -> GraphState:
        decision = _refund_decision(state["order_id"], state["require_approval"])
        return {"decision": decision, "path": ["lookup_order", "check_policy", "approval_gate" if decision == "PENDING_APPROVAL" else "decide"]}

    def track(_state: GraphState) -> GraphState:
        return {"decision": "SENT_TRACKING", "path": ["handle_track"]}

    def give_up(_state: GraphState) -> GraphState:
        return {"decision": "HANDED_TO_AGENT", "path": ["give_up"]}

    builder = StateGraph(GraphState)
    for name, node in (("classify", classify), ("ask_clarify", clarify), ("refund", refund), ("track", track), ("give_up", give_up)):
        builder.add_node(name, node)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges("classify", route)
    builder.add_edge("ask_clarify", "classify")
    for terminal in ("refund", "track", "give_up"):
        builder.add_edge(terminal, END)
    return builder.compile()


def run_graph(request: str, order_id: str = "A-1001", require_approval: bool = False) -> RunResult:
    state = _build_graph().invoke({"messages": [request], "order_id": order_id, "require_approval": require_approval, "path": []})
    return RunResult(state["decision"], tuple(state["path"]), dict(state), "declared conditional edges")


def run_crew_simulation(request: str, order_id: str = "A-1001", require_approval: bool = False) -> RunResult:
    """Model a coordinator delegating to specialists; this is not CrewAI code."""
    messages = [request]
    path = ["coordinator", "intent_specialist"]
    intent = _intent(messages)
    if intent == "unknown":
        path.extend(("coordinator", "clarification_specialist", "intent_specialist"))
        messages.append("user clarification: where is my parcel")
        intent = _intent(messages)
    if intent == "track":
        path.extend(("coordinator", "tracking_specialist")); decision = "SENT_TRACKING"
    else:
        path.extend(("coordinator", "policy_specialist")); decision = _refund_decision(order_id, require_approval)
    return RunResult(decision, tuple(path), {"messages": messages, "intent": intent}, "coordinator simulation")


RUNNERS = {"raw": run_raw, "chain": run_chain, "graph": run_graph, "crew": run_crew_simulation}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach", choices=RUNNERS, default="graph")
    parser.add_argument("--request", default="hi, this isn't what I expected")
    parser.add_argument("--order-id", choices=ORDERS, default="A-1001")
    parser.add_argument("--require-approval", action="store_true")
    parser.add_argument("--trace", action="store_true")
    args = parser.parse_args()
    result = RUNNERS[args.approach](args.request, args.order_id, args.require_approval)
    print(f"approach: {args.approach}\nrouting:  {result.routing_owner}\ndecision: {result.decision}")
    if args.trace:
        print("path:     " + " -> ".join(result.path))


if __name__ == "__main__":
    main()
