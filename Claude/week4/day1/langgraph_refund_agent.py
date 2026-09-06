"""A deterministic retail agent built with the real LangGraph Graph API.

The classifier intentionally uses keyword matching rather than an LLM. That keeps
the example free, repeatable, and focused on orchestration: typed state, reducers,
partial updates, conditional edges, streaming, and bounded loops.

Usage:
    python langgraph_refund_agent.py
    python langgraph_refund_agent.py --stream
    python langgraph_refund_agent.py --request "refund please" --order-id A-1002
    python langgraph_refund_agent.py --runaway --max-steps 4
"""

from __future__ import annotations

import argparse
import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict, total=False):
    messages: Annotated[list[str], operator.add]
    order_id: str
    intent: str
    confident: bool
    clarify_count: int
    clarification_replies: list[str]
    order: dict[str, Any]
    within_window: bool
    days_since_delivery: int
    decision: str
    reason: str


CLEAR_INTENTS = {
    "refund": ["refund", "money back", "return this"],
    "track": ["where is", "tracking", "delivered yet"],
    "cancel": ["cancel"],
}

ORDERS = {
    "A-1001": dict(item="laptop sleeve", days_since_delivery=9, price=24.00),
    "A-1002": dict(item="usb c charger", days_since_delivery=41, price=39.00),
}


def classify(state: AgentState) -> AgentState:
    """Classify the whole conversation and return only changed scalar fields."""
    text = " ".join(state["messages"]).lower()
    for intent, phrases in CLEAR_INTENTS.items():
        if any(phrase in text for phrase in phrases):
            return {"intent": intent, "confident": True}
    return {"intent": "unknown", "confident": False}


def ask_clarify(state: AgentState) -> AgentState:
    """Append one simulated customer reply; a real system would interrupt here."""
    count = state.get("clarify_count", 0)
    replies = state.get(
        "clarification_replies",
        ["no i just want to know where is my parcel"],
    )
    reply = replies[count] if count < len(replies) else "still unclear"
    return {
        "clarify_count": count + 1,
        "messages": [f"user clarification: {reply}"],
    }


def lookup_order(state: AgentState) -> AgentState:
    order_id = state.get("order_id", "A-1001")
    return {"order": ORDERS[order_id], "order_id": order_id}


def check_policy(state: AgentState) -> AgentState:
    days = int(state["order"]["days_since_delivery"])
    return {"within_window": days <= 30, "days_since_delivery": days}


def approve(state: AgentState) -> AgentState:
    return {
        "decision": "APPROVED",
        "reason": f"within the 30-day window ({state['days_since_delivery']} days)",
    }


def escalate(state: AgentState) -> AgentState:
    return {
        "decision": "ESCALATED_TO_HUMAN",
        "reason": f"outside the 30-day window ({state['days_since_delivery']} days)",
    }


def handle_track(state: AgentState) -> AgentState:
    return {"decision": "SENT_TRACKING", "reason": "tracking link emailed"}


def handle_cancel(state: AgentState) -> AgentState:
    return {"decision": "CANCELLATION_REQUESTED", "reason": "sent to cancellation flow"}


def give_up(state: AgentState) -> AgentState:
    return {
        "decision": "HANDED_TO_AGENT",
        "reason": f"still unclear after {state.get('clarify_count', 0)} questions",
    }


def after_classify(
    state: AgentState,
) -> Literal["lookup_order", "handle_track", "handle_cancel", "ask_clarify", "give_up"]:
    if state["intent"] == "refund":
        return "lookup_order"
    if state["intent"] == "track":
        return "handle_track"
    if state["intent"] == "cancel":
        return "handle_cancel"
    if state.get("clarify_count", 0) >= 2:
        return "give_up"
    return "ask_clarify"


def after_policy(state: AgentState) -> Literal["approve", "escalate"]:
    return "approve" if state["within_window"] else "escalate"


def build_graph(*, runaway: bool = False):
    """Build and compile the graph; runaway=True demonstrates the runtime guard."""
    builder = StateGraph(AgentState)
    for node in (
        classify,
        ask_clarify,
        lookup_order,
        check_policy,
        approve,
        escalate,
        handle_track,
        handle_cancel,
        give_up,
    ):
        builder.add_node(node.__name__, node)

    builder.add_edge(START, "classify")
    route = (lambda _state: "ask_clarify") if runaway else after_classify
    builder.add_conditional_edges("classify", route)
    builder.add_edge("ask_clarify", "classify")
    builder.add_edge("lookup_order", "check_policy")
    builder.add_conditional_edges("check_policy", after_policy)
    for terminal in (
        "approve",
        "escalate",
        "handle_track",
        "handle_cancel",
        "give_up",
    ):
        builder.add_edge(terminal, END)
    return builder.compile()


def initial_state(
    request: str,
    order_id: str = "A-1001",
    clarification_replies: list[str] | None = None,
) -> AgentState:
    state: AgentState = {"messages": [request], "order_id": order_id}
    if clarification_replies is not None:
        state["clarification_replies"] = clarification_replies
    return state


def run_agent(
    request: str,
    *,
    order_id: str = "A-1001",
    clarification_replies: list[str] | None = None,
    max_steps: int = 25,
) -> AgentState:
    graph = build_graph()
    return graph.invoke(
        initial_state(request, order_id, clarification_replies),
        config={"recursion_limit": max_steps},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", default="hi, this isn't what I expected")
    parser.add_argument("--order-id", choices=sorted(ORDERS), default="A-1001")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--runaway", action="store_true")
    parser.add_argument("--max-steps", type=int, default=25)
    args = parser.parse_args()

    graph = build_graph(runaway=args.runaway)
    state = initial_state(args.request, args.order_id)
    config = {"recursion_limit": args.max_steps}

    try:
        if args.stream:
            for update in graph.stream(state, config=config, stream_mode="updates"):
                for node, changes in update.items():
                    print(f"{node:>14} -> {changes}")
        else:
            result = graph.invoke(state, config=config)
            print(f"decision: {result['decision']}")
            print(f"reason:   {result['reason']}")
            print(f"messages: {result['messages']}")
    except GraphRecursionError:
        print(
            f"HALTED: recursion limit {args.max_steps} reached. "
            "The runtime guard caught a loop whose business exit condition did not fire."
        )


if __name__ == "__main__":
    main()
