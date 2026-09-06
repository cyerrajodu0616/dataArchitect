"""Integrated deterministic Week 4 agent: graph, approval, retry, and audit."""

from __future__ import annotations

import argparse
import operator
from threading import RLock
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt


class State(TypedDict, total=False):
    messages: Annotated[list[str], operator.add]
    trace: Annotated[list[str], operator.add]
    audit: Annotated[list[dict[str, Any]], operator.add]
    case_id: str
    intent: str
    amount: float
    needs_approval: bool
    approval_status: str
    decision: str
    refund_id: str
    ambiguous_commit: bool


class RefundReceiver:
    def __init__(self):
        self.records: dict[str, tuple[float, str]] = {}
        self.side_effect_count = 0
        self.failed_after_commit: set[str] = set()
        self._lock = RLock()

    def refund(self, key: str, amount: float, ambiguous: bool) -> str:
        with self._lock:
            return self._refund_locked(key, amount, ambiguous)

    def _refund_locked(self, key: str, amount: float, ambiguous: bool) -> str:
        if key in self.records:
            stored_amount, result = self.records[key]
            if stored_amount != amount: raise ValueError("key reused with different amount")
            return result
        self.side_effect_count += 1
        result = f"refund-{self.side_effect_count}"
        self.records[key] = (amount, result)
        if ambiguous and key not in self.failed_after_commit:
            self.failed_after_commit.add(key)
            raise ConnectionError("response lost after committed refund")
        return result


class ProductionAgent:
    def __init__(self):
        self.saver = InMemorySaver()
        self.receiver = RefundReceiver()
        self.pending: dict[str, dict[str, int]] = {}
        self._lock = RLock()
        self.graph = self._build()

    @staticmethod
    def config(case_id: str):
        return {"configurable": {"thread_id": case_id}}

    def _build(self):
        def classify(state: State) -> State:
            text = " ".join(state["messages"]).lower()
            intent = "refund" if "refund" in text else "track" if any(x in text for x in ("where is", "parcel", "tracking")) else "unknown"
            return {"intent": intent, "trace": ["classify"]}

        def clarify(_state: State) -> State:
            return {"messages": ["user clarification: where is my parcel"], "trace": ["ask_clarify"]}

        def after_classify(state: State) -> Literal["ask_clarify", "check_policy", "handle_track"]:
            if state["intent"] == "refund": return "check_policy"
            if state["intent"] == "track": return "handle_track"
            return "ask_clarify"

        def check_policy(state: State) -> State:
            return {"needs_approval": state["amount"] > 170, "trace": ["check_policy"]}

        def after_policy(state: State) -> Literal["request_approval", "execute_refund"]:
            return "request_approval" if state["needs_approval"] else "execute_refund"

        def request_approval(state: State) -> State:
            response = interrupt({"case_id": state["case_id"], "amount": state["amount"], "allowed_actions": ["approve", "reject"]})
            return {"approval_status": response["status"], "trace": ["request_approval"], "audit": [{"actor": response["actor"], "action": response["status"]}]}

        def after_approval(state: State) -> Literal["execute_refund", "finish"]:
            return "execute_refund" if state["approval_status"] == "APPROVED" else "finish"

        def execute_refund(state: State) -> State:
            key = f"{state['case_id']}:refund"
            refund_id = self.receiver.refund(key, state["amount"], state.get("ambiguous_commit", False))
            return {"refund_id": refund_id, "decision": "REFUNDED", "trace": ["execute_refund"], "audit": [{"actor": "system", "action": "REFUND_EXECUTED", "key": key}]}

        def handle_track(_state: State) -> State:
            return {"decision": "SENT_TRACKING", "trace": ["handle_track"]}

        def finish(state: State) -> State:
            return {"decision": state["approval_status"], "trace": ["finish"]}

        builder = StateGraph(State)
        builder.add_node("classify", classify)
        builder.add_node("ask_clarify", clarify)
        builder.add_node("check_policy", check_policy)
        builder.add_node("request_approval", request_approval)
        builder.add_node("execute_refund", execute_refund, retry_policy=RetryPolicy(initial_interval=0, backoff_factor=1, max_interval=0, max_attempts=2, jitter=False, retry_on=ConnectionError))
        builder.add_node("handle_track", handle_track)
        builder.add_node("finish", finish)
        builder.add_edge(START, "classify")
        builder.add_conditional_edges("classify", after_classify)
        builder.add_edge("ask_clarify", "classify")
        builder.add_conditional_edges("check_policy", after_policy)
        builder.add_conditional_edges("request_approval", after_approval)
        for node in ("execute_refund", "handle_track", "finish"):
            builder.add_edge(node, END)
        return builder.compile(checkpointer=self.saver)

    def start(self, case_id: str, message: str, amount: float = 100, *, ambiguous_commit: bool = False, now: int = 100, ttl: int = 10):
        result = self.graph.invoke({"case_id": case_id, "messages": [message], "amount": amount, "trace": [], "audit": [], "ambiguous_commit": ambiguous_commit}, config=self.config(case_id))
        if "__interrupt__" in result:
            self.pending[case_id] = {"expires_at": now + ttl}
        return result

    def resume(self, case_id: str, action: Literal["approve", "reject"], *, actor: str = "reviewer-1", now: int = 101):
        with self._lock:
            if case_id not in self.pending:
                raise ValueError("case is not pending")
            status = "EXPIRED" if now >= self.pending[case_id]["expires_at"] else "APPROVED" if action == "approve" else "REJECTED"
            del self.pending[case_id]
            return self.graph.invoke(Command(resume={"status": status, "actor": actor}), config=self.config(case_id))


SCENARIOS = ("tracking", "refund-approved", "refund-rejected", "refund-expired", "ambiguous-commit")


def run_scenario(name: str):
    agent = ProductionAgent()
    if name == "tracking": result = agent.start("c1", "where is my parcel")
    elif name == "ambiguous-commit": result = agent.start("c1", "refund please", 100, ambiguous_commit=True)
    else:
        result = agent.start("c1", "refund please", 240, now=100, ttl=10)
        if name == "refund-approved": result = agent.resume("c1", "approve", now=101)
        elif name == "refund-rejected": result = agent.resume("c1", "reject", now=101)
        else: result = agent.resume("c1", "approve", now=111)
    return agent, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("scenario", choices=SCENARIOS)
    args = parser.parse_args(); agent, result = run_scenario(args.scenario)
    print(f"decision={result.get('decision')}\ntrace={result.get('trace')}\naudit={result.get('audit')}\nside_effects={agent.receiver.side_effect_count}")


if __name__ == "__main__": main()
