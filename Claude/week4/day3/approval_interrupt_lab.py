"""A deterministic, fail-closed human approval lifecycle using LangGraph interrupts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class ApprovalState(TypedDict, total=False):
    case_id: str
    amount: float
    state_version: int
    approval_status: str
    reviewer: str
    edited_amount: float
    decision: str
    refund_executed: bool


@dataclass(frozen=True)
class AuditEvent:
    sequence: int
    actor: str
    action: str
    version: int
    timestamp: int
    detail: str = ""


@dataclass
class ApprovalRecord:
    case_id: str
    amount: float
    version: int
    expires_at: int
    status: str = "PENDING"
    interrupt_payload: dict[str, Any] = field(default_factory=dict)
    audit: list[AuditEvent] = field(default_factory=list)
    result: dict[str, Any] | None = None


def build_graph(checkpointer: InMemorySaver):
    def request_approval(state: ApprovalState) -> ApprovalState:
        response = interrupt({
            "kind": "refund_approval",
            "case_id": state["case_id"],
            "amount": state["amount"],
            "state_version": state["state_version"],
            "allowed_actions": ["approve", "reject", "edit"],
        })
        update: ApprovalState = {
            "approval_status": response["status"],
            "reviewer": response["actor"],
        }
        if "edited_amount" in response:
            update["edited_amount"] = float(response["edited_amount"])
        return update

    def route(state: ApprovalState) -> Literal["execute_refund", "finish"]:
        return "execute_refund" if state["approval_status"] == "APPROVED" else "finish"

    def execute_refund(state: ApprovalState) -> ApprovalState:
        amount = state.get("edited_amount", state["amount"])
        return {"refund_executed": True, "decision": f"REFUNDED_{amount:.2f}"}

    def finish(state: ApprovalState) -> ApprovalState:
        return {"refund_executed": False, "decision": state["approval_status"]}

    builder = StateGraph(ApprovalState)
    builder.add_node("request_approval", request_approval)
    builder.add_node("execute_refund", execute_refund)
    builder.add_node("finish", finish)
    builder.add_edge(START, "request_approval")
    builder.add_conditional_edges("request_approval", route)
    builder.add_edge("execute_refund", END)
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer)


class ApprovalService:
    """Application boundary enforcing auth, expiry, versions, and terminal states."""

    def __init__(self) -> None:
        self.checkpointer = InMemorySaver()
        self.graph = build_graph(self.checkpointer)
        self.records: dict[str, ApprovalRecord] = {}
        self._lock = RLock()

    @staticmethod
    def _config(case_id: str) -> dict[str, dict[str, str]]:
        return {"configurable": {"thread_id": case_id}}

    def _audit(self, record: ApprovalRecord, actor: str, action: str, now: int, detail: str = "") -> None:
        record.audit.append(AuditEvent(len(record.audit) + 1, actor, action, record.version, now, detail))

    def start(self, case_id: str, amount: float, *, version: int = 1, now: int = 100, ttl: int = 60) -> ApprovalRecord:
        with self._lock:
            return self._start_locked(case_id, amount, version=version, now=now, ttl=ttl)

    def _start_locked(self, case_id: str, amount: float, *, version: int, now: int, ttl: int) -> ApprovalRecord:
        if case_id in self.records:
            raise ValueError("case already exists")
        result = self.graph.invoke(
            {"case_id": case_id, "amount": amount, "state_version": version},
            config=self._config(case_id),
        )
        payload = result["__interrupt__"][0].value
        record = ApprovalRecord(case_id, amount, version, now + ttl, interrupt_payload=payload)
        self._audit(record, "system", "REQUESTED", now)
        self.records[case_id] = record
        return record

    def respond(
        self,
        case_id: str,
        *,
        actor: str,
        roles: set[str],
        action: Literal["approve", "reject", "edit"],
        expected_version: int,
        now: int,
        edited_amount: float | None = None,
    ) -> ApprovalRecord:
        with self._lock:
            return self._respond_locked(
                case_id,
                actor=actor,
                roles=roles,
                action=action,
                expected_version=expected_version,
                now=now,
                edited_amount=edited_amount,
            )

    def _respond_locked(
        self,
        case_id: str,
        *,
        actor: str,
        roles: set[str],
        action: Literal["approve", "reject", "edit"],
        expected_version: int,
        now: int,
        edited_amount: float | None,
    ) -> ApprovalRecord:
        record = self.records[case_id]
        if record.status != "PENDING":
            self._audit(record, actor, "DUPLICATE_IGNORED", now, f"already {record.status}")
            return record
        if "approver" not in roles:
            self._audit(record, actor, "UNAUTHORIZED", now)
            raise PermissionError("actor lacks approver role")

        if now >= record.expires_at:
            status = "EXPIRED"
        elif expected_version != record.version:
            status = "STALE"
        elif action == "reject":
            status = "REJECTED"
        elif action == "edit":
            if edited_amount is None or edited_amount < 0:
                raise ValueError("edit requires a non-negative edited_amount")
            status = "APPROVED"
        else:
            status = "APPROVED"

        response: dict[str, Any] = {"status": status, "actor": actor}
        if action == "edit":
            response["edited_amount"] = edited_amount
        record.result = self.graph.invoke(Command(resume=response), config=self._config(case_id))
        record.status = status
        self._audit(record, actor, action.upper() if status == "APPROVED" else status, now)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("approve", "reject", "edit", "expire"), default="approve")
    parser.add_argument("--amount", type=float, default=240.0)
    args = parser.parse_args()
    service = ApprovalService()
    record = service.start("case-001", args.amount, now=100, ttl=10)
    print(f"interrupt: {record.interrupt_payload}")
    now = 111 if args.action == "expire" else 101
    action = "approve" if args.action == "expire" else args.action
    record = service.respond("case-001", actor="reviewer-7", roles={"approver"}, action=action, expected_version=1, now=now, edited_amount=120.0 if action == "edit" else None)
    print(f"status: {record.status}\nresult: {record.result}\naudit: {record.audit}")


if __name__ == "__main__":
    main()
