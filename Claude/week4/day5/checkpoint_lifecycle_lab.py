"""Checkpoint history plus deterministic lease/concurrency and schema migration."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from threading import RLock
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph


class DemoState(TypedDict, total=False):
    value: int
    doubled: int


def build_demo_graph(checkpointer: InMemorySaver):
    def increment(state: DemoState) -> DemoState:
        return {"value": state.get("value", 0) + 1}

    def double(state: DemoState) -> DemoState:
        return {"doubled": state["value"] * 2}

    builder = StateGraph(DemoState)
    builder.add_node("increment", increment)
    builder.add_node("double", double)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", "double")
    builder.add_edge("double", END)
    return builder.compile(checkpointer=checkpointer)


def run_checkpoint_demo(thread_id: str, value: int = 0):
    saver = InMemorySaver()
    graph = build_demo_graph(saver)
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"value": value}, config=config)
    history = list(graph.get_state_history(config))
    return graph, result, history


@dataclass
class LeaseRecord:
    state: dict
    version: int = 1
    schema_version: int = 2
    status: str = "READY"
    lease_owner: str | None = None
    lease_until: int = 0


class CheckpointLeaseStore:
    """A compare-and-swap model for preventing duplicate concurrent resumes."""

    def __init__(self):
        self.records: dict[str, LeaseRecord] = {}
        self._lock = RLock()

    def create(self, thread_id: str, state: dict, schema_version: int = 2) -> LeaseRecord:
        with self._lock:
            if thread_id in self.records:
                raise ValueError("thread already exists")
            record = LeaseRecord(dict(state), schema_version=schema_version)
            self.records[thread_id] = record
            return record

    def acquire(self, thread_id: str, owner: str, expected_version: int, *, now: int, ttl: int = 30) -> LeaseRecord:
        with self._lock:
            record = self.records[thread_id]
            if record.status == "TERMINAL":
                raise ValueError("terminal checkpoint cannot resume")
            if record.version != expected_version:
                raise ValueError("stale checkpoint version")
            if record.lease_owner is not None and record.lease_until > now:
                raise RuntimeError("checkpoint already leased")
            record.lease_owner, record.lease_until, record.status = owner, now + ttl, "RUNNING"
            return record

    def complete(self, thread_id: str, owner: str, state: dict, *, terminal: bool = False) -> LeaseRecord:
        with self._lock:
            record = self.records[thread_id]
            if record.lease_owner != owner:
                raise PermissionError("worker does not own lease")
            record.state = dict(state)
            record.version += 1
            record.lease_owner, record.lease_until = None, 0
            record.status = "TERMINAL" if terminal else "READY"
            return record


def migrate_v1_to_v2(state: dict) -> dict:
    migrated = dict(state)
    if "request" in migrated and "messages" not in migrated:
        migrated["messages"] = [migrated.pop("request")]
    migrated["schema_version"] = 2
    return migrated


def trim_state(state: dict) -> dict:
    """Keep decision inputs; remove raw payloads and direct PII from checkpoints."""
    return {key: value for key, value in state.items() if key not in {"raw_tool_output", "ssn", "email"}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--thread-id", default="demo-1")
    args = parser.parse_args()
    _, result, history = run_checkpoint_demo(args.thread_id)
    print(f"result={result}\ncheckpoints={len(history)}")
    for snapshot in history:
        print(snapshot.metadata.get("step"), snapshot.next, snapshot.values)


if __name__ == "__main__":
    main()
