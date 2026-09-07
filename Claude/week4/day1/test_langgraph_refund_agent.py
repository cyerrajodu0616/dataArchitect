"""Tests for the real LangGraph Day 1 companion."""

import sys
import unittest
from pathlib import Path

from langgraph.errors import GraphRecursionError

sys.path.insert(0, str(Path(__file__).resolve().parent))

from langgraph_refund_agent import build_graph, initial_state, run_agent


class LangGraphRefundAgentTests(unittest.TestCase):
    def test_ambiguous_request_becomes_tracking(self) -> None:
        result = run_agent("hi, this isn't what I expected")
        self.assertEqual(result["decision"], "SENT_TRACKING")

    def test_message_reducer_preserves_and_appends(self) -> None:
        original = "hi, this isn't what I expected"
        result = run_agent(original)
        self.assertEqual(result["messages"][0], original)
        self.assertEqual(len(result["messages"]), 2)
        self.assertIn("user clarification:", result["messages"][1])

    def test_refund_inside_window_is_approved(self) -> None:
        result = run_agent("I want a refund", order_id="A-1001")
        self.assertEqual(result["decision"], "APPROVED")

    def test_refund_outside_window_is_escalated(self) -> None:
        result = run_agent("I want a refund", order_id="A-1002")
        self.assertEqual(result["decision"], "ESCALATED_TO_HUMAN")

    def test_two_unresolved_replies_hand_off(self) -> None:
        result = run_agent(
            "something happened",
            clarification_replies=["not sure", "still not saying"],
        )
        self.assertEqual(result["decision"], "HANDED_TO_AGENT")
        self.assertEqual(result["clarify_count"], 2)

    def test_stream_exposes_node_updates_in_order(self) -> None:
        graph = build_graph()
        chunks = list(
            graph.stream(
                initial_state("hi, this isn't what I expected"),
                stream_mode="updates",
            )
        )
        nodes = [node for chunk in chunks for node in chunk]
        self.assertEqual(nodes, ["classify", "ask_clarify", "classify", "handle_track"])

    def test_runtime_guard_stops_runaway_graph(self) -> None:
        graph = build_graph(runaway=True)
        with self.assertRaises(GraphRecursionError):
            graph.invoke(
                initial_state("still ambiguous"),
                config={"recursion_limit": 4},
            )


if __name__ == "__main__":
    unittest.main()
