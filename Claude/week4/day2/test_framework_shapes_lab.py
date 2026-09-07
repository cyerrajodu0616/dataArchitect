import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework_shapes_lab import RUNNERS, run_chain, run_graph, run_raw


class FrameworkShapesTests(unittest.TestCase):
    def test_clear_tracking_outcome_is_common(self):
        for runner in RUNNERS.values():
            self.assertEqual(runner("where is my parcel").decision, "SENT_TRACKING")

    def test_chain_cannot_clarify(self):
        result = run_chain("this is not what I expected")
        self.assertEqual(result.decision, "APPROVED_GUESSED_REFUND")
        self.assertNotIn("ask_clarify", result.path)

    def test_raw_and_graph_clarify(self):
        for runner in (run_raw, run_graph):
            self.assertIn("ask_clarify", runner("this is not what I expected").path)

    def test_routing_ownership_is_explicit(self):
        labels = {name: runner("where is my parcel").routing_owner for name, runner in RUNNERS.items()}
        self.assertEqual(len(set(labels.values())), 4)

    def test_graph_trace_contains_second_classification(self):
        result = run_graph("this is not what I expected")
        self.assertEqual(result.path[:3], ("classify", "ask_clarify", "classify"))

    def test_approval_change_is_visible(self):
        for runner in RUNNERS.values():
            result = runner("refund please", require_approval=True)
            self.assertEqual(result.decision, "PENDING_APPROVAL")


if __name__ == "__main__":
    unittest.main()
