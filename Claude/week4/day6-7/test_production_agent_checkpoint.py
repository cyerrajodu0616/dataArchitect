import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from production_agent_checkpoint import ProductionAgent, run_scenario


class ProductionCheckpointTests(unittest.TestCase):
    def test_tracking_scenario(self):
        _, result = run_scenario("tracking"); self.assertEqual(result["decision"], "SENT_TRACKING")

    def test_approved_refund(self):
        agent, result = run_scenario("refund-approved"); self.assertEqual(result["decision"], "REFUNDED"); self.assertEqual(agent.receiver.side_effect_count, 1)

    def test_rejected_refund(self):
        agent, result = run_scenario("refund-rejected"); self.assertEqual(result["decision"], "REJECTED"); self.assertEqual(agent.receiver.side_effect_count, 0)

    def test_expired_refund_fails_closed(self):
        agent, result = run_scenario("refund-expired"); self.assertEqual(result["decision"], "EXPIRED"); self.assertEqual(agent.receiver.side_effect_count, 0)

    def test_ambiguous_commit_is_deduplicated(self):
        agent, result = run_scenario("ambiguous-commit"); self.assertEqual(result["decision"], "REFUNDED"); self.assertEqual(agent.receiver.side_effect_count, 1)

    def test_trace_order(self):
        _, result = run_scenario("refund-approved"); self.assertEqual(result["trace"], ["classify", "check_policy", "request_approval", "execute_refund"])

    def test_interrupt_is_checkpoint_boundary(self):
        agent = ProductionAgent(); result = agent.start("x", "refund please", 240)
        self.assertIn("__interrupt__", result)
        self.assertTrue(list(agent.graph.get_state_history(agent.config("x"))))

    def test_no_side_effect_before_approval(self):
        agent = ProductionAgent(); agent.start("x", "refund please", 240)
        self.assertEqual(agent.receiver.side_effect_count, 0)

    def test_terminal_resume_is_protected(self):
        agent = ProductionAgent(); agent.start("x", "refund please", 240); agent.resume("x", "approve")
        with self.assertRaises(ValueError): agent.resume("x", "approve")

    def test_audit_records_approval_and_refund(self):
        _, result = run_scenario("refund-approved")
        self.assertEqual([event["action"] for event in result["audit"]], ["APPROVED", "REFUND_EXECUTED"])


if __name__ == "__main__": unittest.main()
