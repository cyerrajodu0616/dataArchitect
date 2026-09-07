import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from approval_interrupt_lab import ApprovalService


class ApprovalInterruptTests(unittest.TestCase):
    def setUp(self):
        self.service = ApprovalService()

    def test_initial_interrupt_payload(self):
        record = self.service.start("a", 240, version=3)
        self.assertEqual(record.interrupt_payload["state_version"], 3)
        self.assertEqual(record.status, "PENDING")

    def test_authorized_approval(self):
        self.service.start("a", 240)
        record = self.service.respond("a", actor="r1", roles={"approver"}, action="approve", expected_version=1, now=101)
        self.assertTrue(record.result["refund_executed"])

    def test_rejection_is_fail_closed(self):
        self.service.start("a", 240)
        record = self.service.respond("a", actor="r1", roles={"approver"}, action="reject", expected_version=1, now=101)
        self.assertFalse(record.result["refund_executed"])

    def test_edit_then_approve(self):
        self.service.start("a", 240)
        record = self.service.respond("a", actor="r1", roles={"approver"}, action="edit", expected_version=1, now=101, edited_amount=120)
        self.assertEqual(record.result["decision"], "REFUNDED_120.00")

    def test_unauthorized_actor(self):
        self.service.start("a", 240)
        with self.assertRaises(PermissionError):
            self.service.respond("a", actor="viewer", roles={"viewer"}, action="approve", expected_version=1, now=101)

    def test_expired_approval(self):
        self.service.start("a", 240, now=100, ttl=2)
        record = self.service.respond("a", actor="r1", roles={"approver"}, action="approve", expected_version=1, now=103)
        self.assertEqual(record.status, "EXPIRED")
        self.assertFalse(record.result["refund_executed"])

    def test_stale_version(self):
        self.service.start("a", 240, version=2)
        record = self.service.respond("a", actor="r1", roles={"approver"}, action="approve", expected_version=1, now=101)
        self.assertEqual(record.status, "STALE")

    def test_duplicate_response_is_ignored(self):
        self.service.start("a", 240)
        first = self.service.respond("a", actor="r1", roles={"approver"}, action="approve", expected_version=1, now=101)
        second = self.service.respond("a", actor="r2", roles={"approver"}, action="reject", expected_version=1, now=102)
        self.assertIs(first, second)
        self.assertEqual(second.status, "APPROVED")

    def test_first_concurrent_response_wins(self):
        self.service.start("a", 240)
        barrier = Barrier(2)

        def respond(actor, action):
            barrier.wait()
            return self.service.respond(
                "a",
                actor=actor,
                roles={"approver"},
                action=action,
                expected_version=1,
                now=101,
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            records = list(pool.map(lambda item: respond(*item), [("r1", "reject"), ("r2", "approve")]))

        self.assertIs(records[0], records[1])
        record = records[0]
        self.assertIn(record.status, {"APPROVED", "REJECTED"})
        self.assertEqual([event.action for event in record.audit].count("DUPLICATE_IGNORED"), 1)
        self.assertEqual(record.result["refund_executed"], record.status == "APPROVED")

    def test_audit_and_same_thread_resume(self):
        self.service.start("case-x", 240)
        record = self.service.respond("case-x", actor="r1", roles={"approver"}, action="approve", expected_version=1, now=101)
        self.assertEqual([event.action for event in record.audit], ["REQUESTED", "APPROVE"])
        snapshot = self.service.graph.get_state({"configurable": {"thread_id": "case-x"}})
        self.assertEqual(snapshot.values["decision"], "REFUNDED_240.00")


if __name__ == "__main__":
    unittest.main()
