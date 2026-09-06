import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from resilient_refund_lab import CircuitBreaker, CircuitState, FailureKind, IdempotencyStatus, IdempotentReceiver, OperationFailure, ReliabilityBoundary, classify_exception, end_to_end_independent, full_jitter, repair_model_output


class ReliabilityBoundaryTests(unittest.TestCase):
    def test_failure_classification(self):
        self.assertEqual(classify_exception(ConnectionError()), FailureKind.TRANSIENT)
        self.assertEqual(classify_exception(ValueError()), FailureKind.PERMANENT)

    def test_permanent_is_not_retried(self):
        boundary = ReliabilityBoundary()
        boundary.execute("k", "p", lambda: (_ for _ in ()).throw(OperationFailure(FailureKind.PERMANENT, "bad")))
        self.assertEqual(boundary.dead_letters[0].attempts, 1)

    def test_ambiguous_commit_uses_stable_key(self):
        receiver, boundary = IdempotentReceiver(), ReliabilityBoundary()
        outcomes = iter(["ambiguous", "success"])
        result = boundary.execute("k", "p", lambda: receiver.refund("k", "p", next(outcomes)))
        self.assertTrue(result.deduplicated)
        self.assertEqual(receiver.side_effect_count, 1)
        self.assertEqual(receiver.records["k"].status, IdempotencyStatus.COMPLETED)

    def test_conflicting_payload_is_rejected(self):
        receiver = IdempotentReceiver(); receiver.refund("k", "p1")
        with self.assertRaises(ValueError): receiver.refund("k", "p2")

    def test_concurrent_duplicate_is_executed_once(self):
        receiver = IdempotentReceiver()
        barrier = Barrier(2)

        def refund():
            barrier.wait()
            return receiver.refund("k", "p")

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _item: refund(), range(2)))

        self.assertEqual(receiver.side_effect_count, 1)
        self.assertEqual(results[0].value, results[1].value)
        self.assertEqual(sum(result.deduplicated for result in results), 1)

    def test_full_jitter_bounds(self):
        self.assertEqual(full_jitter(2, 3, lambda: 0), 0)
        self.assertLess(full_jitter(2, 3, lambda: 0.999), 8)

    def test_retry_uses_injected_sleeper(self):
        delays = []
        outcomes = iter(["transient", "success"])
        receiver = IdempotentReceiver()
        boundary = ReliabilityBoundary(random_value=lambda: 0.25, sleeper=delays.append)
        boundary.execute("k", "p", lambda: receiver.refund("k", "p", next(outcomes)))
        self.assertEqual(delays, [0.25])

    def test_attempt_deadline_is_bounded(self):
        calls = []
        boundary = ReliabilityBoundary(max_attempts=1, attempt_timeout=2, duration_source=lambda _attempt: 3)
        boundary.execute("k", "p", lambda: calls.append("called"))
        self.assertEqual(calls, [])
        self.assertEqual(boundary.dead_letters[0].last_error, "attempt deadline exceeded")

    def test_malformed_output_repair_receives_feedback(self):
        seen = []
        repaired, feedback = repair_model_output(
            {"amount": "240"},
            lambda raw, error: seen.append(error) or {"amount": float(raw["amount"])},
        )
        self.assertEqual(repaired["amount"], 240.0)
        self.assertEqual(seen, [feedback])

    def test_breaker_transitions(self):
        breaker = CircuitBreaker(threshold=1, cooldown=10)
        breaker.failure(5); self.assertEqual(breaker.state, CircuitState.OPEN)
        self.assertFalse(breaker.allow(14)); self.assertTrue(breaker.allow(15))
        self.assertEqual(breaker.state, CircuitState.HALF_OPEN)
        breaker.success(); self.assertEqual(breaker.state, CircuitState.CLOSED)

    def test_dead_letter_contains_state(self):
        boundary = ReliabilityBoundary(max_attempts=1)
        boundary.execute("k", "payload", lambda: (_ for _ in ()).throw(OperationFailure(FailureKind.PERMANENT, "bad")))
        self.assertEqual((boundary.dead_letters[0].key, boundary.dead_letters[0].payload), ("k", "payload"))

    def test_replay_uses_original_key(self):
        boundary = ReliabilityBoundary(max_attempts=1)
        boundary.execute("k", "p", lambda: (_ for _ in ()).throw(OperationFailure(FailureKind.PERMANENT, "bad")))
        result = boundary.replay(boundary.dead_letters[0], lambda: __import__("resilient_refund_lab").AttemptResult("ok"), now=1)
        self.assertEqual(result.value, "ok")

    def test_correlated_failures_are_outside_independent_model(self):
        self.assertAlmostEqual(end_to_end_independent([0.99] * 6), 0.99 ** 6)
        self.assertNotEqual(end_to_end_independent([0.99] * 6), 0.99)

    def test_correlated_failures_trip_shared_breaker(self):
        boundary = ReliabilityBoundary(max_attempts=1, breaker=CircuitBreaker(threshold=2))
        fail = lambda: (_ for _ in ()).throw(OperationFailure(FailureKind.TRANSIENT, "shared outage"))
        boundary.execute("k1", "p", fail, now=1)
        boundary.execute("k2", "p", fail, now=2)
        boundary.execute("k3", "p", fail, now=3)
        self.assertEqual(boundary.breaker.state, CircuitState.OPEN)
        self.assertEqual(boundary.dead_letters[-1].attempts, 0)


if __name__ == "__main__":
    unittest.main()
