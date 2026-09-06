import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_lifecycle_lab import CheckpointLeaseStore, build_demo_graph, migrate_v1_to_v2, run_checkpoint_demo, trim_state
from langgraph.checkpoint.memory import InMemorySaver


class CheckpointLifecycleTests(unittest.TestCase):
    def test_checkpoint_history(self):
        _, result, history = run_checkpoint_demo("a")
        self.assertEqual(result["doubled"], 2)
        self.assertGreaterEqual(len(history), 3)

    def test_thread_isolation(self):
        saver = InMemorySaver(); graph = build_demo_graph(saver)
        for thread, value in (("a", 1), ("b", 5)):
            graph.invoke({"value": value}, config={"configurable": {"thread_id": thread}})
        self.assertNotEqual(graph.get_state({"configurable": {"thread_id": "a"}}).values, graph.get_state({"configurable": {"thread_id": "b"}}).values)

    def test_same_thread_continuation(self):
        saver = InMemorySaver(); graph = build_demo_graph(saver); config = {"configurable": {"thread_id": "a"}}
        graph.invoke({"value": 1}, config=config)
        result = graph.invoke({"value": 10}, config=config)
        self.assertEqual(result["doubled"], 22)

    def test_only_one_worker_wins_lease(self):
        store = CheckpointLeaseStore(); store.create("a", {})
        barrier = Barrier(2)

        def acquire(owner):
            barrier.wait()
            try:
                store.acquire("a", owner, 1, now=0)
                return "won"
            except RuntimeError:
                return "lost"

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(acquire, ("w1", "w2")))

        self.assertCountEqual(outcomes, ["won", "lost"])

    def test_expired_lease_takeover(self):
        store = CheckpointLeaseStore(); store.create("a", {})
        store.acquire("a", "w1", 1, now=0, ttl=2)
        self.assertEqual(store.acquire("a", "w2", 1, now=2).lease_owner, "w2")

    def test_stale_version_rejected(self):
        store = CheckpointLeaseStore(); store.create("a", {})
        with self.assertRaises(ValueError): store.acquire("a", "w", 0, now=0)

    def test_terminal_state_protected(self):
        store = CheckpointLeaseStore(); store.create("a", {})
        store.acquire("a", "w", 1, now=0); store.complete("a", "w", {}, terminal=True)
        with self.assertRaises(ValueError): store.acquire("a", "w2", 2, now=2)

    def test_schema_migration(self):
        self.assertEqual(migrate_v1_to_v2({"request": "hello"}), {"messages": ["hello"], "schema_version": 2})

    def test_state_trimming(self):
        trimmed = trim_state({"intent": "refund", "raw_tool_output": {}, "ssn": "x", "email": "x@y"})
        self.assertEqual(trimmed, {"intent": "refund"})


if __name__ == "__main__":
    unittest.main()
