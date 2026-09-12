import os
import tempfile
import unittest

from memory.episodic import EpisodicMemory


class EpisodicMemoryTests(unittest.TestCase):
    def test_structured_episode_persists_and_can_be_recalled(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "episodes.json")
            memory = EpisodicMemory(path)
            memory.record(
                goal="research Fibonacci",
                observation="search results",
                action={"name": "open", "args": {"url": "https://example.test"}},
                outcome="success",
                reflection={"confidence": 0.9},
                success=True,
            )
            restored = EpisodicMemory(path)
            episodes = restored.search("Fibonacci success", limit=1)
            self.assertEqual(len(episodes), 1)
            self.assertEqual(episodes[0]["event"]["goal"], "research Fibonacci")
            self.assertTrue(episodes[0]["event"]["success"])

    def test_capacity_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = EpisodicMemory(os.path.join(directory, "episodes.json"), capacity=2)
            for index in range(3):
                memory.remember(f"event {index}")
            self.assertEqual(len(memory.recent(10)), 2)
            self.assertEqual(memory.recent(10)[0]["event"], "event 1")


if __name__ == "__main__":
    unittest.main()
