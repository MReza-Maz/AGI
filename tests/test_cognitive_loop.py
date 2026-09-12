import unittest

from cognition.agent import CognitiveAgent


class CognitiveLoopTests(unittest.TestCase):
    def test_plan_creates_revision_and_world_state(self):
        agent = CognitiveAgent()
        result = agent.plan("calculate Fibonacci")

        self.assertEqual(result["goal"], "calculate Fibonacci")
        self.assertGreaterEqual(len(result["plan"]), 1)
        self.assertEqual(result["revision"], 1)
        self.assertEqual(agent.world.get("active_goal"), "calculate Fibonacci")
        self.assertEqual(agent.world.get("active_plan"), result["plan"])

    def test_plan_revision_increments_on_replan(self):
        agent = CognitiveAgent()
        first = agent.plan("open a website")
        second = agent.plan("open a different website")

        self.assertEqual(first["revision"], 1)
        self.assertEqual(second["revision"], 2)
        self.assertEqual(agent.world.get("active_goal"), "open a different website")

    def test_run_cycle_executes_callback_and_reflects(self):
        agent = CognitiveAgent()
        seen = []

        def callback(plan, result):
            seen.append((plan, result["turn"]))
            return {"ok": True}

        result = agent.run_cycle("observe the environment", callback)

        self.assertTrue(result["action"]["executed"])
        self.assertEqual(result["action"]["result"], {"ok": True})
        self.assertEqual(len(seen), 1)
        self.assertEqual(agent.turn, 1)
        self.assertIn("reflection", result)
        self.assertIsNotNone(agent.world.get("last_confidence"))


if __name__ == "__main__":
    unittest.main()
