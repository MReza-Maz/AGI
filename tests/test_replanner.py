import unittest

from cognition.agent import CognitiveAgent
from cognition.replanner import DynamicReplanner


class TestDynamicReplanner(unittest.TestCase):
    def setUp(self):
        self.agent = CognitiveAgent()
        self.replanner = DynamicReplanner(self.agent)
        self.agent.world.update({"ready": True})

    def test_failure_selects_viable_alternative(self):
        failed = {"name": "primary", "effects": {"result": "bad"}, "risk": 0.1}
        alternative = {"name": "fallback", "effects": {"result": "good"}, "risk": 0.0}
        result = self.replanner.repair(
            failed,
            [failed, alternative],
            outcome=False,
            observation="primary failed",
        )
        self.assertTrue(result["replanned"])
        self.assertEqual(result["replacement"]["name"], "fallback")
        self.assertEqual(result["replacement"]["replanned_from"], "primary")

    def test_failed_action_is_excluded(self):
        failed = {"name": "primary", "effects": {"x": 1}}
        result = self.replanner.alternatives(failed, [failed], None)
        self.assertIsNone(result["selected"])

    def test_success_does_not_replan(self):
        action = {"name": "primary"}
        result = self.replanner.repair(action, [], outcome=True)
        self.assertFalse(result["replanned"])
        self.assertEqual(result["reason"], "action_succeeded")

    def test_no_alternative_is_reported(self):
        action = {"name": "primary"}
        result = self.replanner.repair(action, [action], outcome=False)
        self.assertFalse(result["replanned"])
        self.assertEqual(result["reason"], "no_viable_alternative")


if __name__ == "__main__":
    unittest.main()
