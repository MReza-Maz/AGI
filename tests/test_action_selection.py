import unittest

from cognition.action_selection import ActionSelector
from cognition.world_model import WorldState


class TestActionSelection(unittest.TestCase):
    def setUp(self):
        self.world = WorldState({"power": "off", "network": "ready"})
        self.selector = ActionSelector(self.world)

    def test_selects_action_aligned_with_goal(self):
        actions = [
            {"name": "wait", "effects": {"power": "off"}, "confidence": 0.9},
            {"name": "start", "effects": {"power": "on"}, "confidence": 0.9},
        ]
        selected = self.selector.select(actions, {"power": "on"})
        self.assertIsNotNone(selected)
        self.assertEqual(selected["action"]["name"], "start")
        self.assertGreater(selected["goal_alignment"], 0.0)

    def test_rejects_action_with_unsatisfied_preconditions(self):
        actions = [
            {
                "name": "unsafe_start",
                "preconditions": {"power": "on"},
                "effects": {"network": "down"},
            },
            {"name": "safe_start", "effects": {"power": "on"}},
        ]
        selected = self.selector.select(actions, {"power": "on"})
        self.assertEqual(selected["action"]["name"], "safe_start")

    def test_risk_reduces_score(self):
        safe = self.selector.evaluate(
            {"name": "safe", "effects": {"power": "on"}, "confidence": 0.9},
            {"power": "on"},
        )
        risky = self.selector.evaluate(
            {
                "name": "risky",
                "effects": {"power": "on"},
                "confidence": 0.9,
                "risk": 0.9,
            },
            {"power": "on"},
        )
        self.assertLess(risky["score"], safe["score"])

    def test_selection_does_not_mutate_world(self):
        before = self.world.snapshot()
        self.selector.select(
            [{"name": "start", "effects": {"power": "on"}}],
            {"power": "on"},
        )
        self.assertEqual(self.world.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
