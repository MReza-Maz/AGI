import unittest

from cognition.agi_core import AGICore


class AGICoreTests(unittest.TestCase):
    def test_pursue_and_execute_hierarchical_action(self):
        core = AGICore()
        plan = core.pursue(
            "prepare report",
            subgoals=["collect data"],
            actions_by_subgoal={"subgoal_1": [{"name": "collect", "effects": {"data": "ready"}}]},
        )
        self.assertEqual(plan["goal"], "prepare report")
        self.assertEqual(core.agent.next_planned_action()["name"], "collect")
        self.assertTrue(core.agent.complete_planned_action("collect"))
        self.assertEqual(plan["plan"]["status"], "completed")

    def test_cycle_retains_integrated_state(self):
        core = AGICore()

        def callback(actions, result):
            return {"ok": True}

        result = core.cycle("observe environment", callback)
        self.assertIn("thought", result)
        self.assertEqual(result["action"]["result"], {"ok": True})
        state = core.state()
        self.assertEqual(state["mode"], "idle")
        self.assertIsNotNone(state["last_cycle"])


if __name__ == "__main__":
    unittest.main()
