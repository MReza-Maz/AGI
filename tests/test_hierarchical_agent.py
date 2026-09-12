import unittest

from cognition.agent import CognitiveAgent


class HierarchicalAgentTests(unittest.TestCase):
    def test_hierarchical_plan_becomes_active(self):
        agent = CognitiveAgent()
        result = agent.hierarchical_plan(
            "prepare report",
            subgoals=["collect data", "write report"],
            actions_by_subgoal={
                "subgoal_1": [{"name": "collect"}],
                "subgoal_2": [{"name": "write"}],
            },
        )
        self.assertEqual(result["revision"], 1)
        self.assertEqual(agent.next_planned_action()["name"], "collect")
        self.assertEqual(agent.world.get("active_goal"), "prepare report")

    def test_completed_action_advances_hierarchy(self):
        agent = CognitiveAgent()
        agent.hierarchical_plan(
            "prepare report",
            subgoals=["collect data", "write report"],
            actions_by_subgoal={
                "subgoal_1": [{"name": "collect"}],
                "subgoal_2": [{"name": "write"}],
            },
        )
        self.assertTrue(agent.complete_planned_action("collect"))
        self.assertEqual(agent.next_planned_action()["name"], "write")


if __name__ == "__main__":
    unittest.main()
