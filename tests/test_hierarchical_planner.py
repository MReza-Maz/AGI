import unittest

from cognition.hierarchical_planner import HierarchicalPlanner


class HierarchicalPlannerTests(unittest.TestCase):
    def test_builds_goal_hierarchy(self):
        planner = HierarchicalPlanner()
        plan = planner.plan(
            "prepare report",
            subgoals=["collect data", "write report"],
            actions_by_subgoal={
                "subgoal_1": [{"name": "collect", "effects": {"data": "ready"}}],
                "subgoal_2": [{"name": "write", "effects": {"report": "ready"}}],
            },
        )
        self.assertEqual(plan["goal"], "prepare report")
        self.assertEqual(len(plan["subgoals"]), 2)
        self.assertEqual(plan["current_subgoal"], "subgoal_1")
        self.assertEqual(planner.next_action(plan)["name"], "collect")

    def test_completed_subgoal_advances_to_next(self):
        planner = HierarchicalPlanner()
        plan = planner.plan(
            "prepare report",
            subgoals=["collect data", "write report"],
            actions_by_subgoal={
                "subgoal_1": [{"name": "collect"}],
                "subgoal_2": [{"name": "write"}],
            },
        )
        self.assertTrue(planner.complete_action(plan, "collect"))
        self.assertEqual(plan["current_subgoal"], "subgoal_2")
        self.assertEqual(planner.next_action(plan)["name"], "write")

    def test_plan_completes_after_all_actions(self):
        planner = HierarchicalPlanner()
        plan = planner.plan("task", subgoals=["step"], actions_by_subgoal={"subgoal_1": ["finish"]})
        self.assertTrue(planner.complete_action(plan, "finish"))
        self.assertEqual(plan["status"], "completed")
        self.assertIsNone(planner.next_action(plan))

    def test_failed_action_does_not_complete_subgoal(self):
        planner = HierarchicalPlanner()
        plan = planner.plan("task", subgoals=["step"], actions_by_subgoal={"subgoal_1": ["finish"]})
        self.assertTrue(planner.complete_action(plan, "finish", success=False))
        self.assertEqual(plan["subgoals"][0]["status"], "pending")
        self.assertEqual(plan["status"], "active")


if __name__ == "__main__":
    unittest.main()
