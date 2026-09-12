import unittest

from cognition.agent import CognitiveAgent
from cognition.autonomous_loop import AutonomousLoop


class TestAutonomousLoop(unittest.TestCase):
    def test_runs_hierarchical_actions_to_completion(self):
        agent = CognitiveAgent()
        loop = AutonomousLoop(agent, max_steps=5)
        loop.start("build system", subgoals=["prepare", "finish"], actions_by_subgoal={
            "subgoal_1": [{"name": "prepare", "effects": {"prepared": True}}],
            "subgoal_2": [{"name": "finish", "effects": {"done": True}}],
        })
        result = loop.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["steps"], 2)

    def test_failure_stops_for_replanning(self):
        agent = CognitiveAgent()
        loop = AutonomousLoop(agent, max_steps=5)
        loop.start("goal", subgoals=["work"], actions_by_subgoal={
            "subgoal_1": [{"name": "work"}],
        })
        result = loop.step(False)
        self.assertEqual(result["status"], "replan_required")
        self.assertEqual(loop.steps, 1)
        self.assertIsNotNone(agent.next_planned_action())

    def test_step_limit_is_enforced(self):
        agent = CognitiveAgent()
        loop = AutonomousLoop(agent, max_steps=1)
        loop.start("goal", subgoals=["work", "more"], actions_by_subgoal={
            "subgoal_1": [{"name": "work"}],
            "subgoal_2": [{"name": "more"}],
        })
        result = loop.run()
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(result["steps"], 1)


if __name__ == "__main__":
    unittest.main()
