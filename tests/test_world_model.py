import unittest

from cognition.world_model import WorldState


class WorldModelTests(unittest.TestCase):
    def test_predict_does_not_mutate_current_state(self):
        world = WorldState({"connected": False})
        prediction = world.predict({"name": "connect", "effects": {"connected": True}})
        self.assertTrue(prediction["possible"])
        self.assertEqual(prediction["state"]["connected"], True)
        self.assertFalse(world.get("connected"))

    def test_preconditions_are_checked(self):
        world = WorldState({"authenticated": False})
        prediction = world.predict(
            {"name": "read", "effects": {"data": "ok"}},
            preconditions={"authenticated": True},
        )
        self.assertFalse(prediction["possible"])
        self.assertEqual(prediction["reason"], "preconditions_not_satisfied")

    def test_prediction_can_be_scored_against_goal(self):
        world = WorldState({"connected": False})
        prediction = world.predict({"name": "connect", "effects": {"connected": True}})
        evaluation = world.evaluate_prediction(prediction, {"connected": True})
        self.assertEqual(evaluation["score"], 1.0)
        self.assertTrue(evaluation["useful"])


if __name__ == "__main__":
    unittest.main()
