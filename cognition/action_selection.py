"""Predictive action selection for choosing the safest useful next action."""


class ActionSelector:
    """Rank candidate actions using prediction, goal alignment, risk and confidence."""

    def __init__(self, world):
        self.world = world

    @staticmethod
    def _clamp(value):
        return max(0.0, min(1.0, float(value)))

    def _risk(self, action, prediction):
        explicit = self._clamp(action.get("risk", 0.0))
        irreversible = 0.25 if action.get("irreversible", False) else 0.0
        impossible = 1.0 if not prediction.get("possible", False) else 0.0
        return self._clamp(max(explicit, irreversible) + impossible)

    def evaluate(self, action, goal=None):
        action = dict(action or {})
        prediction = self.world.predict(action)
        alignment = self.world.evaluate_prediction(prediction, goal)
        risk = self._risk(action, prediction)
        confidence = self._clamp(action.get("confidence", 1.0))
        expected = self._clamp((alignment.get("score", 0.0) + confidence) / 2.0)
        score = self._clamp(
            0.55 * alignment.get("score", 0.0)
            + 0.30 * expected
            + 0.15 * confidence
            - 0.35 * risk
        )
        return {
            "action": action,
            "prediction": prediction,
            "goal_alignment": alignment.get("score", 0.0),
            "risk": risk,
            "confidence": confidence,
            "expected_outcome": expected,
            "score": score,
            "possible": prediction.get("possible", False),
        }

    def rank(self, actions, goal=None):
        """Evaluate and deterministically rank candidate actions from best to worst."""
        evaluations = [self.evaluate(action, goal) for action in actions]
        evaluations.sort(
            key=lambda item: (
                item["possible"],
                item["score"],
                item["confidence"],
            ),
            reverse=True,
        )
        return evaluations

    def select(self, actions, goal=None):
        """Return the highest-scoring possible action, or None when no action is viable."""
        ranked = self.rank(actions, goal)
        for item in ranked:
            if item["possible"]:
                return item
        return None
