"""Predictive symbolic world-state representation for planning and evaluation."""
from copy import deepcopy


class WorldState:
    def __init__(self, facts=None):
        self.facts = dict(facts or {})

    def get(self, key, default=None):
        return self.facts.get(key, default)

    def set(self, key, value):
        self.facts[key] = value

    def update(self, values):
        self.facts.update(values)

    def snapshot(self):
        return deepcopy(self.facts)

    def copy(self):
        return WorldState(self.snapshot())

    def satisfies(self, conditions):
        return all(self.facts.get(key) == value for key, value in conditions.items())

    def apply(self, effects):
        next_state = self.copy()
        next_state.update(effects)
        return next_state

    def predict(self, action, effects=None, preconditions=None):
        """Predict the next state without changing the current world."""
        action = dict(action or {})
        required = dict(preconditions or action.get("preconditions", {}))
        if required and not self.satisfies(required):
            return {
                "possible": False,
                "reason": "preconditions_not_satisfied",
                "state": self.snapshot(),
            }
        predicted = self.apply(effects if effects is not None else action.get("effects", {}))
        return {
            "possible": True,
            "action": action,
            "state": predicted.snapshot(),
            "changed": {
                key: predicted.get(key)
                for key in predicted.facts
                if predicted.get(key) != self.get(key)
            },
        }

    def evaluate_prediction(self, prediction, goal=None):
        """Score a predicted state against optional goal conditions."""
        if not prediction.get("possible"):
            return {"score": 0.0, "useful": False, "reason": prediction.get("reason")}
        state = prediction.get("state", {})
        conditions = dict(goal or {})
        if not conditions:
            return {"score": 1.0, "useful": True, "matched": 0, "total": 0}
        matched = sum(1 for key, value in conditions.items() if state.get(key) == value)
        total = len(conditions)
        score = matched / total if total else 1.0
        return {"score": score, "useful": score > 0.0, "matched": matched, "total": total}

    def __repr__(self):
        return f"WorldState({self.facts!r})"
