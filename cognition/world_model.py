"""Explicit symbolic world-state representation for planning and evaluation."""
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

    def __repr__(self):
        return f"WorldState({self.facts!r})"
