"""Goal representation and deterministic priority management."""
from dataclasses import dataclass, field


@dataclass
class Goal:
    description: str
    priority: float = 1.0
    target: dict = field(default_factory=dict)
    status: str = "pending"

    def complete(self, state):
        return state.satisfies(self.target)


class GoalManager:
    def __init__(self):
        self.goals = []

    def add(self, goal):
        self.goals.append(goal)
        self.goals.sort(key=lambda item: item.priority, reverse=True)
        return goal

    def active(self):
        return [goal for goal in self.goals if goal.status == "pending"]

    def next_goal(self):
        active = self.active()
        return active[0] if active else None

    def refresh(self, state):
        for goal in self.goals:
            if goal.status == "pending" and goal.complete(state):
                goal.status = "completed"
        return self.active()
