"""Autonomous goal pursuit with observation, replanning and bounded execution."""


class AutonomousLoop:
    """Drive a hierarchical plan while reacting to action outcomes."""

    def __init__(self, agent, max_steps=32):
        self.agent = agent
        self.max_steps = max(1, int(max_steps))
        self.steps = 0
        self.events = []

    def reset(self):
        self.steps = 0
        self.events = []

    def start(self, goal, subgoals=None, actions_by_subgoal=None):
        """Create a plan and reset execution state without performing actions."""
        self.reset()
        return self.agent.hierarchical_plan(goal, subgoals, actions_by_subgoal)

    def step(self, outcome=True, observation=None):
        """Advance one action using the supplied execution outcome."""
        if self.steps >= self.max_steps:
            return {"status": "limit_reached", "steps": self.steps}

        action = self.agent.next_planned_action()
        if action is None:
            return {"status": "completed", "steps": self.steps}

        name = action.get("name", action.get("step", "action"))
        success = bool(outcome)
        self.agent.complete_planned_action(name, success)
        self.steps += 1
        event = {
            "step": self.steps,
            "action": action,
            "success": success,
            "observation": observation,
        }
        self.events.append(event)

        if success:
            status = "completed" if self.agent.next_planned_action() is None else "active"
        else:
            status = "replan_required"
        return {"status": status, "steps": self.steps, "event": event}

    def run(self, executor=None, observer=None):
        """Execute until completion, failure requiring replanning, or step limit."""
        while self.steps < self.max_steps:
            action = self.agent.next_planned_action()
            if action is None:
                return {"status": "completed", "steps": self.steps, "events": list(self.events)}

            observation = observer(action) if observer else None
            outcome = executor(action) if executor else True
            result = self.step(outcome, observation)
            if result["status"] != "active":
                return {"status": result["status"], "steps": self.steps, "events": list(self.events)}

        return {"status": "limit_reached", "steps": self.steps, "events": list(self.events)}
