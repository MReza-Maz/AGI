"""Unified cognitive core for open-ended AGI research experiments."""

from cognition.agent import CognitiveAgent
from cognition.autonomous_loop import AutonomousLoop


class AGICore:
    """Coordinate perception, planning, prediction, action, learning and replanning."""

    def __init__(self, agent=None, inference=None, max_steps=32):
        self.agent = agent or CognitiveAgent(inference=inference)
        self.autonomy = AutonomousLoop(self.agent, max_steps=max_steps)
        self.mode = "idle"
        self.last_cycle = None

    def observe(self, observation):
        """Convert an external observation into the agent's internal state."""
        self.mode = "observing"
        return self.agent.think(observation)

    def pursue(self, goal, subgoals=None, actions_by_subgoal=None):
        """Create an explicit multi-level plan for an open-ended goal."""
        self.mode = "planning"
        return self.autonomy.start(goal, subgoals, actions_by_subgoal)

    def decide(self, actions, goal=None):
        """Predict consequences and select the safest useful action."""
        self.mode = "deciding"
        return self.agent.select_action(actions, goal)

    def learn(self, thought, outcome):
        """Reflect on an outcome and persist the resulting experience."""
        self.mode = "learning"
        return self.agent.reflect(thought, outcome)

    def step(self, outcome=True, observation=None):
        """Advance the active autonomous plan by one bounded step."""
        self.mode = "active"
        result = self.autonomy.step(outcome, observation)
        if result["status"] in ("completed", "replan_required", "limit_reached"):
            self.mode = "idle"
        return result

    def run(self, executor=None, observer=None):
        """Run the active plan until completion, replanning is required, or limit."""
        self.mode = "active"
        result = self.autonomy.run(executor, observer)
        self.mode = "idle"
        return result

    def cycle(self, observation, action_callback=None):
        """Run one complete cognitive cycle and retain its result."""
        self.mode = "active"
        self.last_cycle = self.agent.run_cycle(observation, action_callback)
        self.mode = "idle"
        return self.last_cycle

    def state(self):
        """Return the current integrated cognitive state."""
        return {
            "mode": self.mode,
            "agent": self.agent.snapshot(),
            "autonomy": {
                "steps": self.autonomy.steps,
                "max_steps": self.autonomy.max_steps,
                "events": list(self.autonomy.events),
            },
            "last_cycle": self.last_cycle,
        }
