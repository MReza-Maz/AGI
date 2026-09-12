"""Unified cognitive core for open-ended AGI research experiments."""

from cognition.agent import CognitiveAgent


class AGICore:
    """Coordinate perception, memory, planning, prediction, action and learning."""

    def __init__(self, agent=None, inference=None):
        self.agent = agent or CognitiveAgent(inference=inference)
        self.mode = "idle"
        self.last_cycle = None

    def observe(self, observation):
        """Convert an external observation into the agent's internal state."""
        self.mode = "observing"
        return self.agent.think(observation)

    def pursue(self, goal, subgoals=None, actions_by_subgoal=None):
        """Create an explicit multi-level plan for an open-ended goal."""
        self.mode = "planning"
        return self.agent.hierarchical_plan(goal, subgoals, actions_by_subgoal)

    def decide(self, actions, goal=None):
        """Predict consequences and select the safest useful action."""
        self.mode = "deciding"
        return self.agent.select_action(actions, goal)

    def learn(self, thought, outcome):
        """Reflect on an outcome and persist the resulting experience."""
        self.mode = "learning"
        return self.agent.reflect(thought, outcome)

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
            "last_cycle": self.last_cycle,
        }
