"""Dynamic replanning from failed actions and new observations."""


class DynamicReplanner:
    """Repair an active plan without restarting the entire goal pursuit."""

    def __init__(self, agent):
        self.agent = agent
        self.replans = 0

    @staticmethod
    def _action_key(action):
        if isinstance(action, dict):
            return action.get("name", action.get("step", "action"))
        return str(action)

    def diagnose(self, action, outcome=None, observation=None):
        """Classify why an action should be reconsidered."""
        if outcome:
            return {"failure": False, "reason": "action_succeeded"}
        if observation:
            return {
                "failure": True,
                "reason": "execution_failed_with_observation",
                "observation": observation,
            }
        return {"failure": True, "reason": "execution_failed"}

    def alternatives(self, action, candidates, goal=None):
        """Rank alternative actions while excluding the failed action."""
        failed = self._action_key(action)
        remaining = [item for item in candidates if self._action_key(item) != failed]
        selection = self.agent.select_action(remaining, goal)
        return selection

    def repair(self, action, candidates, outcome=None, observation=None, goal=None):
        """Diagnose failure, select an alternative, and return a replacement plan."""
        diagnosis = self.diagnose(action, outcome, observation)
        if not diagnosis["failure"]:
            return {"replanned": False, "reason": diagnosis["reason"]}

        selection = self.alternatives(action, candidates, goal)
        chosen = selection.get("selected")
        self.replans += 1
        if chosen is None:
            return {
                "replanned": False,
                "reason": "no_viable_alternative",
                "diagnosis": diagnosis,
                "selection": selection,
                "replans": self.replans,
            }

        replacement = dict(chosen["action"])
        replacement["replanned_from"] = self._action_key(action)
        replacement["replan_reason"] = diagnosis["reason"]
        return {
            "replanned": True,
            "diagnosis": diagnosis,
            "replacement": replacement,
            "selection": selection,
            "replans": self.replans,
        }
