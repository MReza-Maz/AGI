"""Hierarchical planning from goals to sub-goals and executable actions."""


class HierarchicalPlanner:
    """Build a deterministic goal -> sub-goal -> action hierarchy."""

    def __init__(self, reasoning_engine=None):
        self.reasoning = reasoning_engine

    @staticmethod
    def _normalize_action(action):
        if isinstance(action, dict):
            return dict(action)
        return {"name": str(action), "effects": {}}

    def decompose(self, goal, subgoals=None):
        """Create explicit sub-goals while preserving caller-provided ordering."""
        text = str(goal).strip()
        if not text:
            raise ValueError("goal is required")
        if subgoals is None:
            return [{"id": "goal", "description": text, "status": "pending", "actions": []}]
        result = []
        for index, item in enumerate(subgoals, 1):
            if isinstance(item, dict):
                node = dict(item)
                node.setdefault("description", f"Sub-goal {index}")
            else:
                node = {"description": str(item)}
            node.setdefault("id", f"subgoal_{index}")
            node.setdefault("status", "pending")
            node.setdefault("actions", [])
            result.append(node)
        return result

    def plan(self, goal, subgoals=None, actions_by_subgoal=None):
        """Build a hierarchical plan without executing any side effects."""
        nodes = self.decompose(goal, subgoals)
        mapping = actions_by_subgoal or {}
        for node in nodes:
            actions = node.get("actions") or mapping.get(node["id"], [])
            node["actions"] = [self._normalize_action(action) for action in actions]
        return {
            "goal": str(goal).strip(),
            "subgoals": nodes,
            "current_subgoal": nodes[0]["id"] if nodes else None,
            "status": "planned",
        }

    @staticmethod
    def next_action(plan):
        """Return the first pending action of the current pending sub-goal."""
        for node in plan.get("subgoals", []):
            if node.get("status") == "completed":
                continue
            for action in node.get("actions", []):
                if action.get("status", "pending") != "completed":
                    return action
        return None

    @staticmethod
    def complete_action(plan, action_name, success=True):
        """Update action/sub-goal status after an execution result."""
        for node in plan.get("subgoals", []):
            for action in node.get("actions", []):
                if action.get("name") == action_name and action.get("status", "pending") != "completed":
                    action["status"] = "completed" if success else "failed"
                    if success and node.get("actions") and all(
                        item.get("status") == "completed" for item in node["actions"]
                    ):
                        node["status"] = "completed"
                    plan["current_subgoal"] = next(
                        (item["id"] for item in plan.get("subgoals", []) if item.get("status") != "completed"),
                        None,
                    )
                    plan["status"] = "completed" if plan["current_subgoal"] is None else "active"
                    return True
        return False
