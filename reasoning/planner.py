class Planner:
    """Simple deterministic planner; later versions can learn planning policies."""
    def plan(self,goal):
        if isinstance(goal,str):
            return [{"step":f"Understand goal: {goal}"},{"step":"Generate candidate actions"},{"step":"Evaluate candidates"},{"step":"Execute safest useful action"},{"step":"Observe result and learn"}]
        return [{"step":"Inspect goal"},{"step":"Decompose goal"},{"step":"Evaluate plan"},{"step":"Execute"},{"step":"Reflect"}]
