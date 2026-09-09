"""Permission-aware registry for deterministic tool invocation."""
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ToolSpec:
    name: str
    handler: Callable[[dict], Any]
    description: str = ""
    dangerous: bool = False


class ToolRegistry:
    def __init__(self, require_approval=True):
        self.require_approval = bool(require_approval)
        self._tools = {}

    def register(self, name, handler, description="", dangerous=False):
        if not name or not callable(handler):
            raise ValueError("tool name and callable handler are required")
        if name in self._tools:
            raise ValueError(f"tool already registered: {name}")
        self._tools[name] = ToolSpec(name, handler, description, bool(dangerous))

    def get(self, name):
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def describe(self):
        return [
            {"name": s.name, "description": s.description, "dangerous": s.dangerous}
            for s in self._tools.values()
        ]

    def execute(self, name, arguments=None, approved=False):
        spec = self.get(name)
        if spec.dangerous and self.require_approval and not approved:
            return {"executed": False, "blocked": True, "reason": "approval required", "tool": name}
        args = dict(arguments or {})
        result = spec.handler(args)
        return {"executed": True, "blocked": False, "tool": name, "result": result}
