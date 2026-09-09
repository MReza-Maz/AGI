"""Integrated cognitive runtime with memory, reasoning, goals and tool use.

This module is an AGI research architecture, not a claim of artificial general intelligence.
External side effects remain behind explicit, permission-aware tools.
"""
import hashlib
import json
import math
import os
from dataclasses import asdict

from cognition.goals import Goal, GoalManager
from cognition.world_model import WorldState
from memory.manager import MemoryManager
from reasoning.engine import ReasoningEngine
from self_improvement.controller import SelfImprovementController
from tools.registry import ToolRegistry


class CognitiveAgent:
    """Closed loop: perceive -> recall -> reason -> tool action -> observe -> reflect."""

    def __init__(self, config=None, inference=None, tools=None):
        self.config = config or {}
        memory_cfg = self.config.get("memory", {})
        security_cfg = self.config.get("security", {})
        self.memory = MemoryManager(memory_cfg.get("working_capacity", 32))
        self.memory.semantic.path = memory_cfg.get("semantic_path", "data/semantic.json")
        self.memory.semantic._load()
        self.reasoning = ReasoningEngine()
        self.goals = GoalManager()
        self.world = WorldState()
        self.improvement = SelfImprovementController(security_cfg.get("require_human_approval", True))
        self.tools = tools or ToolRegistry(security_cfg.get("require_human_approval", True))
        self.inference = inference
        self.turn = 0
        self.history = []

    @staticmethod
    def _embedding(text, dimensions=32):
        values = [0.0] * dimensions
        for token in str(text).lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(8):
                index = digest[offset] % dimensions
                sign = 1.0 if digest[offset + 8] & 1 else -1.0
                values[index] += sign * (1.0 + digest[offset + 16] / 255.0)
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    def add_goal(self, description, priority=1.0, target=None):
        return self.goals.add(Goal(description, float(priority), dict(target or {})))

    def perceive(self, observation, metadata=None):
        vector = self._embedding(observation)
        self.memory.remember(observation, vector, metadata)
        self.world.set("last_observation", observation)
        self.world.set("turn", self.turn)
        return vector

    def recall(self, query, k=5):
        return self.memory.semantic.search(self._embedding(query), k=int(k))

    def _context(self, observation, memories, goal):
        parts = ["Observation: " + str(observation)]
        if goal:
            parts.append("Goal: " + goal.description)
        if memories:
            parts.append("Relevant memory:")
            parts.extend("- " + item["text"] for item in memories)
        return "\n".join(parts)

    def think(self, observation, k=5):
        self.turn += 1
        self.perceive(observation, {"turn": self.turn})
        memories = self.recall(observation, k=k)
        goal = self.goals.next_goal()
        reasoning = self.reasoning.reason(goal.description if goal else observation, {
            "observation": observation, "memory_count": len(memories), "turn": self.turn,
        })
        result = {"turn": self.turn, "observation": observation, "memories": memories,
                  "goal": asdict(goal) if goal else None, "reasoning": reasoning,
                  "context": self._context(observation, memories, goal)}
        if self.inference is not None:
            try:
                result["response"] = self.inference.generate_text(result["context"], max_new_tokens=80,
                    temperature=0.8, top_k=20, top_p=0.9)
            except (ValueError, IndexError, OverflowError) as error:
                result["response_error"] = str(error)
        self.history.append(result)
        return result

    def execute_tool(self, name, arguments=None, approved=False):
        """Execute a registered capability through the security boundary."""
        result = self.tools.execute(name, arguments, approved=approved)
        self.world.set("last_tool", name)
        self.world.set("last_tool_result", result)
        return result

    def act(self, result, action_callback=None):
        if action_callback is None:
            return {"executed": False, "reason": "no action callback supplied"}
        action = result.get("reasoning", {}).get("plan", [])
        return {"executed": True, "result": action_callback(action, result)}

    def reflect(self, result, outcome=None):
        if outcome is not None:
            self.world.set("last_outcome", outcome)
        goal = result.get("goal") or {}
        goal_text = goal.get("description", result.get("observation", ""))
        reflection = self.reasoning.reflection.evaluate(goal_text, outcome)
        self.world.set("last_confidence", reflection.get("confidence", 0.0))
        self.goals.refresh(self.world)
        return reflection

    def propose_improvements(self, metrics):
        return self.improvement.propose(dict(metrics))

    def snapshot(self):
        return {"turn": self.turn, "world": self.world.snapshot(),
                "goals": [asdict(goal) for goal in self.goals.goals],
                "history_size": len(self.history), "tools": self.tools.describe()}

    def save_state(self, path="data/agent_state.json"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.snapshot(), handle, ensure_ascii=False, indent=2)
