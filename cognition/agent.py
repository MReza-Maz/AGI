"""Integrated cognitive runtime combining memory, goals, reasoning and optional language inference.

This module is an AGI research architecture, not a claim of artificial general intelligence.
It deliberately keeps external actions behind explicit application callbacks.
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


class CognitiveAgent:
    """Closed-loop research agent: perceive -> recall -> reason -> act -> reflect."""

    def __init__(self, config=None, inference=None):
        self.config = config or {}
        memory_cfg = self.config.get("memory", {})
        security_cfg = self.config.get("security", {})
        self.memory = MemoryManager(memory_cfg.get("working_capacity", 32))
        semantic_path = memory_cfg.get("semantic_path", "data/semantic.json")
        self.memory.semantic.path = semantic_path
        self.memory.semantic._load()
        self.reasoning = ReasoningEngine()
        self.goals = GoalManager()
        self.world = WorldState()
        self.improvement = SelfImprovementController(
            security_cfg.get("require_human_approval", True)
        )
        self.inference = inference
        self.turn = 0
        self.history = []

    @staticmethod
    def _embedding(text, dimensions=32):
        """Create a deterministic stdlib-only lexical hash embedding."""
        values = [0.0] * dimensions
        tokens = str(text).lower().split()
        if not tokens:
            return values
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for offset in range(8):
                index = digest[offset] % dimensions
                sign = 1.0 if digest[offset + 8] & 1 else -1.0
                values[index] += sign * (1.0 + digest[offset + 16] / 255.0)
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values] if norm else values

    def add_goal(self, description, priority=1.0, target=None):
        goal = Goal(description, float(priority), dict(target or {}))
        return self.goals.add(goal)

    def perceive(self, observation, metadata=None):
        """Store an observation in working, episodic and semantic memory."""
        vector = self._embedding(observation)
        self.memory.remember(observation, vector, metadata)
        self.world.set("last_observation", observation)
        self.world.set("turn", self.turn)
        return vector

    def recall(self, query, k=5):
        """Retrieve semantically similar observations from persistent memory."""
        vector = self._embedding(query)
        return self.memory.semantic.search(vector, k=int(k))

    def _context(self, observation, memories, goal):
        parts = ["Observation: " + str(observation)]
        if goal:
            parts.append("Goal: " + goal.description)
        if memories:
            parts.append("Relevant memory:")
            parts.extend("- " + item["text"] for item in memories)
        return "\n".join(parts)

    def think(self, observation, k=5):
        """Run one cognitive cycle without performing external side effects."""
        self.turn += 1
        self.perceive(observation, {"turn": self.turn})
        memories = self.recall(observation, k=k)
        goal = self.goals.next_goal()
        goal_text = goal.description if goal else observation
        reasoning = self.reasoning.reason(goal_text, {
            "observation": observation,
            "memory_count": len(memories),
            "turn": self.turn,
        })
        context = self._context(observation, memories, goal)
        result = {
            "turn": self.turn,
            "observation": observation,
            "memories": memories,
            "goal": asdict(goal) if goal else None,
            "reasoning": reasoning,
            "context": context,
        }
        if self.inference is not None:
            try:
                result["response"] = self.inference.generate_text(
                    context,
                    max_new_tokens=80,
                    temperature=0.8,
                    top_k=20,
                    top_p=0.9,
                )
            except (ValueError, IndexError, OverflowError) as error:
                result["response_error"] = str(error)
        self.history.append(result)
        return result

    def act(self, result, action_callback=None):
        """Execute a caller-supplied action only; never execute arbitrary model output."""
        if action_callback is None:
            return {"executed": False, "reason": "no action callback supplied"}
        action = result.get("reasoning", {}).get("plan", [])
        return {"executed": True, "result": action_callback(action, result)}

    def reflect(self, result, outcome=None):
        """Evaluate an observed outcome and update the symbolic world state."""
        if outcome is not None:
            self.world.set("last_outcome", outcome)
        goal = result.get("goal") or {}
        goal_text = goal.get("description", result.get("observation", ""))
        reflection = self.reasoning.reflection.evaluate(goal_text, outcome)
        self.world.set("last_confidence", reflection.get("confidence", 0.0))
        self.goals.refresh(self.world)
        return reflection

    def propose_improvements(self, metrics):
        """Run the guarded improvement pipeline; deployment remains approval-gated."""
        return self.improvement.propose(dict(metrics))

    def snapshot(self):
        return {
            "turn": self.turn,
            "world": self.world.snapshot(),
            "goals": [asdict(goal) for goal in self.goals.goals],
            "history_size": len(self.history),
        }

    def save_state(self, path="data/agent_state.json"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.snapshot(), handle, ensure_ascii=False, indent=2)
