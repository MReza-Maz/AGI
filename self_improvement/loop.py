"""Cognitive self-improvement loop.

This module turns a high-level improvement request into an evidence-based
improvement objective before delegating code changes to AutonomousCoder.
"""

from __future__ import annotations

from dataclasses import asdict

from self_improvement.autocoder import AutonomousCoder


class SelfImprovementLoop:
    """Analyze the agent state, build an improvement objective, and apply it."""

    def __init__(self, agent, coder=None):
        self.agent = agent
        self.coder = coder or AutonomousCoder()
        self.last_analysis = None
        self.last_result = None

    def analyze(self, request=None):
        """Build a compact diagnosis from the current cognitive state."""
        snapshot = self.agent.snapshot()
        world = snapshot.get("world", {})
        goals = snapshot.get("goals", [])

        weaknesses = []
        if snapshot.get("history_size", 0) == 0:
            weaknesses.append("There is not enough interaction history to evaluate learning behavior.")
        if snapshot.get("episodic_memory_size", 0) == 0:
            weaknesses.append("Episodic learning has no recorded experiences yet.")
        if not snapshot.get("tools"):
            weaknesses.append("The agent currently exposes no registered tools.")
        if world.get("last_confidence") is not None and float(world.get("last_confidence", 1.0)) < 0.5:
            weaknesses.append("Recent reasoning confidence is low.")
        if not goals:
            weaknesses.append("No persistent goals are currently active.")

        analysis = {
            "request": str(request or "Improve the AGI's weakest current capability."),
            "turn": snapshot.get("turn", 0),
            "history_size": snapshot.get("history_size", 0),
            "episodic_memory_size": snapshot.get("episodic_memory_size", 0),
            "active_goals": [goal for goal in goals if goal.get("status") == "pending"],
            "last_confidence": world.get("last_confidence"),
            "weaknesses": weaknesses,
        }
        self.last_analysis = analysis
        return analysis

    def build_objective(self, request=None):
        """Convert diagnosis into a concrete engineering objective."""
        analysis = self.analyze(request)
        weaknesses = analysis["weaknesses"]
        evidence = " ".join(weaknesses) if weaknesses else "No obvious runtime weakness was detected from the current state."
        objective = (
            f"{analysis['request']}\n"
            f"Current evidence: {evidence}\n"
            "Inspect the existing architecture and implement the smallest safe, testable improvement. "
            "Preserve existing APIs and do not weaken security or self-improvement boundaries."
        )
        return objective

    def improve(self, request=None):
        """Analyze, improve, and return the complete self-improvement result."""
        objective = self.build_objective(request)
        result = self.coder.improve(objective)
        self.last_result = result
        return {
            "ok": bool(result.get("ok")),
            "objective": objective,
            "analysis": self.last_analysis,
            "result": result,
        }

    def state(self):
        return {
            "analysis": self.last_analysis,
            "result": self.last_result,
        }
