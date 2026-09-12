"""Run the autonomous AGI loop with a local Qwen3 Ollama model."""

import argparse
import json
import os

from cognition.agent import CognitiveAgent
from cognition.ollama import OllamaReasoner, OllamaError


SYSTEM_PROMPT = """You are the reasoning core of an experimental AGI system.
Your job is to pursue the user's goal, not merely answer the latest message.
Use the supplied memory, world state, available tools, and previous experience.
Separate facts from assumptions. Generate competing hypotheses when uncertainty
matters. Predict consequences before selecting an action. Prefer reversible and
low-risk actions. Never claim an action happened unless the runtime reports it.
Never invent tool results. If required information is missing, state exactly what
is needed. External side effects are controlled by the runtime and its approval
policy. The model proposes actions; the runtime decides whether they are allowed.
"""


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def context_for(agent, user_input, goal):
    memories = agent.recall(user_input, k=8)
    episodes = agent.recall_episodes(user_input, k=5)
    return json.dumps(
        {
            "user_input": user_input,
            "goal": goal,
            "world": agent.world.snapshot(),
            "working_memory": memories,
            "episodes": episodes,
            "available_tools": agent.tools.describe(),
            "history": agent.history[-4:],
        },
        ensure_ascii=False,
        indent=2,
    )


def print_decision(decision):
    print("\n=== AGI ===")
    print("Understanding:", decision.get("understanding", ""))
    if decision.get("hypotheses"):
        print("Hypotheses:")
        for item in decision["hypotheses"]:
            print("  -", item)
    if decision.get("plan"):
        print("Plan:")
        for index, item in enumerate(decision["plan"], 1):
            print(f"  {index}. {item}")
    action = decision.get("selected_action")
    if action:
        print("Selected action:", json.dumps(action, ensure_ascii=False))
    print("Confidence:", decision.get("confidence"))
    print("Uncertainty:", decision.get("uncertainty"))
    print("\n", decision.get("final_response", ""))


def main():
    parser = argparse.ArgumentParser(description="Autonomous AGI runtime")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", default=None)
    parser.add_argument("--goal", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--state", default="data/agi_state.json")
    args = parser.parse_args()

    config = load_json(args.config)
    llm_cfg = config.get("llm", {})
    model = args.model or llm_cfg.get("model", "qwen3:30b")
    endpoint = llm_cfg.get("endpoint", "http://127.0.0.1:11434/api/chat")
    timeout = llm_cfg.get("timeout", 120)
    reasoner = OllamaReasoner(model=model, endpoint=endpoint, timeout=timeout)
    agent = CognitiveAgent(config=config)

    goal = args.goal
    if not goal:
        goal = "General assistance: understand the user's request and accomplish it safely."
    agent.add_goal(goal)

    print(f"AGI runtime ready. Model: {model}")
    print("Type 'exit' to stop.")

    while True:
        try:
            user_input = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        try:
            observation = agent.think(user_input, k=8)
            context = context_for(agent, user_input, goal)
            decision = reasoner.reason(SYSTEM_PROMPT, context)
            observation["llm_decision"] = decision
            agent.world.update(
                {
                    "last_understanding": decision.get("understanding"),
                    "last_hypotheses": decision.get("hypotheses", []),
                    "last_plan": decision.get("plan", []),
                    "last_confidence": decision.get("confidence", 0.0),
                    "last_uncertainty": decision.get("uncertainty"),
                }
            )
            print_decision(decision)
            agent.episodic.record(
                goal=goal,
                observation=user_input,
                action=decision.get("selected_action"),
                outcome={"response": decision.get("final_response", "")},
                reflection={
                    "confidence": decision.get("confidence", 0.0),
                    "uncertainty": decision.get("uncertainty"),
                },
                success=True,
                metadata={"model": model, "turn": agent.turn},
            )
        except OllamaError as error:
            print("\nAGI model error:", error)

    agent.save_state(args.state)
    print("State saved:", args.state)


if __name__ == "__main__":
    main()
