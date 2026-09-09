"""Run the integrated AGI research cognitive loop from the command line."""
import argparse
import json
import os

from cognition.agent import CognitiveAgent
from cognition.inference import InferenceEngine


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_inference(args):
    if not args.checkpoint or not args.tokenizer:
        return None
    if not os.path.exists(args.checkpoint) or not os.path.exists(args.tokenizer):
        raise SystemExit("checkpoint/tokenizer not found; run without --checkpoint for symbolic mode")
    return InferenceEngine.from_files(args.checkpoint, args.tokenizer)


def print_cycle(result):
    print("\n=== COGNITIVE CYCLE ===")
    print("Turn:", result["turn"])
    print("Observation:", result["observation"])
    goal = result.get("goal")
    if goal:
        print("Goal:", goal["description"])
    print("Memory matches:", len(result["memories"]))
    print("Plan:")
    for index, step in enumerate(result["reasoning"]["plan"], 1):
        print(f"  {index}. {step}")
    reflection = result["reasoning"]["reflection"]
    print("Confidence:", reflection.get("confidence"))
    if "response" in result:
        print("Model response:", result["response"])
    if "response_error" in result:
        print("Model response error:", result["response_error"])


def main():
    parser = argparse.ArgumentParser(description="Integrated AGI research cognitive runtime")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--goal", action="append", default=[])
    parser.add_argument("--observation", action="append", default=[])
    parser.add_argument("--memory-k", type=int, default=5)
    parser.add_argument("--state", default="data/agent_state.json")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    config = load_json(args.config)
    agent = CognitiveAgent(config=config, inference=build_inference(args))
    for goal in args.goal:
        agent.add_goal(goal)

    observations = list(args.observation)
    if args.interactive or not observations:
        print("Integrated cognitive runtime ready. Type 'exit' to stop.")
        while True:
            try:
                observation = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if observation.lower() in {"exit", "quit"}:
                break
            if not observation:
                continue
            print_cycle(agent.think(observation, k=args.memory_k))
    else:
        for observation in observations:
            print_cycle(agent.think(observation, k=args.memory_k))

    agent.save_state(args.state)
    print("State saved:", args.state)


if __name__ == "__main__":
    main()
