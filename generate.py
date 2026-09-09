"""Generate text from a trained AGI causal language model checkpoint."""
import argparse
import json
import math
import os
import random

from cognition.language_model import CausalLanguageModel
from cognition.tokenizer import ByteTokenizer
from learning.optimizer import Adam
from learning.trainer import Trainer


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sample_token(logits, temperature=1.0, top_k=0, seed=None):
    """Sample from temperature-scaled logits with optional top-k filtering."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    values = [float(value) / temperature for value in logits]
    if top_k > 0:
        top_k = min(int(top_k), len(values))
        allowed = sorted(range(len(values)), key=values.__getitem__, reverse=True)[:top_k]
    else:
        allowed = list(range(len(values)))
    maximum = max(values[index] for index in allowed)
    weights = [math.exp(values[index] - maximum) for index in allowed]
    total = sum(weights)
    if total <= 0.0:
        return allowed[0]
    rng = random.Random(seed) if seed is not None else random
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in zip(allowed, weights):
        cumulative += weight
        if cumulative >= threshold:
            return index
    return allowed[-1]


def main():
    parser = argparse.ArgumentParser(description="Generate text from an AGI language model checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/model.json")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must not be negative")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.top_k < 0:
        parser.error("--top-k must not be negative")

    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.tokenizer):
        parser.error(f"tokenizer not found: {args.tokenizer}")

    try:
        checkpoint = load_json(args.checkpoint)
        tokenizer = ByteTokenizer.from_dict(load_json(args.tokenizer))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(f"could not load model files: {error}")

    metadata = checkpoint.get("metadata", {})
    try:
        model_size = int(metadata["model_size"])
        layers = int(metadata["layers"])
        heads = int(metadata["heads"])
        sequence_length = int(metadata["sequence_length"])
        seed = int(metadata.get("seed", 42))
    except (KeyError, TypeError, ValueError) as error:
        parser.error(f"checkpoint metadata is incomplete: {error}")

    if int(metadata.get("vocabulary_size", len(tokenizer))) != len(tokenizer):
        parser.error("checkpoint and tokenizer vocabulary sizes do not match")

    model = CausalLanguageModel(
        len(tokenizer),
        model_size=model_size,
        layers=layers,
        heads=heads,
        seed=seed,
        max_sequence_length=sequence_length,
    )
    optimizer = Adam(model.parameters(), learning_rate=float(metadata.get("learning_rate", 0.001)))
    trainer = Trainer(model, optimizer, lambda predictions, targets: None)
    try:
        trainer.load_checkpoint(args.checkpoint)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
        parser.error(f"could not restore checkpoint: {error}")

    token_ids = tokenizer.encode(args.prompt)
    if not token_ids:
        parser.error("prompt must not be empty")

    generated = list(token_ids)
    rng = random.Random(args.seed) if args.seed is not None else random.Random()
    for _ in range(args.max_new_tokens):
        context = generated[-sequence_length:]
        logits = model.next_token_logits(context)
        next_id = sample_token(logits, args.temperature, args.top_k, seed=rng.random())
        generated.append(next_id)

    print(tokenizer.decode(generated))


if __name__ == "__main__":
    main()
