"""Generate text from a trained AGI causal language model checkpoint."""
import argparse
import os

from cognition.inference import InferenceEngine


def main():
    parser = argparse.ArgumentParser(description="Generate text from an AGI language model checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/model.json")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--beam-width", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-eos-stop", action="store_true")
    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must not be negative")
    if args.temperature <= 0.0:
        parser.error("--temperature must be positive")
    if args.top_k < 0:
        parser.error("--top-k must not be negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in (0, 1]")
    if args.beam_width < 0:
        parser.error("--beam-width must not be negative")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"checkpoint not found: {args.checkpoint}")
    if not os.path.isfile(args.tokenizer):
        parser.error(f"tokenizer not found: {args.tokenizer}")

    try:
        engine = InferenceEngine.from_files(args.checkpoint, args.tokenizer)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(f"could not load model files: {error}")

    try:
        prompt_ids = engine.tokenizer.encode(args.prompt)
        if not prompt_ids:
            parser.error("prompt must not be empty")
        eos_id = None if args.no_eos_stop else engine.tokenizer.token_to_id.get("<EOS>")
        if args.beam_width > 0:
            generated = engine.beam_search(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                beam_width=args.beam_width,
                eos_token_id=eos_id,
            )
        else:
            generated = engine.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                eos_token_id=eos_id,
                seed=args.seed,
            )
    except ValueError as error:
        parser.error(str(error))

    print(engine.tokenizer.decode(generated))


if __name__ == "__main__":
    main()
