"""Command-line training entry point for the dependency-free causal language model."""
import argparse
import json
import os

from cognition.language_model import CausalLanguageModel
from cognition.tokenizer import ByteTokenizer
from learning.loss import cross_entropy
from learning.optimizer import Adam
from learning.sequence import SequenceDataset
from learning.trainer import Trainer


def read_text(path):
    """Read a UTF-8 training corpus from disk."""
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def build_dataset(text, tokenizer, sequence_length):
    """Encode text and build overlapping next-token training examples."""
    token_ids = tokenizer.encode(text)
    return SequenceDataset(token_ids, sequence_length)


def save_tokenizer(path, tokenizer):
    """Save tokenizer configuration for reproducible generation."""
    payload = tokenizer.to_dict()
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description="Train the AGI causal language model.")
    parser.add_argument("--data", required=True, help="UTF-8 training text file")
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--model-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--checkpoint", default="checkpoints/model.json")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sequence_length < 1:
        parser.error("--sequence-length must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.model_size < 1 or args.model_size % args.heads != 0:
        parser.error("--model-size must be positive and divisible by --heads")

    try:
        text = read_text(args.data)
    except FileNotFoundError:
        parser.error(
            f"training corpus not found: {args.data}. "
            "Create a UTF-8 text file and pass it with --data."
        )
    except UnicodeDecodeError:
        parser.error(f"training corpus is not valid UTF-8: {args.data}")

    if not text:
        parser.error("training corpus must not be empty")

    tokenizer = ByteTokenizer()
    try:
        dataset = build_dataset(text, tokenizer, args.sequence_length)
    except ValueError as error:
        parser.error(str(error))

    model = CausalLanguageModel(
        len(tokenizer),
        model_size=args.model_size,
        layers=args.layers,
        heads=args.heads,
        seed=args.seed,
        max_sequence_length=args.sequence_length,
    )
    optimizer = Adam(model.parameters(), learning_rate=args.learning_rate)
    trainer = Trainer(model, optimizer, cross_entropy, clip_norm=1.0)

    print("Training corpus:", args.data)
    print("Characters:", len(text))
    print("Vocabulary size:", len(tokenizer))
    print("Training examples:", len(dataset))
    print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))

    records = trainer.fit(dataset, epochs=args.epochs)
    for record in records:
        print("epoch={epoch} loss={loss:.6f}".format(**record))

    trainer.save_checkpoint(
        args.checkpoint,
        metadata={
            "format": "agi-causal-lm-v1",
            "data": os.path.abspath(args.data),
            "vocabulary_size": len(tokenizer),
            "sequence_length": args.sequence_length,
            "model_size": args.model_size,
            "layers": args.layers,
            "heads": args.heads,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "epochs": args.epochs,
            "final_loss": records[-1]["loss"],
        },
    )
    save_tokenizer(args.tokenizer, tokenizer)
    print("Checkpoint:", args.checkpoint)
    print("Tokenizer:", args.tokenizer)


if __name__ == "__main__":
    main()
