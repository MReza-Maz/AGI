"""Command-line training entry point for the dependency-free causal language model."""
import argparse
import json
import math
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


def split_tokens(token_ids, validation_split, sequence_length):
    """Split a token stream while keeping both partitions valid datasets."""
    if not 0.0 <= validation_split < 1.0:
        raise ValueError("validation split must be between 0 and 1")
    if validation_split == 0.0:
        return token_ids, None
    split = int(len(token_ids) * (1.0 - validation_split))
    split = max(sequence_length + 1, min(split, len(token_ids) - sequence_length - 1))
    train_tokens = token_ids[:split]
    validation_tokens = token_ids[split - 1:]
    if len(train_tokens) < sequence_length + 1 or len(validation_tokens) < sequence_length + 1:
        raise ValueError("corpus is too short for the requested validation split")
    return train_tokens, validation_tokens


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
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--checkpoint", default="checkpoints/model.json")
    parser.add_argument("--tokenizer", default="checkpoints/tokenizer.json")
    parser.add_argument("--resume", action="store_true", help="Resume from the existing checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.sequence_length < 1:
        parser.error("--sequence-length must be positive")
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.heads < 1 or args.model_size < 1 or args.model_size % args.heads != 0:
        parser.error("--model-size must be positive and divisible by --heads")
    if not 0.0 <= args.validation_split < 1.0:
        parser.error("--validation-split must be between 0 and 1")

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
    token_ids = tokenizer.encode(text)
    try:
        train_tokens, validation_tokens = split_tokens(
            token_ids, args.validation_split, args.sequence_length
        )
        dataset = SequenceDataset(train_tokens, args.sequence_length)
        validation_dataset = (
            SequenceDataset(validation_tokens, args.sequence_length)
            if validation_tokens is not None else None
        )
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

    start_epoch = 0
    if args.resume:
        if not os.path.isfile(args.checkpoint):
            parser.error(f"checkpoint not found for --resume: {args.checkpoint}")
        try:
            metadata = trainer.load_checkpoint(args.checkpoint)
            start_epoch = int(metadata.get("completed_epochs", 0))
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            parser.error(f"could not resume checkpoint: {error}")

    print("Training corpus:", args.data)
    print("Characters:", len(text))
    print("Vocabulary size:", len(tokenizer))
    print("Training examples:", len(dataset))
    print("Parameters:", sum(parameter.numel() for parameter in model.parameters()))
    print("Validation split:", args.validation_split)
    if args.resume:
        print("Resuming from epoch:", start_epoch)

    records = trainer.fit(
        dataset,
        epochs=args.epochs,
        eval_dataset=validation_dataset,
        start_epoch=start_epoch,
    )
    for record in records:
        if "eval_loss" in record:
            print(
                "epoch={epoch} loss={loss:.6f} eval_loss={eval_loss:.6f} "
                "perplexity={perplexity:.4f}".format(**record)
            )
        else:
            print("epoch={epoch} loss={loss:.6f}".format(**record))

    final_record = records[-1]
    trainer.save_checkpoint(
        args.checkpoint,
        metadata={
            "format": "agi-causal-lm-v2",
            "data": os.path.abspath(args.data),
            "vocabulary_size": len(tokenizer),
            "sequence_length": args.sequence_length,
            "model_size": args.model_size,
            "layers": args.layers,
            "heads": args.heads,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "completed_epochs": final_record["epoch"],
            "final_loss": final_record["loss"],
            "final_eval_loss": final_record.get("eval_loss"),
            "final_perplexity": final_record.get(
                "perplexity", math.exp(min(final_record["loss"], 50.0))
            ),
        },
    )
    save_tokenizer(args.tokenizer, tokenizer)
    print("Checkpoint:", args.checkpoint)
    print("Tokenizer:", args.tokenizer)


if __name__ == "__main__":
    main()
