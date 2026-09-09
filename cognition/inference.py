"""Standalone inference utilities for the AGI causal language model."""
import json
import math
import random

from cognition.language_model import CausalLanguageModel
from cognition.tokenizer import ByteTokenizer


class InferenceEngine:
    """Load a trained causal LM and generate tokens without the training stack."""
    def __init__(self, model, tokenizer, sequence_length):
        self.model = model
        self.tokenizer = tokenizer
        self.sequence_length = int(sequence_length)
        if self.sequence_length <= 0:
            raise ValueError("sequence length must be positive")

    @classmethod
    def from_files(cls, checkpoint_path, tokenizer_path):
        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            checkpoint = json.load(handle)
        with open(tokenizer_path, "r", encoding="utf-8") as handle:
            tokenizer = ByteTokenizer.from_dict(json.load(handle))

        metadata = checkpoint.get("metadata", {})
        try:
            model_size = int(metadata["model_size"])
            layers = int(metadata["layers"])
            heads = int(metadata["heads"])
            sequence_length = int(metadata["sequence_length"])
            seed = int(metadata.get("seed", 42))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"checkpoint metadata is incomplete: {error}") from error

        expected_vocab = int(metadata.get("vocabulary_size", len(tokenizer)))
        if expected_vocab != len(tokenizer):
            raise ValueError("checkpoint and tokenizer vocabulary sizes do not match")

        model = CausalLanguageModel(
            len(tokenizer),
            model_size=model_size,
            layers=layers,
            heads=heads,
            seed=seed,
            max_sequence_length=sequence_length,
        )
        values = checkpoint.get("parameters")
        parameters = model.parameters()
        if not isinstance(values, list) or len(values) != len(parameters):
            raise ValueError("checkpoint parameter count does not match model")
        for parameter, value in zip(parameters, values):
            if parameter.shape != cls._shape(value):
                raise ValueError("checkpoint parameter shape does not match model")
            parameter.data = value
        return cls(model, tokenizer, sequence_length)

    @staticmethod
    def _shape(value):
        shape = []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else []
        return tuple(shape)

    @staticmethod
    def _log_softmax(logits):
        maximum = max(logits)
        shifted = [float(value) - maximum for value in logits]
        total = sum(math.exp(value) for value in shifted)
        log_total = math.log(total)
        return [value - log_total for value in shifted]

    @staticmethod
    def _select_indices(logits, top_k=0, top_p=1.0):
        indices = list(range(len(logits)))
        if top_k > 0:
            indices = sorted(indices, key=logits.__getitem__, reverse=True)[:min(int(top_k), len(indices))]
        else:
            indices.sort(key=logits.__getitem__, reverse=True)
        if top_p < 1.0:
            if top_p <= 0.0:
                raise ValueError("top_p must be positive")
            maximum = max(logits[index] for index in indices)
            weights = [math.exp(logits[index] - maximum) for index in indices]
            total = sum(weights)
            cumulative = 0.0
            kept = []
            for index, weight in zip(indices, weights):
                kept.append(index)
                cumulative += weight / total
                if cumulative >= top_p:
                    break
            indices = kept
        return indices

    @classmethod
    def sample_token(cls, logits, temperature=1.0, top_k=0, top_p=1.0, rng=None):
        """Sample one token using temperature, top-k and nucleus filtering."""
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if top_k < 0:
            raise ValueError("top_k must not be negative")
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        scaled = [float(value) / temperature for value in logits]
        allowed = cls._select_indices(scaled, top_k=top_k, top_p=top_p)
        maximum = max(scaled[index] for index in allowed)
        weights = [math.exp(scaled[index] - maximum) for index in allowed]
        total = sum(weights)
        generator = rng if rng is not None else random
        threshold = generator.random() * total
        cumulative = 0.0
        for index, weight in zip(allowed, weights):
            cumulative += weight
            if cumulative >= threshold:
                return index
        return allowed[-1]

    def generate(self, token_ids, max_new_tokens=20, temperature=1.0, top_k=0,
                 top_p=1.0, eos_token_id=None, seed=None):
        """Generate tokens with deterministic optional sampling and EOS stopping."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        result = [int(token_id) for token_id in token_ids]
        if not result:
            raise ValueError("token sequence must not be empty")
        rng = random.Random(seed) if seed is not None else random
        for _ in range(int(max_new_tokens)):
            context = result[-self.sequence_length:]
            logits = self.model.next_token_logits(context)
            next_id = self.sample_token(logits, temperature, top_k, top_p, rng)
            result.append(next_id)
            if eos_token_id is not None and next_id == eos_token_id:
                break
        return result

    def beam_search(self, token_ids, max_new_tokens=20, beam_width=3, eos_token_id=None):
        """Generate using deterministic log-probability beam search."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must not be negative")
        if beam_width <= 0:
            raise ValueError("beam_width must be positive")
        initial = [int(token_id) for token_id in token_ids]
        if not initial:
            raise ValueError("token sequence must not be empty")
        beams = [(initial, 0.0)]
        for _ in range(int(max_new_tokens)):
            candidates = []
            all_finished = True
            for sequence, score in beams:
                if eos_token_id is not None and sequence[-1] == eos_token_id:
                    candidates.append((sequence, score))
                    continue
                all_finished = False
                logits = self.model.next_token_logits(sequence[-self.sequence_length:])
                log_probs = self._log_softmax(logits)
                best_ids = sorted(range(len(log_probs)), key=log_probs.__getitem__, reverse=True)[:beam_width]
                for token_id in best_ids:
                    candidates.append((sequence + [token_id], score + log_probs[token_id]))
            candidates.sort(key=lambda item: item[1], reverse=True)
            beams = candidates[:beam_width]
            if all_finished:
                break
        return beams[0][0]

    def generate_text(self, prompt, **kwargs):
        """Encode a prompt, generate, and decode the resulting text."""
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            raise ValueError("prompt must not be empty")
        result = self.generate(token_ids, **kwargs)
        return self.tokenizer.decode(result)
