"""Dependency-free causal language model built from the AGI tensor stack."""
import math

from core.nn import Linear, Module
from core.transformer import PositionalEncoding, Transformer
from cognition.embeddings import Embedding


class CausalLanguageModel(Module):
    """Small autoregressive language model for sequence-level experiments."""
    def __init__(self, vocabulary_size, model_size=32, layers=2, heads=4, seed=1, max_sequence_length=256):
        if vocabulary_size <= 0:
            raise ValueError("vocabulary size must be positive")
        if model_size <= 0 or model_size % heads != 0:
            raise ValueError("model size must be positive and divisible by heads")
        self.vocabulary_size = int(vocabulary_size)
        self.model_size = int(model_size)
        self.max_sequence_length = int(max_sequence_length)
        self.embedding = Embedding(self.vocabulary_size, self.model_size, seed=seed)
        self.position = PositionalEncoding(self.model_size, max_length=self.max_sequence_length)
        self.transformer = Transformer(
            self.model_size,
            layers=layers,
            heads=heads,
            causal=True,
            seed=seed + 100,
        )
        self.lm_head = Linear(self.model_size, self.vocabulary_size, seed=seed + 1000)

    def __call__(self, token_ids):
        if not token_ids:
            raise ValueError("token sequence must not be empty")
        if len(token_ids) > self.max_sequence_length:
            raise ValueError("token sequence exceeds max_sequence_length")
        hidden = self.embedding(token_ids)
        hidden = self.position(hidden)
        hidden = self.transformer(hidden)
        return self.lm_head(hidden)

    def next_token_logits(self, token_ids):
        """Return logits for the next token after the supplied context."""
        return self(token_ids).data[-1]

    def generate(self, token_ids, max_new_tokens=20, temperature=1.0):
        """Greedy/temperature sampling using the standard library only."""
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        result = [int(token_id) for token_id in token_ids]
        for _ in range(int(max_new_tokens)):
            context = result[-self.max_sequence_length:]
            logits = self.next_token_logits(context)
            scaled = [value / temperature for value in logits]
            maximum = max(scaled)
            probabilities = [math.exp(value - maximum) for value in scaled]
            total = sum(probabilities)
            probabilities = [value / total for value in probabilities]
            next_id = max(range(len(probabilities)), key=probabilities.__getitem__)
            result.append(next_id)
        return result
