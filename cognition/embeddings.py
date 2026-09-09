"""Deterministic trainable token embeddings without external dependencies."""
import math
from core.tensor import Tensor
from core.nn import Module


class Embedding(Module):
    def __init__(self, vocabulary_size, dimension, seed=1):
        self.weight = Tensor.random((vocabulary_size, dimension), scale=1.0 / math.sqrt(dimension), seed=seed, requires_grad=True)

    def __call__(self, token_ids):
        if not isinstance(token_ids, list):
            token_ids = [token_ids]
        rows = []
        for token_id in token_ids:
            index = int(token_id)
            if index < 0 or index >= self.weight.shape[0]:
                raise IndexError("token id out of range")
            rows.append(list(self.weight.data[index]))
        return Tensor(rows, requires_grad=self.weight.requires_grad, _parents=(self.weight,))
