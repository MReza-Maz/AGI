"""Deterministic trainable token embeddings without external dependencies."""
import math
from core.tensor import Tensor
from core.nn import Module


class Embedding(Module):
    """Lookup table with sparse gradient accumulation for selected token rows."""
    def __init__(self, vocabulary_size, dimension, seed=1):
        self.weight = Tensor.random(
            (vocabulary_size, dimension),
            scale=1.0 / math.sqrt(dimension),
            seed=seed,
            requires_grad=True,
        )

    def __call__(self, token_ids):
        if not isinstance(token_ids, list):
            token_ids = [token_ids]
        indices = []
        rows = []
        for token_id in token_ids:
            index = int(token_id)
            if index < 0 or index >= self.weight.shape[0]:
                raise IndexError("token id out of range")
            indices.append(index)
            rows.append(list(self.weight.data[index]))

        out = Tensor(rows, requires_grad=self.weight.requires_grad, _parents=(self.weight,))

        def backward():
            if not self.weight.requires_grad:
                return
            gradient = [[0.0] * self.weight.shape[1] for _ in range(self.weight.shape[0])]
            for output_row, index in zip(out.grad, indices):
                gradient[index] = Tensor._add_values(gradient[index], output_row)
            self.weight._accumulate(gradient)

        out._backward = backward
        return out
