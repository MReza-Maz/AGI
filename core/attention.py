"""Dependency-free scaled dot-product self-attention."""
import math
from .tensor import Tensor
from .nn import Linear, Module


def softmax_rows(data):
    result = []
    for row in data:
        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        total = sum(exponentials)
        result.append([value / total for value in exponentials])
    return result


class SelfAttention(Module):
    """Single-head scaled dot-product attention for 2D [sequence, features] input."""

    def __init__(self, size, seed=1):
        self.q = Linear(size, size, seed)
        self.k = Linear(size, size, seed + 1)
        self.v = Linear(size, size, seed + 2)

    def __call__(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        scale = 1.0 / math.sqrt(q.shape[-1])
        scores = (q @ k.transpose()) * scale

        weights = Tensor(softmax_rows(scores.data), scores.requires_grad, (scores,))

        def backward_softmax():
            if not scores.requires_grad:
                return
            grad_scores = []
            for row, grad_row in zip(weights.data, weights.grad):
                dot = sum(g * y for g, y in zip(grad_row, row))
                grad_scores.append([y * (g - dot) for y, g in zip(row, grad_row)])
            scores._accumulate(grad_scores)

        weights._backward = backward_softmax
        return weights @ v
