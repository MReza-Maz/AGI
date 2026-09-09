"""Dependency-free transformer components."""
import math
from .nn import Linear, MLP, Module
from .attention import SelfAttention
from .tensor import Tensor


class LayerNorm(Module):
    """Layer normalization over the feature dimension for 2D tensors."""
    def __init__(self, size, eps=1e-5):
        self.gamma = Tensor([[1.0] * size], requires_grad=True)
        self.beta = Tensor([[0.0] * size], requires_grad=True)
        self.eps = eps

    def __call__(self, x):
        if x.ndim != 2:
            raise ValueError("LayerNorm expects a 2D tensor")

        normalized_rows = []
        inverses = []
        for row in x.data:
            mean = sum(row) / len(row)
            variance = sum((v - mean) ** 2 for v in row) / len(row)
            inv = 1.0 / math.sqrt(variance + self.eps)
            inverses.append(inv)
            normalized_rows.append([(v - mean) * inv for v in row])

        normalized = Tensor(normalized_rows, x.requires_grad, (x,))

        def backward_normalization():
            if not x.requires_grad:
                return
            feature_count = len(x.data[0])
            gradient = []
            for y, upstream, inv in zip(normalized.data, normalized.grad, inverses):
                mean_grad = sum(upstream) / feature_count
                mean_grad_y = sum(g * value for g, value in zip(upstream, y)) / feature_count
                gradient.append([
                    inv * (g - mean_grad - value * mean_grad_y)
                    for g, value in zip(upstream, y)
                ])
            x._accumulate(gradient)

        normalized._backward = backward_normalization
        return normalized * self.gamma + self.beta


class PositionalEncoding:
    """Deterministic sinusoidal positional encoding."""
    def __init__(self, size, max_length=2048):
        self.size = size
        self.max_length = max_length

    def __call__(self, x):
        if x.ndim != 2 or x.shape[1] != self.size:
            raise ValueError("positional encoding expects [sequence, features]")
        if x.shape[0] > self.max_length:
            raise ValueError("sequence exceeds positional encoding limit")
        values = []
        for position in range(x.shape[0]):
            row = []
            for i in range(self.size):
                angle = position / (10000.0 ** (2.0 * (i // 2) / self.size))
                row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
            values.append(row)
        return x + Tensor(values)


class TransformerBlock(Module):
    """Pre-norm transformer block with residual attention and feed-forward paths."""
    def __init__(self, size, seed=1, ff_multiplier=4):
        self.norm1 = LayerNorm(size)
        self.attention = SelfAttention(size, seed)
        self.norm2 = LayerNorm(size)
        self.feed_forward = MLP([size, size * ff_multiplier, size], seed + 10)

    def __call__(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


class Transformer(Module):
    def __init__(self, size, layers=2, positional_encoding=True):
        self.position = PositionalEncoding(size) if positional_encoding else None
        self.blocks = [TransformerBlock(size, i + 1) for i in range(layers)]
        self.final_norm = LayerNorm(size)

    def __call__(self, x):
        if self.position is not None:
            x = self.position(x)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
