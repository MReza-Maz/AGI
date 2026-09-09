"""Dependency-free scaled dot-product and multi-head self-attention."""
import math
from .tensor import Tensor
from .nn import Linear, Module


def softmax_rows(data):
    """Numerically stable row-wise softmax for ordinary Python lists."""
    result = []
    for row in data:
        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        total = sum(exponentials)
        result.append([value / total for value in exponentials])
    return result


class SelfAttention(Module):
    """Single-head scaled dot-product attention for [sequence, features]."""
    def __init__(self, size, seed=1, causal=False):
        self.q = Linear(size, size, seed)
        self.k = Linear(size, size, seed + 1)
        self.v = Linear(size, size, seed + 2)
        self.causal = bool(causal)

    def __call__(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scale = 1.0 / math.sqrt(q.shape[-1])
        raw_scores = (q @ k.transpose()) * scale
        scores = self._masked_scores(raw_scores)
        weights = self._softmax_tensor(scores)
        return weights @ v

    def _masked_scores(self, scores):
        if not self.causal:
            return scores
        data = [list(row) for row in scores.data]
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                data[i][j] = -1.0e9
        masked = Tensor(data, scores.requires_grad, (scores,))

        def backward():
            if scores.requires_grad:
                grad = [
                    [g if j <= i else 0.0 for j, g in enumerate(row)]
                    for i, row in enumerate(masked.grad)
                ]
                scores._accumulate(grad)
        masked._backward = backward
        return masked

    @staticmethod
    def _softmax_tensor(scores):
        weights = Tensor(softmax_rows(scores.data), scores.requires_grad, (scores,))

        def backward():
            if not scores.requires_grad:
                return
            grad_scores = []
            for row, grad_row in zip(weights.data, weights.grad):
                dot = sum(g * y for g, y in zip(grad_row, row))
                grad_scores.append([y * (g - dot) for y, g in zip(row, grad_row)])
            scores._accumulate(grad_scores)
        weights._backward = backward
        return weights


class MultiHeadSelfAttention(Module):
    """Multi-head self-attention using 2D tensors, causal masking, and output projection."""
    def __init__(self, size, heads=4, seed=1, causal=False):
        if size <= 0 or heads <= 0 or size % heads != 0:
            raise ValueError("model size must be positive and divisible by heads")
        self.size = int(size)
        self.heads = int(heads)
        self.head_size = self.size // self.heads
        self.causal = bool(causal)
        self.q = [Linear(size, self.head_size, seed + 3 * i) for i in range(heads)]
        self.k = [Linear(size, self.head_size, seed + 3 * i + 1) for i in range(heads)]
        self.v = [Linear(size, self.head_size, seed + 3 * i + 2) for i in range(heads)]
        self.output = Linear(size, size, seed + 3 * heads)

    def __call__(self, x):
        head_outputs = []
        for index in range(self.heads):
            q = self.q[index](x)
            k = self.k[index](x)
            v = self.v[index](x)
            scores = (q @ k.transpose()) * (1.0 / math.sqrt(self.head_size))
            scores = self._masked_scores(scores)
            weights = SelfAttention._softmax_tensor(scores)
            head_outputs.append(weights @ v)

        combined = self._concat_features(head_outputs)
        return self.output(combined)

    def _masked_scores(self, scores):
        if not self.causal:
            return scores
        data = [list(row) for row in scores.data]
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                data[i][j] = -1.0e9
        masked = Tensor(data, scores.requires_grad, (scores,))

        def backward():
            if scores.requires_grad:
                scores._accumulate([
                    [g if j <= i else 0.0 for j, g in enumerate(row)]
                    for i, row in enumerate(masked.grad)
                ])
        masked._backward = backward
        return masked

    @staticmethod
    def _concat_features(tensors):
        if not tensors:
            raise ValueError("at least one attention head is required")
        rows = len(tensors[0].data)
        data = [
            [value for tensor in tensors for value in tensor.data[row]]
            for row in range(rows)
        ]
        out = Tensor(data, any(t.requires_grad for t in tensors), tuple(tensors))

        def backward():
            for tensor in tensors:
                if not tensor.requires_grad:
                    continue
                width = len(tensor.data[0]) if tensor.data else 0
                offset = tensors.index(tensor)
                start = sum(len(t.data[0]) if t.data else 0 for t in tensors[:offset])
                tensor._accumulate([row[start:start + width] for row in out.grad])
        out._backward = backward
        return out
