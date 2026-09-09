"""Dependency-free differentiable loss functions."""
import math
from core.tensor import Tensor


def mse(prediction, target):
    """Mean squared error."""
    target = target if isinstance(target, Tensor) else Tensor(target)
    diff = prediction - target
    return (diff * diff).mean()


def cross_entropy(logits, targets):
    """Stable softmax cross-entropy for 2D [batch, classes] logits.

    Targets are integer class indices. The loss is differentiable with respect
    to logits and uses log-sum-exp for numerical stability.
    """
    if logits.ndim != 2:
        raise ValueError("cross_entropy expects a 2D logits tensor")
    if not isinstance(targets, list) or len(targets) != logits.shape[0]:
        raise ValueError("targets must contain one class index per row")
    losses = []
    probabilities = []
    for row, target in zip(logits.data, targets):
        target = int(target)
        if target < 0 or target >= len(row):
            raise IndexError("target class out of range")
        maximum = max(row)
        exp_values = [math.exp(value - maximum) for value in row]
        total = sum(exp_values)
        probabilities.append([value / total for value in exp_values])
        losses.append(-(row[target] - maximum - math.log(total)))

    out = Tensor(sum(losses) / len(losses), logits.requires_grad, (logits,))

    def backward():
        if not logits.requires_grad:
            return
        gradient = []
        scale = 1.0 / len(targets)
        for row, probability, target in zip(logits.data, probabilities, targets):
            target = int(target)
            gradient.append([scale * (p - (1.0 if i == target else 0.0))
                             for i, p in enumerate(probability)])
        logits._accumulate(gradient)

    out._backward = backward
    return out
