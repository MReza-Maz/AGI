"""Small dependency-free tensor with reverse-mode automatic differentiation."""
import random


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self.data = self._copy(data)
        self.requires_grad = requires_grad
        self.grad = self._zeros_like(self.data) if requires_grad else None
        self._parents = tuple(_parents)
        self._backward = _backward or (lambda: None)

    @staticmethod
    def _copy(x):
        return [Tensor._copy(v) for v in x] if isinstance(x, list) else float(x)

    @staticmethod
    def _zeros_like(x):
        return [Tensor._zeros_like(v) for v in x] if isinstance(x, list) else 0.0

    @property
    def shape(self):
        x = self.data
        result = []
        while isinstance(x, list):
            result.append(len(x))
            x = x[0] if x else []
        return tuple(result)

    @property
    def ndim(self):
        return len(self.shape)

    def clone(self):
        return Tensor(self.data, self.requires_grad)

    def zero_grad(self):
        if self.requires_grad:
            self.grad = self._zeros_like(self.data)

    def flatten(self):
        def walk(x):
            if isinstance(x, list):
                result = []
                for value in x:
                    result.extend(walk(value))
                return result
            return [x]
        return walk(self.data)

    def numel(self):
        return len(self.flatten())

    @classmethod
    def zeros(cls, shape, requires_grad=False):
        shape = tuple(shape)
        def build(s):
            return [build(s[1:]) for _ in range(s[0])] if s else 0.0
        return cls(build(shape), requires_grad)

    @classmethod
    def random(cls, shape, scale=0.02, seed=None, requires_grad=False):
        rng = random.Random(seed) if seed is not None else random
        shape = tuple(shape)
        def build(s):
            return [build(s[1:]) for _ in range(s[0])] if s else rng.uniform(-scale, scale)
        return cls(build(shape), requires_grad)

    @staticmethod
    def _add_values(a, b):
        if not isinstance(a, list) and not isinstance(b, list):
            return a + b
        if not isinstance(a, list):
            return [Tensor._add_values(a, x) for x in b]
        if not isinstance(b, list):
            return [Tensor._add_values(x, b) for x in a]
        if len(b) == 1 and len(a) != 1 and isinstance(a[0], list):
            return [Tensor._add_values(x, b[0]) for x in a]
        if len(a) == 1 and len(b) != 1 and isinstance(b[0], list):
            return [Tensor._add_values(a[0], x) for x in b]
        if len(a) != len(b):
            raise ValueError("incompatible shapes for addition")
        return [Tensor._add_values(x, y) for x, y in zip(a, b)]

    @staticmethod
    def _mul_values(a, b):
        if not isinstance(a, list) and not isinstance(b, list):
            return a * b
        if not isinstance(a, list):
            return [Tensor._mul_values(a, x) for x in b]
        if not isinstance(b, list):
            return [Tensor._mul_values(x, b) for x in a]
        # NumPy-style feature broadcasting: [N,M] * [M].
        if isinstance(a[0], list) and not isinstance(b[0], list) and len(a[0]) == len(b):
            return [Tensor._mul_values(row, b) for row in a]
        if not isinstance(a[0], list) and isinstance(b[0], list) and len(a) == len(b[0]):
            return [Tensor._mul_values(a, row) for row in b]
        if len(a) == 1 and len(b) != 1:
            return [Tensor._mul_values(a[0], x) for x in b]
        if len(b) == 1 and len(a) != 1:
            return [Tensor._mul_values(x, b[0]) for x in a]
        if len(a) != len(b):
            raise ValueError("incompatible shapes for multiplication")
        return [Tensor._mul_values(x, y) for x, y in zip(a, b)]

    @staticmethod
    def _scale(value, scalar):
        if isinstance(value, list):
            return [Tensor._scale(x, scalar) for x in value]
        return value * scalar

    def _accumulate(self, value):
        self.grad = self._add_values(self.grad, value)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._add_values(self.data, other.data), self.requires_grad or other.requires_grad, (self, other))
        def backward():
            if self.requires_grad:
                self._accumulate(out.grad)
            if other.requires_grad:
                grad = out.grad
                if other.ndim > 0 and other.shape != out.shape:
                    grad = self._reduce_broadcast(grad, other.data)
                other._accumulate(grad)
        out._backward = backward
        return out

    __radd__ = __add__

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -float(other))

    def __rsub__(self, other):
        return (-self) + other

    @staticmethod
    def _sum_values(value):
        if isinstance(value, list):
            return sum(Tensor._sum_values(x) for x in value)
        return value

    @staticmethod
    def _reduce_broadcast(grad, target):
        if not isinstance(target, list):
            return Tensor._sum_values(grad)
        if len(target) == 1 and isinstance(target[0], list) and isinstance(grad, list):
            rows = [Tensor._reduce_broadcast(g, target[0]) for g in grad]
            result = rows[0] if rows else Tensor._zeros_like(target[0])
            for row in rows[1:]:
                result = Tensor._add_values(result, row)
            return [result]
        if not isinstance(target[0], list) and isinstance(grad, list) and len(target) == len(grad[0]):
            result = [0.0] * len(target)
            for row in grad:
                result = Tensor._add_values(result, row)
            return result
        if len(target) == len(grad):
            return [Tensor._reduce_broadcast(g, t) for g, t in zip(grad, target)]
        return Tensor._sum_values(grad)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._mul_values(self.data, other.data), self.requires_grad or other.requires_grad, (self, other))
        def backward():
            if self.requires_grad:
                grad = self._mul_values(out.grad, other.data)
                self._accumulate(grad)
            if other.requires_grad:
                grad = self._mul_values(out.grad, self.data)
                if other.ndim == 0 or other.shape != out.shape:
                    grad = self._reduce_broadcast(grad, other.data)
                other._accumulate(grad)
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        return Tensor(other) / self

    def __pow__(self, power):
        if isinstance(power, Tensor):
            raise NotImplementedError("tensor exponents are not supported")
        power = float(power)
        values = self.data
        def apply(x):
            return [apply(v) for v in x] if isinstance(x, list) else x ** power
        out = Tensor(apply(values), self.requires_grad, (self,))
        def backward():
            if self.requires_grad:
                def grad(g, x):
                    if isinstance(x, list):
                        return [grad(a, b) for a, b in zip(g, x)]
                    return g * power * (x ** (power - 1.0))
                self._accumulate(grad(out.grad, values))
        out._backward = backward
        return out

    def __matmul__(self, other):
        if self.ndim != 2 or other.ndim != 2 or self.shape[1] != other.shape[0]:
            raise ValueError("matmul requires compatible 2D tensors")
        a, b = self.data, other.data
        rows, inner, cols = len(a), len(b), len(b[0])
        result = [[sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)] for i in range(rows)]
        out = Tensor(result, self.requires_grad or other.requires_grad, (self, other))
        def backward():
            if self.requires_grad:
                self._accumulate([[sum(out.grad[i][j] * b[k][j] for j in range(cols)) for k in range(inner)] for i in range(rows)])
            if other.requires_grad:
                other._accumulate([[sum(a[i][k] * out.grad[i][j] for i in range(rows)) for j in range(cols)] for k in range(inner)])
        out._backward = backward
        return out

    def transpose(self):
        if self.ndim != 2:
            raise ValueError("transpose currently requires a 2D tensor")
        out = Tensor([list(row) for row in zip(*self.data)], self.requires_grad, (self,))
        def backward():
            if self.requires_grad:
                self._accumulate([list(row) for row in zip(*out.grad)])
        out._backward = backward
        return out

    def sum(self):
        out = Tensor(sum(self.flatten()), self.requires_grad, (self,))
        def backward():
            if self.requires_grad:
                self._accumulate(self._scale(self._ones_like(self.data), out.grad))
        out._backward = backward
        return out

    @staticmethod
    def _ones_like(x):
        return [Tensor._ones_like(v) for v in x] if isinstance(x, list) else 1.0

    def mean(self):
        return self.sum() * (1.0 / self.numel())

    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("backward requires requires_grad=True")
        if self.numel() != 1:
            raise RuntimeError("backward root must be scalar")
        self.grad = 1.0
        topology, seen = [], set()
        def visit(tensor):
            if id(tensor) in seen:
                return
            seen.add(id(tensor))
            for parent in tensor._parents:
                visit(parent)
            topology.append(tensor)
        visit(self)
        for tensor in reversed(topology):
            tensor._backward()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
