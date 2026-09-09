"""Small nested-list tensor with reverse-mode automatic differentiation.

The implementation is intentionally dependency-free and designed for research,
not for replacing mature numerical frameworks.
"""
import math
import random
from typing import Callable, List, Sequence


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
        out = []
        while isinstance(x, list):
            out.append(len(x))
            x = x[0] if x else []
        return tuple(out)

    @property
    def ndim(self):
        return len(self.shape)

    def clone(self):
        return Tensor(self.data, self.requires_grad)

    def flatten(self):
        def walk(x):
            if isinstance(x, list):
                r = []
                for v in x: r.extend(walk(v))
                return r
            return [x]
        return walk(self.data)

    def numel(self):
        return len(self.flatten())

    @classmethod
    def zeros(cls, shape, requires_grad=False):
        def build(s): return [build(s[1:]) for _ in range(s[0])] if s else 0.0
        return cls(build(tuple(shape)), requires_grad)

    @classmethod
    def random(cls, shape, scale=0.02, seed=None, requires_grad=False):
        rng = random.Random(seed) if seed is not None else random
        def build(s):
            return [build(s[1:]) for _ in range(s[0])] if s else rng.uniform(-scale, scale)
        return cls(build(tuple(shape)), requires_grad)

    def _accumulate(self, value):
        def add(a, b):
            if isinstance(a, list): return [add(x, y) for x, y in zip(a, b)]
            return a + b
        self.grad = add(self.grad, value)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        def add(a,b):
            if isinstance(a,list): return [add(x,y) for x,y in zip(a,b)]
            return a+b
        out = Tensor(add(self.data, other.data), self.requires_grad or other.requires_grad, (self,other))
        def backward():
            if self.requires_grad: self._accumulate(out.grad)
            if other.requires_grad: other._accumulate(out.grad)
        out._backward = backward
        return out

    __radd__ = __add__

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        def mul(a,b):
            if isinstance(a,list): return [mul(x,y) for x,y in zip(a,b)]
            return a*b
        out = Tensor(mul(self.data, other.data), self.requires_grad or other.requires_grad, (self,other))
        def backward():
            def scale(g,x):
                if isinstance(g,list): return [scale(gg,xx) for gg,xx in zip(g,x)]
                return g*x
            if self.requires_grad: self._accumulate(scale(out.grad, other.data))
            if other.requires_grad: other._accumulate(scale(out.grad, self.data))
        out._backward = backward
        return out

    __rmul__ = __mul__

    def __matmul__(self, other):
        if self.ndim != 2 or other.ndim != 2 or self.shape[1] != other.shape[0]:
            raise ValueError("matmul requires compatible 2D tensors")
        a,b=self.data,other.data
        result=[[sum(a[i][k]*b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]
        out=Tensor(result,self.requires_grad or other.requires_grad,(self,other))
        def backward():
            if self.requires_grad:
                g=[[sum(out.grad[i][j]*b[k][j] for j in range(len(b[0]))) for k in range(len(b))] for i in range(len(a))]
                self._accumulate(g)
            if other.requires_grad:
                g=[[sum(a[i][k]*out.grad[i][j] for i in range(len(a))) for j in range(len(b[0]))] for k in range(len(b))]
                other._accumulate(g)
        out._backward=backward
        return out

    def sum(self):
        value=sum(self.flatten())
        out=Tensor(value,self.requires_grad,(self,))
        def backward():
            if self.requires_grad:
                def ones(x): return [ones(v) for v in x] if isinstance(x,list) else 1.0
                self._accumulate(ones(self.data))
        out._backward=backward
        return out

    def mean(self):
        return self.sum() * (1.0 / self.numel())

    def backward(self):
        if not self.requires_grad: raise RuntimeError("backward requires requires_grad=True")
        if self.numel() != 1: raise RuntimeError("backward root must be scalar")
        self.grad = 1.0
        topo=[]; seen=set()
        def visit(t):
            if id(t) in seen: return
            seen.add(id(t))
            for p in t._parents: visit(p)
            topo.append(t)
        visit(self)
        for t in reversed(topo): t._backward()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
