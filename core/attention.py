"""Minimal self-attention implemented with the standard library."""
import math
from .tensor import Tensor
from .nn import Linear, Module


def softmax_rows(data):
    result=[]
    for row in data:
        m=max(row); ex=[math.exp(v-m) for v in row]; s=sum(ex)
        result.append([v/s for v in ex])
    return result


class SelfAttention(Module):
    def __init__(self, size, seed=1):
        self.q=Linear(size,size,seed)
        self.k=Linear(size,size,seed+1)
        self.v=Linear(size,size,seed+2)
    def __call__(self,x):
        q,k,v=self.q(x),self.k(x),self.v(x)
        scores=(q @ Tensor([[z for z in row] for row in zip(*k.data)])) * (1.0/math.sqrt(q.shape[-1]))
        weights=Tensor(softmax_rows(scores.data))
        return weights @ v
