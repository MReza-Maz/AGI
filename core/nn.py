"""Dependency-free neural network modules."""
import math
from .tensor import Tensor


class Module:
    def parameters(self):
        result=[]
        for value in self.__dict__.values():
            if isinstance(value, Tensor) and value.requires_grad: result.append(value)
            elif isinstance(value, Module): result.extend(value.parameters())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, Module): result.extend(item.parameters())
        return result


class Linear(Module):
    def __init__(self, in_features, out_features, seed=1):
        self.weight=Tensor.random((in_features,out_features),scale=math.sqrt(2.0/in_features),seed=seed,requires_grad=True)
        self.bias=Tensor.zeros((1,out_features),requires_grad=True)
    def __call__(self,x): return (x @ self.weight) + self.bias


class ReLU(Module):
    def __call__(self,x):
        data=x.data
        def relu(v): return [relu(z) for z in v] if isinstance(v,list) else max(0.0,v)
        out=Tensor(relu(data),x.requires_grad,(x,))
        def backward():
            if x.requires_grad:
                def grad(g,v):
                    if isinstance(v,list): return [grad(a,b) for a,b in zip(g,v)]
                    return g if v>0 else 0.0
                x._accumulate(grad(out.grad,data))
        out._backward=backward
        return out


class MLP(Module):
    def __init__(self, sizes, seed=1):
        self.layers=[]
        for i in range(len(sizes)-1):
            self.layers.append(Linear(sizes[i],sizes[i+1],seed+i))
            if i < len(sizes)-2: self.layers.append(ReLU())
    def __call__(self,x):
        for layer in self.layers: x=layer(x)
        return x
