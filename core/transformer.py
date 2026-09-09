from .nn import Linear,MLP,Module
from .attention import SelfAttention
from .tensor import Tensor


class TransformerBlock(Module):
    def __init__(self,size,seed=1):
        self.attention=SelfAttention(size,seed)
        self.feed_forward=MLP([size,size*2,size])
    def __call__(self,x):
        attended=self.attention(x)
        x=x+attended
        return x+self.feed_forward(x)


class Transformer(Module):
    def __init__(self,size,layers=2): self.blocks=[TransformerBlock(size,i+1) for i in range(layers)]
    def __call__(self,x):
        for block in self.blocks: x=block(x)
        return x
