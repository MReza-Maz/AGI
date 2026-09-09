"""High-level cognitive workspace."""
from .nn import MLP
from .tensor import Tensor
from .attention import SelfAttention


class Brain:
    def __init__(self, input_size=16, hidden_size=32, output_size=16):
        self.encoder=MLP([input_size,hidden_size,output_size])
        self.attention=SelfAttention(output_size)

    def think(self, vector):
        x=vector if isinstance(vector,Tensor) else Tensor(vector)
        if x.ndim==1: x=Tensor([x.data])
        encoded=self.encoder(x)
        return self.attention(encoded)
