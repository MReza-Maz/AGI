from core.tensor import Tensor


def mse(prediction,target):
    target=target if isinstance(target,Tensor) else Tensor(target)
    diff=prediction+target*(-1.0)
    return (diff*diff).mean()
