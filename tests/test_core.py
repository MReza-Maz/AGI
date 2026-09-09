import unittest
from core.tensor import Tensor
from core.nn import Linear, MLP
from learning.loss import mse


class CoreTests(unittest.TestCase):
    def test_matmul_shape(self):
        a=Tensor([[1,2],[3,4]])
        b=Tensor([[1,0,1],[0,1,1]])
        self.assertEqual((a@b).shape,(2,3))
    def test_autograd(self):
        x=Tensor([[2.0,3.0]],requires_grad=True)
        loss=(x*x).mean(); loss.backward()
        self.assertAlmostEqual(x.grad[0][0],2.0)
        self.assertAlmostEqual(x.grad[0][1],3.0)
    def test_mlp(self):
        y=MLP([2,4,2])(Tensor([[1,2]]))
        self.assertEqual(y.shape,(1,2))
    def test_mse(self):
        self.assertAlmostEqual(mse(Tensor([[1,2]]),Tensor([[1,4]])).data,2.0)


if __name__=="__main__": unittest.main()
