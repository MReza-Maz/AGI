import os
import tempfile
import unittest

from core.nn import Linear
from core.tensor import Tensor
from learning.loss import cross_entropy, mse
from learning.optimizer import Adam, GradientTools, SGD
from learning.trainer import Trainer


class LearningTests(unittest.TestCase):
    def test_sgd_changes_parameter(self):
        parameter = Tensor([[1.0]], requires_grad=True)
        parameter.grad = [[2.0]]
        SGD([parameter], learning_rate=0.1).step()
        self.assertAlmostEqual(parameter.data[0][0], 0.8)

    def test_adam_changes_parameter(self):
        parameter = Tensor([[1.0]], requires_grad=True)
        parameter.grad = [[2.0]]
        Adam([parameter], learning_rate=0.1).step()
        self.assertLess(parameter.data[0][0], 1.0)

    def test_gradient_clipping(self):
        parameter = Tensor([[3.0, 4.0]], requires_grad=True)
        parameter.grad = [[3.0, 4.0]]
        norm = GradientTools.clip_global_norm([parameter], 1.0)
        self.assertAlmostEqual(norm, 5.0)
        self.assertAlmostEqual(GradientTools.global_norm([parameter]), 1.0)

    def test_cross_entropy_gradient(self):
        logits = Tensor([[2.0, 0.0]], requires_grad=True)
        loss = cross_entropy(logits, [0])
        loss.backward()
        self.assertLess(logits.grad[0][0], 0.0)
        self.assertGreater(logits.grad[0][1], 0.0)

    def test_linear_regression_training(self):
        model = Linear(1, 1, seed=7)
        optimizer = Adam(model.parameters(), learning_rate=0.05)
        trainer = Trainer(model, optimizer, mse, clip_norm=10.0)
        dataset = [(Tensor([[x]]), Tensor([[2.0 * x + 1.0]])) for x in range(-3, 4)]
        initial = trainer.evaluate(dataset)
        history = trainer.fit(dataset, epochs=80)
        self.assertLess(history[-1]["loss"], initial)

    def test_checkpoint_round_trip(self):
        model = Linear(1, 1, seed=3)
        parameter = model.parameters()[0]
        original = parameter.data
        trainer = Trainer(model, Adam(model.parameters(), learning_rate=0.01), mse)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            trainer.save_checkpoint(path, {"name": "test"})
            parameter.data = [[999.0]]
            metadata = trainer.load_checkpoint(path)
        self.assertEqual(metadata["name"], "test")
        self.assertEqual(parameter.data, original)


if __name__ == "__main__":
    unittest.main()
