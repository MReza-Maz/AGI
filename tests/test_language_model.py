import unittest

from cognition.language_model import CausalLanguageModel
from core.tensor import Tensor
from learning.sequence import SequenceDataset


class LanguageModelTests(unittest.TestCase):
    def test_sequence_dataset_shift(self):
        dataset = SequenceDataset([1, 2, 3, 4, 5], sequence_length=3)
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], ([1, 2, 3], [2, 3, 4]))
        self.assertEqual(dataset[-1], ([2, 3, 4], [3, 4, 5]))

    def test_language_model_logits_shape(self):
        model = CausalLanguageModel(12, model_size=8, layers=1, heads=2, seed=9, max_sequence_length=16)
        logits = model([1, 2, 3, 4])
        self.assertEqual(logits.shape, (4, 12))

    def test_language_model_backward(self):
        model = CausalLanguageModel(10, model_size=8, layers=1, heads=2, seed=4, max_sequence_length=8)
        logits = model([1, 2, 3])
        loss = logits.mean()
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_generation_length(self):
        model = CausalLanguageModel(10, model_size=8, layers=1, heads=2, seed=4, max_sequence_length=8)
        generated = model.generate([1, 2], max_new_tokens=3)
        self.assertEqual(len(generated), 5)
        self.assertTrue(all(0 <= token_id < 10 for token_id in generated))


if __name__ == "__main__":
    unittest.main()
