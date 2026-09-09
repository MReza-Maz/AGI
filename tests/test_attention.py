import unittest

from core.attention import MultiHeadSelfAttention
from core.tensor import Tensor
from core.transformer import Transformer


class AttentionTests(unittest.TestCase):
    def test_multi_head_shape(self):
        attention = MultiHeadSelfAttention(8, heads=4, causal=True, seed=11)
        output = attention(Tensor([[0.1] * 8, [0.2] * 8, [0.3] * 8]))
        self.assertEqual(output.shape, (3, 8))

    def test_causal_attention_has_no_future_dependency(self):
        attention = MultiHeadSelfAttention(4, heads=2, causal=True, seed=5)
        first = Tensor([[0.1] * 4, [0.2] * 4])
        changed_future = Tensor([[0.1] * 4, [100.0] * 4])
        a = attention(first).data[0]
        b = attention(changed_future).data[0]
        for left, right in zip(a, b):
            self.assertAlmostEqual(left, right, places=6)

    def test_transformer_uses_multiple_heads(self):
        model = Transformer(8, layers=2, heads=4, causal=True)
        output = model(Tensor([[0.1] * 8, [0.2] * 8]))
        self.assertEqual(output.shape, (2, 8))


if __name__ == "__main__":
    unittest.main()
