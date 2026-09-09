import unittest
from cognition.tokenizer import BasicTokenizer, ByteTokenizer
from cognition.embeddings import Embedding
from cognition.world_model import WorldState
from cognition.goals import Goal, GoalManager
from core.tensor import Tensor
from core.transformer import Transformer, LayerNorm


class CognitionTests(unittest.TestCase):
    def test_tokenizer_round_trip(self):
        tokenizer = BasicTokenizer()
        tokenizer.fit(["hello world", "hello AGI"])
        ids = tokenizer.encode("hello world")
        self.assertEqual(tokenizer.decode(ids), "hello world")

    def test_byte_tokenizer_multilingual_round_trip(self):
        tokenizer = ByteTokenizer()
        text = "سلام AGI 🌍"
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(tokenizer.decode(ids), "<BOS>" + text + "<EOS>")
        self.assertEqual(len(tokenizer), 260)

    def test_byte_tokenizer_is_deterministic(self):
        tokenizer = ByteTokenizer()
        self.assertEqual(tokenizer.encode("سلام"), tokenizer.encode("سلام"))
        self.assertEqual(tokenizer.decode(tokenizer.encode("hello")), "hello")

    def test_embedding_shape(self):
        embedding = Embedding(10, 8, seed=7)
        self.assertEqual(embedding([1, 3, 5]).shape, (3, 8))
        self.assertEqual(len(embedding.parameters()), 1)

    def test_world_state_copy_and_effect(self):
        state = WorldState({"power": "off"})
        next_state = state.apply({"power": "on"})
        self.assertEqual(state.get("power"), "off")
        self.assertEqual(next_state.get("power"), "on")

    def test_goal_manager(self):
        state = WorldState({"ready": False})
        manager = GoalManager()
        manager.add(Goal("be ready", priority=2, target={"ready": True}))
        self.assertIsNotNone(manager.next_goal())
        manager.refresh(WorldState({"ready": True}))
        self.assertEqual(manager.next_goal(), None)

    def test_transformer_shape(self):
        model = Transformer(8, layers=2)
        output = model(Tensor([[0.1] * 8, [0.2] * 8]))
        self.assertEqual(output.shape, (2, 8))

    def test_layer_norm(self):
        output = LayerNorm(4)(Tensor([[1.0, 2.0, 3.0, 4.0]]))
        self.assertEqual(output.shape, (1, 4))
        self.assertAlmostEqual(sum(output.data[0]) / 4.0, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
