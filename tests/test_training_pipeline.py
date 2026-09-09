import os
import tempfile
import unittest

from cognition.inference import InferenceEngine
from cognition.tokenizer import ByteTokenizer
from cognition.language_model import CausalLanguageModel
from learning.loss import cross_entropy
from learning.optimizer import Adam
from learning.sequence import SequenceDataset
from learning.trainer import Trainer
from generate import sample_token


class TrainingPipelineTests(unittest.TestCase):
    def test_tokenizer_serialization_round_trip(self):
        tokenizer = ByteTokenizer()
        payload = tokenizer.to_dict()
        restored = ByteTokenizer.from_dict(payload)
        text = "سلام AGI 🚀"
        self.assertEqual(restored.decode(restored.encode(text)), text)
        self.assertEqual(len(restored), len(tokenizer))

    def test_checkpoint_restores_adam_state(self):
        model = CausalLanguageModel(260, model_size=8, layers=1, heads=2, seed=7, max_sequence_length=8)
        optimizer = Adam(model.parameters(), learning_rate=0.001)
        trainer = Trainer(model, optimizer, cross_entropy)
        dataset = SequenceDataset(list(range(12)), sequence_length=4)
        trainer.train_step(*dataset[0])
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "checkpoint.json")
            trainer.save_checkpoint(path, metadata={"completed_epochs": 1})
            restored_model = CausalLanguageModel(260, model_size=8, layers=1, heads=2, seed=7, max_sequence_length=8)
            restored_optimizer = Adam(restored_model.parameters(), learning_rate=0.001)
            restored_trainer = Trainer(restored_model, restored_optimizer, cross_entropy)
            metadata = restored_trainer.load_checkpoint(path)
            self.assertEqual(metadata["completed_epochs"], 1)
            self.assertEqual(restored_optimizer.step_count, optimizer.step_count)

    def test_sampling_top_k(self):
        self.assertEqual(sample_token([0.0, 10.0, 1.0], temperature=1.0, top_k=1, seed=1), 1)

    def test_sampling_top_p(self):
        self.assertEqual(
            InferenceEngine.sample_token([10.0, 9.0, 0.0], top_p=0.5, rng=_FixedRandom()),
            0,
        )

    def test_sampling_is_deterministic_with_seed(self):
        logits = [0.2, 1.0, 0.5, 2.0]
        first = InferenceEngine.sample_token(logits, temperature=0.9, top_k=3, top_p=0.9, rng=_seeded(11))
        second = InferenceEngine.sample_token(logits, temperature=0.9, top_k=3, top_p=0.9, rng=_seeded(11))
        self.assertEqual(first, second)

    def test_inference_engine_checkpoint_load(self):
        model = CausalLanguageModel(260, model_size=8, layers=1, heads=2, seed=7, max_sequence_length=8)
        trainer = Trainer(model, Adam(model.parameters(), learning_rate=0.001), cross_entropy)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = os.path.join(directory, "model.json")
            tokenizer_path = os.path.join(directory, "tokenizer.json")
            trainer.save_checkpoint(checkpoint, metadata={
                "model_size": 8, "layers": 1, "heads": 2,
                "sequence_length": 8, "vocabulary_size": 260, "seed": 7,
            })
            tokenizer = ByteTokenizer()
            with open(tokenizer_path, "w", encoding="utf-8") as handle:
                import json
                json.dump(tokenizer.to_dict(), handle)
            engine = InferenceEngine.from_files(checkpoint, tokenizer_path)
            self.assertEqual(engine.sequence_length, 8)
            self.assertEqual(engine.tokenizer.encode("AGI"), [65 + 4, 71 + 4, 73 + 4])

    def test_eos_stops_generation(self):
        model = _StubModel([1, 2])
        engine = InferenceEngine(model, ByteTokenizer(), sequence_length=4)
        result = engine.generate([10], max_new_tokens=5, eos_token_id=1)
        self.assertEqual(result, [10, 1])

    def test_beam_search_returns_best_sequence(self):
        model = _BeamModel()
        engine = InferenceEngine(model, ByteTokenizer(), sequence_length=4)
        result = engine.beam_search([10], max_new_tokens=2, beam_width=2)
        self.assertEqual(result, [10, 1, 3])


class _FixedRandom:
    def random(self):
        return 0.0


def _seeded(seed):
    import random
    return random.Random(seed)


class _StubModel:
    def __init__(self, sequences):
        self.sequences = list(sequences)
        self.calls = 0

    def next_token_logits(self, _context):
        token = self.sequences[min(self.calls, len(self.sequences) - 1)]
        self.calls += 1
        logits = [0.0] * 260
        logits[token] = 10.0
        return logits


class _BeamModel:
    def next_token_logits(self, context):
        logits = [0.0] * 260
        if context[-1] == 10:
            logits[1] = 2.0
            logits[2] = 1.0
        elif context[-1] == 1:
            logits[3] = 5.0
        else:
            logits[3] = 1.0
        return logits


if __name__ == "__main__":
    unittest.main()
