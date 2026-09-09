import json
import os
import tempfile
import unittest

from cognition.tokenizer import ByteTokenizer
from learning.loss import cross_entropy
from learning.optimizer import Adam
from learning.sequence import SequenceDataset
from learning.trainer import Trainer
from cognition.language_model import CausalLanguageModel
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


if __name__ == "__main__":
    unittest.main()
