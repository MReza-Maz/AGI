"""Training orchestration for the dependency-free AGI research stack."""
import json
import os
from learning.optimizer import GradientTools


class Trainer:
    """Deterministic training loop with evaluation, clipping and checkpointing."""
    def __init__(self, model, optimizer, loss_fn, clip_norm=None, evaluator=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_norm = clip_norm
        self.evaluator = evaluator
        self.history = []
        self.step_count = 0

    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        gradient_norm = GradientTools.clip_global_norm(
            self.optimizer.parameters, self.clip_norm
        ) if self.clip_norm is not None else GradientTools.global_norm(self.optimizer.parameters)
        self.optimizer.step()
        self.step_count += 1
        result = {
            "step": self.step_count,
            "loss": float(loss.data),
            "gradient_norm": float(gradient_norm),
        }
        self.history.append(result)
        return result

    def fit(self, dataset, epochs=1, eval_dataset=None):
        records = []
        items = list(dataset)
        for epoch in range(1, int(epochs) + 1):
            total = 0.0
            for inputs, targets in items:
                total += self.train_step(inputs, targets)["loss"]
            record = {"epoch": epoch, "loss": total / max(1, len(items))}
            if eval_dataset is not None:
                record["eval_loss"] = self.evaluate(eval_dataset)
            records.append(record)
        return records

    def train(self, dataset, epochs=1):
        """Backward-compatible alias returning epoch records."""
        return self.fit(dataset, epochs=epochs)

    def evaluate(self, dataset):
        items = list(dataset)
        if not items:
            return 0.0
        total = 0.0
        for inputs, targets in items:
            total += float(self.loss_fn(self.model(inputs), targets).data)
        return total / len(items)

    def save_checkpoint(self, path, metadata=None):
        payload = {
            "metadata": dict(metadata or {}),
            "step_count": self.step_count,
            "parameters": [p.data for p in self.optimizer.parameters],
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(temporary, path)

    def load_checkpoint(self, path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        values = payload.get("parameters", [])
        parameters = self.optimizer.parameters
        if len(values) != len(parameters):
            raise ValueError("checkpoint parameter count does not match model")
        for parameter, value in zip(parameters, values):
            if parameter.shape != self._shape(value):
                raise ValueError("checkpoint parameter shape does not match model")
            parameter.data = value
        self.step_count = int(payload.get("step_count", 0))
        return payload.get("metadata", {})

    @staticmethod
    def _shape(value):
        shape = []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else []
        return tuple(shape)
