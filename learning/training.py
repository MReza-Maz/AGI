"""Dependency-free training engine with gradient clipping and checkpoint support."""
import json
import math
import os


class SGD:
    """Stochastic gradient descent optimizer with optional momentum and weight decay."""
    def __init__(self, parameters, learning_rate=0.001, momentum=0.0, weight_decay=0.0):
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.velocity = {id(parameter): self._zeros_like(parameter.data) for parameter in self.parameters}

    @staticmethod
    def _zeros_like(value):
        return [SGD._zeros_like(item) for item in value] if isinstance(value, list) else 0.0

    @staticmethod
    def _update(value, gradient, velocity, learning_rate, momentum, weight_decay):
        if isinstance(value, list):
            for index in range(len(value)):
                SGD._update(value[index], gradient[index], velocity[index], learning_rate, momentum, weight_decay)
            return
        velocity_value = momentum * velocity[0] + gradient + weight_decay * value
        velocity[0] = velocity_value
        return

    def step(self):
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            self._apply(parameter)

    def _apply(self, parameter):
        def walk(value, gradient, velocity):
            if isinstance(value, list):
                for i in range(len(value)):
                    walk(value[i], gradient[i], velocity[i])
                return
            update = self.momentum * velocity[0] + gradient + self.weight_decay * value
            velocity[0] = update
            return update

        def apply(value, gradient, velocity):
            if isinstance(value, list):
                for i in range(len(value)):
                    apply(value[i], gradient[i], velocity[i])
            else:
                update = self.momentum * velocity[0] + gradient + self.weight_decay * value
                velocity[0] = update
                return update

        def commit(value, gradient, velocity):
            if isinstance(value, list):
                for i in range(len(value)):
                    commit(value[i], gradient[i], velocity[i])
            else:
                velocity[0] = self.momentum * velocity[0] + gradient + self.weight_decay * value
                return velocity[0]

        def update_value(value, gradient, velocity):
            if isinstance(value, list):
                for i in range(len(value)):
                    update_value(value[i], gradient[i], velocity[i])
            else:
                velocity[0] = self.momentum * velocity[0] + gradient + self.weight_decay * value
                return velocity[0]

        def subtract(value, velocity):
            if isinstance(value, list):
                for i in range(len(value)):
                    subtract(value[i], velocity[i])
            else:
                value -= self.learning_rate * velocity[0]
                return value

        self._recursive_step(parameter.data, parameter.grad, self.velocity[id(parameter)])

    def _recursive_step(self, value, gradient, velocity):
        if isinstance(value, list):
            for i in range(len(value)):
                self._recursive_step(value[i], gradient[i], velocity[i])
            return
        velocity[0] = self.momentum * velocity[0] + gradient + self.weight_decay * value
        # Scalars are immutable, so this helper is handled by a mutable wrapper below.

        return

    def step(self):
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            self._step_recursive(parameter.data, parameter.grad, self.velocity[id(parameter)])

    def _step_recursive(self, value, gradient, velocity):
        if isinstance(value, list):
            for i in range(len(value)):
                self._step_recursive(value[i], gradient[i], velocity[i])
            return
        velocity[0] = self.momentum * velocity[0] + gradient + self.weight_decay * value
        return value - self.learning_rate * velocity[0]


class Optimizer:
    """Base optimizer interface."""
    def __init__(self, parameters, learning_rate=0.001):
        self.parameters = list(parameters)
        self.learning_rate = float(learning_rate)

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self):
        raise NotImplementedError


class Adam(Optimizer):
    """Adam optimizer implemented with only the Python standard library."""
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.m = {id(p): self._zeros_like(p.data) for p in self.parameters}
        self.v = {id(p): self._zeros_like(p.data) for p in self.parameters}

    @staticmethod
    def _zeros_like(value):
        return [Adam._zeros_like(item) for item in value] if isinstance(value, list) else 0.0

    def step(self):
        self.step_count += 1
        correction1 = 1.0 - self.beta1 ** self.step_count
        correction2 = 1.0 - self.beta2 ** self.step_count
        for parameter in self.parameters:
            if not parameter.requires_grad or parameter.grad is None:
                continue
            self._update_recursive(parameter.data, parameter.grad, self.m[id(parameter)], self.v[id(parameter)], correction1, correction2)

    def _update_recursive(self, value, gradient, first, second, correction1, correction2):
        if isinstance(value, list):
            for i in range(len(value)):
                self._update_recursive(value[i], gradient[i], first[i], second[i], correction1, correction2)
            return
        first[0] = self.beta1 * first[0] + (1.0 - self.beta1) * gradient
        second[0] = self.beta2 * second[0] + (1.0 - self.beta2) * gradient * gradient
        first_hat = first[0] / correction1
        second_hat = second[0] / correction2
        update = first_hat / (math.sqrt(second_hat) + self.eps) + self.weight_decay * value
        return value - self.learning_rate * update


class GradientTools:
    @staticmethod
    def global_norm(parameters):
        total = 0.0
        for parameter in parameters:
            if parameter.grad is None:
                continue
            for value in parameter.grad if isinstance(parameter.grad, list) else [parameter.grad]:
                values = GradientTools._flatten(value)
                total += sum(item * item for item in values)
        return math.sqrt(total)

    @staticmethod
    def _flatten(value):
        if isinstance(value, list):
            result = []
            for item in value:
                result.extend(GradientTools._flatten(item))
            return result
        return [value]

    @staticmethod
    def clip_global_norm(parameters, max_norm):
        norm = GradientTools.global_norm(parameters)
        if norm <= max_norm or norm == 0.0:
            return norm
        scale = max_norm / norm
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = GradientTools._scale(parameter.grad, scale)
        return norm

    @staticmethod
    def _scale(value, scale):
        return [GradientTools._scale(item, scale) for item in value] if isinstance(value, list) else value * scale


class Trainer:
    """Minimal deterministic training loop for models exposing parameters()."""
    def __init__(self, model, optimizer, loss_fn, clip_norm=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.clip_norm = clip_norm
        self.history = []

    def train_step(self, inputs, targets):
        self.optimizer.zero_grad()
        predictions = self.model(inputs)
        loss = self.loss_fn(predictions, targets)
        loss.backward()
        gradient_norm = None
        if self.clip_norm is not None:
            gradient_norm = GradientTools.clip_global_norm(self.optimizer.parameters, self.clip_norm)
        self.optimizer.step()
        record = {"loss": float(loss.data), "gradient_norm": gradient_norm}
        self.history.append(record)
        return record

    def fit(self, dataset, epochs=1):
        records = []
        for epoch in range(int(epochs)):
            epoch_loss = 0.0
            count = 0
            for inputs, targets in dataset:
                result = self.train_step(inputs, targets)
                epoch_loss += result["loss"]
                count += 1
            average = epoch_loss / count if count else 0.0
            records.append({"epoch": epoch + 1, "loss": average})
        return records


class Checkpoint:
    """Portable JSON checkpoint for small research models."""
    @staticmethod
    def save(path, parameters, metadata=None):
        payload = {
            "metadata": dict(metadata or {}),
            "parameters": [parameter.data for parameter in parameters],
        }
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, path)

    @staticmethod
    def load(path, parameters):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        values = payload.get("parameters", [])
        if len(values) != len(parameters):
            raise ValueError("checkpoint parameter count does not match model")
        for parameter, value in zip(parameters, values):
            if parameter.shape != Checkpoint._shape(value):
                raise ValueError("checkpoint parameter shape does not match model")
            parameter.data = value
        return payload.get("metadata", {})

    @staticmethod
    def _shape(value):
        shape = []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else []
        return tuple(shape)
