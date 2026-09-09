"""Dependency-free optimization algorithms."""
import math


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


class SGD(Optimizer):
    """Stochastic gradient descent with momentum and weight decay."""
    def __init__(self, parameters, learning_rate=0.001, momentum=0.0, weight_decay=0.0):
        super().__init__(parameters, learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.velocity = {id(p): self._zeros_like(p.data) for p in self.parameters}

    @staticmethod
    def _zeros_like(value):
        # Keep scalar optimizer state mutable so nested parameter trees work.
        return [SGD._zeros_like(x) for x in value] if isinstance(value, list) else [0.0]

    def step(self):
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            velocity = self.velocity[id(parameter)]
            parameter.data = self._update(parameter.data, parameter.grad, velocity)

    def _update(self, value, gradient, velocity):
        if isinstance(value, list):
            return [self._update(v, g, m) for v, g, m in zip(value, gradient, velocity)]
        velocity[0] = self.momentum * velocity[0] + gradient + self.weight_decay * value
        return value - self.learning_rate * velocity[0]


class Adam(Optimizer):
    """Adam optimizer with bias correction and optional L2 weight decay."""
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        super().__init__(parameters, learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.weight_decay = float(weight_decay)
        self.step_count = 0
        self.first_moment = {id(p): self._zeros_like(p.data) for p in self.parameters}
        self.second_moment = {id(p): self._zeros_like(p.data) for p in self.parameters}

    @staticmethod
    def _zeros_like(value):
        # Keep scalar optimizer state mutable so nested parameter trees work.
        return [Adam._zeros_like(x) for x in value] if isinstance(value, list) else [0.0]

    def step(self):
        self.step_count += 1
        c1 = 1.0 - self.beta1 ** self.step_count
        c2 = 1.0 - self.beta2 ** self.step_count
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            p_id = id(parameter)
            parameter.data = self._update(
                parameter.data, parameter.grad,
                self.first_moment[p_id], self.second_moment[p_id], c1, c2
            )

    def _update(self, value, gradient, first, second, correction1, correction2):
        if isinstance(value, list):
            return [
                self._update(v, g, m, n, correction1, correction2)
                for v, g, m, n in zip(value, gradient, first, second)
            ]
        first[0] = self.beta1 * first[0] + (1.0 - self.beta1) * gradient
        second[0] = self.beta2 * second[0] + (1.0 - self.beta2) * gradient * gradient
        m_hat = first[0] / correction1
        v_hat = second[0] / correction2
        update = m_hat / (math.sqrt(v_hat) + self.eps) + self.weight_decay * value
        return value - self.learning_rate * update


class GradientTools:
    """Gradient diagnostics and global-norm clipping."""
    @staticmethod
    def _flatten(value):
        if isinstance(value, list):
            result = []
            for item in value:
                result.extend(GradientTools._flatten(item))
            return result
        return [value]

    @classmethod
    def global_norm(cls, parameters):
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += sum(x * x for x in cls._flatten(parameter.grad))
        return math.sqrt(total)

    @classmethod
    def clip_global_norm(cls, parameters, max_norm):
        norm = cls.global_norm(parameters)
        if norm <= max_norm or norm == 0.0:
            return norm
        scale = float(max_norm) / norm
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = cls._scale(parameter.grad, scale)
        return norm

    @staticmethod
    def _scale(value, scale):
        return [GradientTools._scale(x, scale) for x in value] if isinstance(value, list) else value * scale
