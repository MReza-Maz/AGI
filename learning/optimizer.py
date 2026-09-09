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

    @staticmethod
    def _copy_state(value):
        if isinstance(value, list):
            return [Optimizer._copy_state(item) for item in value]
        return float(value)

    @staticmethod
    def _shape(value):
        shape = []
        while isinstance(value, list):
            shape.append(len(value))
            value = value[0] if value else []
        return tuple(shape)

    @staticmethod
    def _state_matches_parameter(state, parameter_data):
        """Validate optimizer state against parameter data, including scalar wrappers."""
        if isinstance(parameter_data, list):
            if not isinstance(state, list) or len(state) != len(parameter_data):
                return False
            return all(
                Optimizer._state_matches_parameter(state_item, parameter_item)
                for state_item, parameter_item in zip(state, parameter_data)
            )
        return (
            isinstance(state, list)
            and len(state) == 1
            and not isinstance(state[0], list)
        )


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

    def state_dict(self):
        return {
            "type": "sgd",
            "learning_rate": self.learning_rate,
            "momentum": self.momentum,
            "weight_decay": self.weight_decay,
            "velocity": [self._copy_state(self.velocity[id(p)]) for p in self.parameters],
        }

    def load_state_dict(self, state):
        if state.get("type") != "sgd":
            raise ValueError("optimizer type does not match checkpoint")
        values = state.get("velocity", [])
        if len(values) != len(self.parameters):
            raise ValueError("optimizer parameter count does not match checkpoint")
        for parameter, value in zip(self.parameters, values):
            if not self._state_matches_parameter(value, parameter.data):
                raise ValueError("optimizer state shape does not match model")
            self.velocity[id(parameter)] = value
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.momentum = float(state.get("momentum", self.momentum))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))


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

    def state_dict(self):
        return {
            "type": "adam",
            "learning_rate": self.learning_rate,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "step_count": self.step_count,
            "first_moment": [self._copy_state(self.first_moment[id(p)]) for p in self.parameters],
            "second_moment": [self._copy_state(self.second_moment[id(p)]) for p in self.parameters],
        }

    def load_state_dict(self, state):
        if state.get("type") != "adam":
            raise ValueError("optimizer type does not match checkpoint")
        first = state.get("first_moment", [])
        second = state.get("second_moment", [])
        if len(first) != len(self.parameters) or len(second) != len(self.parameters):
            raise ValueError("optimizer parameter count does not match checkpoint")
        for parameter, first_value, second_value in zip(self.parameters, first, second):
            if not self._state_matches_parameter(first_value, parameter.data):
                raise ValueError("optimizer state shape does not match model")
            if not self._state_matches_parameter(second_value, parameter.data):
                raise ValueError("optimizer state shape does not match model")
            self.first_moment[id(parameter)] = first_value
            self.second_moment[id(parameter)] = second_value
        self.step_count = int(state.get("step_count", 0))
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.beta1 = float(state.get("beta1", self.beta1))
        self.beta2 = float(state.get("beta2", self.beta2))
        self.eps = float(state.get("eps", self.eps))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))


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
