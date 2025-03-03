import math
from typing import Literal


class CurriculumMixin:
    """intend to support trainer, recorder is required"""

    def set_curriculum_strategy(
        self,
        strategy: Literal[
            "linear",
            "exponential",
            "inverse_sigmoid",
        ] = "inverse_sigmoid",
        k: int = 10,
        c: int = 1,
        epsilon: int = 0.1,
    ):
        self.k = k
        self.c = c
        self.epsilon = epsilon
        self.curriculum_strategy = strategy

    @property
    def teacher_forcing_raio(self) -> float:
        strategy_mapping = {
            "linear": self.inverse_sigmoid_strategy,
            "exponential": self.inverse_sigmoid_strategy,
            "inverse_sigmoid": self.inverse_sigmoid_strategy,
        }

        curriculum_strategy = getattr(self, "curriculum_strategy", None)
        if curriculum_strategy:
            return strategy_mapping[self.curriculum_strategy]()
        else:
            return 0.5

    def linear_strategy(self) -> float:
        return max(self.epsilon, self.k - self.c * self.recorder.current_epoch)

    def exponential_strategy(self) -> float:
        assert self.k < 1, "exponential strategy does not support k >= 1"

        return self.k**self.recorder.current_epoch

    def inverse_sigmoid_strategy(self) -> float:
        assert self.k >= 1, "inverse sigmoid strategy does not support k < 1"

        return self.k / (self.k + math.exp(self.recorder.current_epoch / self.k))
