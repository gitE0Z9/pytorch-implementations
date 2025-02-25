import math
from typing import Iterable, Literal

import torch
from torch import nn
from torchlake.common.controller.trainer import ClassificationTrainer

from ..models.base import RNNGenerator


class RNNGeneratorTrainer(ClassificationTrainer):
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

    def _predict(
        self,
        row: tuple[Iterable],
        model: RNNGenerator,
        *args,
        **kwargs,
    ):
        x, y = row
        x = x.to(self.device)
        y = y.to(self.device)

        if "teacher_forcing_ratio" in kwargs:
            teacher_forcing_ratio = kwargs.pop("teacher_forcing_ratio")
        else:
            teacher_forcing_ratio = self.teacher_forcing_raio
        output = model.forward(
            x,
            y,
            teacher_forcing_ratio=teacher_forcing_ratio,
            *args,
            **kwargs,
        )

        if self.feature_last:
            output = output.permute(0, -1, *range(1, len(output.shape) - 1))

        return output

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        # early stopping since pad are ignored when training
        return criterion(y_hat, y.long()[:, : y_hat.size(2)])
