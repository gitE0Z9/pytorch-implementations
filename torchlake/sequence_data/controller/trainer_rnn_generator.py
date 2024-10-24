from typing import Iterable

from torchlake.common.controller.trainer import ClassificationTrainer

from ..models.base import RNNGenerator


class RNNGeneratorTrainer(ClassificationTrainer):
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
        output = model.forward(x, y, *args, **kwargs)

        if self.feature_last:
            output = output.permute(0, -1, *range(1, len(output.shape) - 1))

        return output
