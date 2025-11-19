from typing import Iterable, TypeVar

from torch import nn

from torchlake.common.controller.evaluator import ClassificationEvaluator

T = TypeVar("T")


class EpisodeEvaluator(ClassificationEvaluator):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ):
        query_set, support_set, _ = row
        query_set, support_set = query_set.to(self.device), support_set.to(self.device)

        output = model(query_set, support_set)
        return output
