from typing import Iterable, Iterator, TypeVar

import torch
from torch import nn
from tqdm import tqdm

from torchlake.common.controller.evaluator import ClassificationEvaluator

T = TypeVar("T")


class PrototypicalEvaluator(ClassificationEvaluator):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ):
        q, s = row
        q, s = q.to(self.device), s.to(self.device)
        output = model(q, s)
        # for output that has feature in last
        if self.feature_last:
            output = output.permute(0, -1, *range(1, len(output.shape) - 1))
        return output

    def run(self, data: Iterator, model: nn.Module, metric: T | None = None) -> T:
        model.eval()
        with torch.no_grad():
            metric = metric or self._build_metric()
            for row in tqdm(data):
                output = self._predict(row, model)
                output = self._decode_output(output, row=row)
                self._update_metric(
                    metric,
                    torch.arange(self.label_size).expand_as(output),
                    output,
                )

        return metric
