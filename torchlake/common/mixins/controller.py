from typing import Any, Iterable

import torch
from torch import nn


class PredictFunctionMixin:
    def build_predict_function_by_data_type(self, data: Iterable):
        x, _ = next(data)
        if isinstance(x, list):
            self._predict = self._predict_with_tensors
        else:
            self._predict = self._predict_with_tensor

    def _predict_with_tensor(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor | Any:
        x, _ = row
        x: torch.Tensor = x.to(self.device)
        return model(x, *args, **kwargs)

    def _predict_with_tensors(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor | Any:
        x, _ = row
        for i in range(len(x)):
            x[i] = x[i].to(self.device)

        return model(*x, *args, **kwargs)
