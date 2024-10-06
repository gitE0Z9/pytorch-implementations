from typing import Any, Iterable

import torch
from torch import nn


class PredictFunctionMixin:
    def set_predict_strategy(
        self,
        is_x_collection: bool = False,
        nested_level: int = 0,
    ):
        if not is_x_collection:
            self._predict = self._predict_with_tensor
            return
        else:
            assert (
                nested_level > 0
            ), "if x is a collection, please tell us nested level of tensors"

            if nested_level == 1:
                self._predict = self._predict_with_tensors
                return

            if nested_level > 1:
                print(
                    "for deeply nested tensors, please send to device in collate_fn or earlier stage, we will pass x as it is"
                )
                self._predict = self._predict_do_nothing

    def build_predict_function_by_data_type(self, data: Iterable):
        x, _ = next(data)
        is_x_collection = isinstance(x, list | tuple | set)

        if not is_x_collection:
            self._predict = self._predict_with_tensor
            return
        else:
            is_x_tensor = isinstance(x[0], torch.Tensor)
            if is_x_tensor:
                self._predict = self._predict_with_tensors
            else:
                self._predict = self._predict_do_nothing

    def _predict_with_tensor(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor | Any:
        x, _ = row
        x: torch.Tensor = x.to(self.device)
        output = model(x, *args, **kwargs)
        # for output that has feature in last
        if self.feature_last:
            output = output.permute(0, -1, *range(1, len(output.shape) - 1))
        return output

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
        output = model(*x, *args, **kwargs)
        # for output that has feature in last
        if self.feature_last:
            for i in range(len(output)):
                output[i] = output[i].permute(0, -1, *range(1, len(output.shape) - 1))
        return output

    def _predict_do_nothing(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor | Any:
        x, _ = row
        return model(*x, *args, **kwargs)
