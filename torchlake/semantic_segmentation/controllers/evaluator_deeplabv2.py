from typing import Iterable

import torch
import torch.nn.functional as F
from torch import nn
from torchlake.common.controller.evaluator import ClassificationEvaluator


class DeepLabV2Evaluator(ClassificationEvaluator):
    def set_multiscales(self, multi_scales: list[int] = [1, 0.75, 0.5]):
        self.multi_scales = multi_scales

    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ):
        x, _ = row
        x = x.to(self.device)

        output = []

        for scale in self.multi_scales:
            yhat = model(
                F.interpolate(
                    x,
                    size=int(x.size(3) * scale) + 1,
                    mode="bilinear",
                    align_corners=True,
                ),
                *args,
                **kwargs,
            )

            yhat = F.interpolate(
                yhat,
                size=x.shape[2:],
                mode="bilinear",
                align_corners=True,
            )

            output.append(yhat)

        return torch.stack(output, dim=0).max(0).values
