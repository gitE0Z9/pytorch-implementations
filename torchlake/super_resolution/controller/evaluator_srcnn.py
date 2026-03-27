# from typing import Any, Iterable

import torch

from torchlake.common.controller.evaluator import RegressionEvaluator


class SRCNNEvaluator(RegressionEvaluator):
    def set_scale_factor(self, scale_factor: float):
        self.scale_factor = scale_factor

    #     def _predict(
    #         self,
    #         row: tuple[Iterable],
    #         model: nn.Module,
    #         *args,
    #         **kwargs,
    #     ) -> torch.Tensor | Any:
    #         x, _ = row
    #         x: torch.Tensor = x.to(self.device)
    #         output = model(x[:, 0:1, :, :], *args, **kwargs)
    #         # for output that has feature in last
    #         if self.feature_last:
    #             output = output.permute(0, -1, *range(1, len(output.shape) - 1))

    #         output = torch.cat((output, x[:, 1:3]), 1)
    #         return output

    def _update_metric(self, metric, y: torch.Tensor, yhat: torch.Tensor):
        metric.update(
            yhat.detach()
            .cpu()
            .mul(255)
            .int()[
                :,
                :,
                self.scale_factor : -self.scale_factor,
                self.scale_factor : -self.scale_factor,
            ],
            y.mul(255).int()[
                :,
                :,
                self.scale_factor : -self.scale_factor,
                self.scale_factor : -self.scale_factor,
            ],
        )
