from math import log2
from typing import Iterable, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torchlake.common.controller.trainer import ClassificationTrainer
from torchvision.transforms import CenterCrop


class DeepLabV2Trainer(ClassificationTrainer):
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

        return [
            model(F.interpolate(x, size=int(x.size(3) * scale) + 1), *args, **kwargs)
            for scale in self.multi_scales
        ]

    def _calc_loss(
        self,
        y_hat: list[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y = y.to(self.device)
        cropper = CenterCrop(y.shape[2:])

        loss = 0
        fused = torch.zeros((y.size(0), y_hat[0].size(1), y.size(1), y.size(2))).to(
            self.device
        )
        for ele, scale in zip(y_hat, self.multi_scales):
            copied_y = y.clone()
            feature_interpolated = False
            # scale is 2's multiplier
            if scale == 1:
                pass
            elif int(log2(scale)) == log2(scale):
                copied_y = torch.nn.functional.interpolate(
                    copied_y.unsqueeze(1),
                    size=int(copied_y.size(2) * scale) + 1,
                    mode="nearest",
                ).squeeze(1)
            else:
                ele = F.interpolate(
                    ele,
                    scale_factor=1 / scale,
                    mode="bilinear",
                )
                feature_interpolated = True

            loss += criterion(ele, copied_y.long())

            if not feature_interpolated:
                ele = F.interpolate(
                    ele,
                    scale_factor=1 / scale,
                    mode="bilinear",
                )

            fused = torch.maximum(fused, cropper(ele))

        loss += criterion(fused, y.long())

        return loss

    def run(
        self,
        data: Iterator,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        scheduler: LRScheduler | None = None,
        *args,
        **kwargs,
    ) -> list[float]:
        assert hasattr(
            self, "multi_scales"
        ), "please set multi_scales by set_multiscales method first"

        return super().run(
            data, model, optimizer, criterion, scheduler, *args, **kwargs
        )
