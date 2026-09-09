from typing import Any, Iterable

import torch
from torch import nn

from torchlake.common.controller.trainer import ClassificationTrainer


class FastTextTrainer(ClassificationTrainer):
    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        *args,
        **kwargs,
    ) -> torch.Tensor | Any:
        x, _ = row
        # x is a tuple of (ngrams, words, word_spans)
        _x = (
            [it.to(self.device) for it in x[0]],
            x[1].to(self.device),
            [it.to(self.device) for it in x[2]],
        )

        return model(*_x, *args, **kwargs)
