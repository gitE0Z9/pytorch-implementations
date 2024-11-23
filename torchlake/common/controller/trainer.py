from abc import ABC, abstractmethod
from typing import Iterable, Iterator

import torch
from torch import nn
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm

from ..mixins.controller import PredictFunctionMixin


class TrainerBase(PredictFunctionMixin, ABC):
    def __init__(
        self,
        epoches: int,
        device: torch.device,
        acc_iters: int = 1,
        feature_last: bool = False,
    ):
        """Base class of trainer

        Args:
            epoches (int): how many epoch to run
            device (torch.device): which device to use
            acc_iters (int, optional): how many epoch to finish gradient accumulation. Defaults to 1.
            feature_last (bool, optional): do we need to move index -1 of output to index 1, default value intends to work with image and entropy loss. Defaults to False.
        """
        self.epoches = epoches
        self.device = device
        self.acc_iters = acc_iters
        self.feature_last = feature_last

    def get_criterion(self):
        raise NotImplementedError

    @staticmethod
    def get_optimizer(name: str, *args, **kwargs) -> Optimizer:
        optim_class = getattr(torch.optim, name)
        return optim_class(*args, **kwargs)

    @abstractmethod
    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        raise NotImplementedError

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
        if not hasattr(self, "_predict"):
            self.build_predict_function_by_data_type(iter(data))

        training_loss = []
        model.train()
        model = model.to(self.device)  # some model extended layer when train
        for e in range(self.epoches):
            running_loss = 0.0
            data_size = 0
            optimizer.zero_grad()

            for i, row in enumerate(tqdm(data)):
                # get x
                # case 1: row is a list, e.g. features and labels
                # case 2: row is not a list, e.g. features only or features also serve as labels
                if isinstance(row, list):
                    x = row[0]
                else:
                    x = row

                # get batch size to calculate dataset size
                # hard to choose if running once before or dynamically like this
                # since the former will run an empty cycle
                # the latter will waste resource when dataset size is fixed
                if isinstance(x, torch.Tensor):
                    data_size += x.size(0)
                elif isinstance(x, list | tuple | set):
                    data_size += len(x)

                output = self._predict(row, model, *args, **kwargs)
                loss: torch.Tensor = self._calc_loss(output, row, criterion)

                loss /= self.acc_iters
                assert not torch.isnan(loss)
                loss.backward()
                running_loss += loss.item()

                if (i + 1) % self.acc_iters == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            mean_loss = running_loss / data_size
            training_loss.append(mean_loss)

            if scheduler:
                scheduler.step(mean_loss)

            print(f"epoch {e+1} : {mean_loss}")

        return training_loss


class DoNothingTrainer(TrainerBase):
    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row

        return criterion(y_hat, y)


class ClassificationTrainer(TrainerBase):
    @staticmethod
    def get_criterion(
        label_size: int, *args, **kwargs
    ) -> nn.BCEWithLogitsLoss | nn.CrossEntropyLoss:
        if label_size == 1:
            return nn.BCEWithLogitsLoss(*args, **kwargs)
        elif label_size > 1:
            return nn.CrossEntropyLoss(*args, **kwargs)
        else:
            raise ValueError(f"label size {label_size} is not valid")

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        return criterion(y_hat, y.long())


class RegressionTrainer(TrainerBase):
    @staticmethod
    def get_criterion() -> nn.MSELoss | nn.SmoothL1Loss:
        return nn.MSELoss()

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        return criterion(y_hat, y.float())


class MutltOutputClassificationTrainer(ClassificationTrainer):
    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        return criterion(*y_hat, y.long())
