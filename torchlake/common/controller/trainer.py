from abc import ABC
from typing import Iterable, Iterator

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm


class TrainerBase(ABC):
    def __init__(self, epoches: int, device: torch.device):
        self.epoches = epoches
        self.device = device

    def get_criterion(self):
        raise NotImplementedError

    @staticmethod
    def get_optimizer(name: str, *args, **kwargs) -> Optimizer:
        optim_class = getattr(torch.optim, name)
        return optim_class(*args, **kwargs)

    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        criterion: nn.Module,
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def run(
        self,
        data: Iterator,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        *args,
        **kwargs,
    ):
        training_loss = []

        model.train()
        for e in range(self.epoches):
            running_loss = 0.0
            data_size = 0

            for row in tqdm(data):
                x = row[0]
                if isinstance(x, torch.Tensor):
                    data_size += x.size(0)
                elif isinstance(x, list):
                    data_size += len(x)

                optimizer.zero_grad()

                loss = self._predict(row, model, criterion, *args, **kwargs)
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            training_loss.append(running_loss / data_size)

            print(f"epoch {e+1} : {running_loss / data_size}")

        return training_loss


class ClassificationTrainer(TrainerBase):
    @staticmethod
    def get_criterion(label_size: int):
        if label_size == 1:
            return nn.BCEWithLogitsLoss()
        elif label_size > 1:
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"label size {label_size} is not valid")

    def _predict(
        self,
        row: tuple[Iterable],
        model: nn.Module,
        criterion: nn.Module,
        *args,
        **kwargs,
    ):
        x, y = row
        x = x.to(self.device)
        y = y.to(self.device)

        output = model(x)

        loss = criterion(output, y.long())

        return loss
