from typing import Iterator

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from tqdm import tqdm
import matplotlib.pyplot as plt


class ClassificationTrainer:
    def __init__(
        self,
        epoches: int,
        device: torch.device,
    ):
        self.epoches = epoches
        self.device = device

    @staticmethod
    def get_criterion(label_size: int):
        if label_size == 1:
            return nn.BCEWithLogitsLoss()
        elif label_size > 1:
            return nn.CrossEntropyLoss()
        else:
            raise ValueError(f"label size {label_size} is not valid")

    @staticmethod
    def get_optimizer(name: str, *args, **kwargs) -> Optimizer:
        optim_class = getattr(torch.optim, name)
        return optim_class(*args, **kwargs)

    def run(
        self,
        data: Iterator,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
    ):
        training_loss = []

        model.train()
        for e in range(self.epoches):
            running_loss = 0.0
            data_size = 0

            for x, y in tqdm(data):
                if isinstance(x, torch.Tensor):
                    data_size += x.size(0)
                elif isinstance(x, list):
                    data_size += len(x)

                optimizer.zero_grad()

                x = x.to(self.device)
                y = y.to(self.device)

                output = model(x)

                loss = criterion(output, y.long())
                running_loss += loss.item()

                loss.backward()
                optimizer.step()

            training_loss.append(running_loss / data_size)

            print(f"epoch {e+1} : {running_loss / data_size}")

        return training_loss
