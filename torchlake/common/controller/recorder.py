import json
from pathlib import Path
from typing import Iterable, Sequence

import torch
from tqdm import tqdm


class TrainRecorder:
    def __init__(
        self,
        current_epoch: int = 0,
        total_epoch: int = 1,
        num_loss: int = 1,
        loss_names: Sequence[str] = ["total"],
        is_static_dataset: bool = True,
    ):
        """recorder of training progress

        Args:
            current_epoch (int, optional): current epoch. Defaults to 0.
            total_epoch (int, optional): total epoch. Defaults to 1.
            num_loss (int, optional): how many losses term to record. Defaults to 1.
            loss_names (Sequence[str], optional): names of each loss term. Defaults to ["total"].
            is_static_dataset (bool, optional): if False, calculate the dataset size during training. Defaults to True.
        """
        self.current_epoch = current_epoch
        self.total_epoch = total_epoch

        if len(loss_names) != num_loss:
            loss_names = [*loss_names[:num_loss]]
            for i in range(num_loss - len(loss_names)):
                loss_names.append(f"subloss {i+1}")
        self.num_loss = num_loss
        self.loss_names = loss_names
        self.is_static_dataset = is_static_dataset

        self.current_dataset_size = 0
        self.dataset_size = 0
        self.reset_running_loss()
        self.reset_training_loss()

    def reset_epoch(self):
        self.current_epoch = 0

    def increment_epoch(self):
        self.current_epoch += 1

    def is_final_epoch(self) -> bool:
        return self.current_epoch >= self.total_epoch

    def reset_dataset_size(self):
        self.dataset_size = 0

    def reset_current_dataset_size(self):
        self.current_dataset_size = 0

    def increment_dataset_size(self, size: int):
        self.dataset_size += size

    def increment_current_dataset_size(self, size: int):
        self.current_dataset_size += size

    def calc_batch_size(self, row: tuple) -> int:
        # get x
        # case 1: row is a list, e.g. features and labels
        # case 2: row is not a list, e.g. features only or features also serve as labels
        if isinstance(row, list | tuple | set):
            x = row[0]
        else:
            x = row

        # get batch size to calculate dataset size
        # hard to choose if running once before or dynamically like this
        # since the former will run an empty cycle
        # the latter will waste resource when dataset size is fixed
        count = 0
        if isinstance(x, torch.Tensor):
            count = x.size(0)
        elif isinstance(x, list | tuple | set):
            count = len(x)

        return count

    def calc_dataset_size(self, dataset: Iterable) -> int:
        self.reset_dataset_size()

        for row in tqdm(dataset):
            count = self.calc_batch_size(row)
            self.increment_dataset_size(count)

        return self.dataset_size

    def reset_running_loss(self):
        self.running_losses: list[float] = [0.0] * self.num_loss

    def increment_running_loss(self, *losses: float):
        for i in range(self.num_loss):
            self.running_losses[i] += losses[i]

    def reset_training_loss(self):
        self.training_losses: list[list[float]] = [[] for _ in range(self.num_loss)]

    def enqueue_training_loss(self):
        for i in range(self.num_loss):
            self.training_losses[i].append(self.running_losses[i])

    def get_last_improvement(self, decimal: int = 2) -> list[int]:
        ratios = []
        for training_loss in self.training_losses:
            ratio = 0
            if len(training_loss) > 1:
                try:
                    ratio = training_loss[-1] / training_loss[-2] - 1
                    ratio = round(10**decimal * ratio)
                except ZeroDivisionError:
                    ratio = 0

            ratios.append(ratio)

        return ratios

    def display_epoch_result(self):
        print("------------------------------------")
        print(f"Epoch {self.current_epoch+1}")
        print("------------------------------------")
        for loss_name, training_loss, last_improvement in zip(
            self.loss_names, self.training_losses, self.get_last_improvement()
        ):
            print(f"{loss_name}: {training_loss[-1]:.4e} ({last_improvement:.2f}%)")

    def to_state_dict(self, path: str | Path):
        path = Path(path)
        state = {
            "current_epoch": self.current_epoch,
            "total_epoch": self.total_epoch,
            "current_dataset_size": self.current_dataset_size,
            "dataset_size": self.dataset_size,
            "num_loss": self.num_loss,
            "loss_names": self.loss_names,
            "running_losses": self.running_losses,
            "training_losses": self.training_losses,
        }
        path.write_text(json.dumps(state))

    def load_state_dict(self, path: str | Path):
        path = Path(path)
        state = json.loads(path.read_text())
        self.current_epoch = state["current_epoch"]
        self.total_epoch = state["total_epoch"]
        self.current_dataset_size = state["current_dataset_size"]
        self.dataset_size = state["dataset_size"]
        self.num_loss = state["num_loss"]
        self.loss_names = state["loss_names"]
        self.running_losses = state["running_losses"]
        self.training_losses = state["training_losses"]


class EvalRecorder: ...
