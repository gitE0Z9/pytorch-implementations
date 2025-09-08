from typing import Iterable, Sequence

import torch
from tqdm import tqdm


class TrainRecorder:
    def __init__(
        self,
        current_epoch: int = 0,
        total_epoch: int = 1,
        num_loss: int = 1,
        is_data_size_static: bool = True,
        loss_names: Sequence[str] = ["total"],
    ):
        """_summary_

        Args:
            current_epoch (int, optional): current epoch. Defaults to 0.
            total_epoch (int, optional): total epoch. Defaults to 1.
            num_loss (int, optional): how many losses term to record. Defaults to 1.
            is_data_size_static (bool, optional): if calculate during training. Defaults to True.
        """
        self.current_epoch = current_epoch
        self.total_epoch = total_epoch

        if len(loss_names) != num_loss:
            loss_names = [*loss_names[:num_loss]]
            for i in range(num_loss - len(loss_names)):
                loss_names.append(f"subloss {i+1}")
        self.num_loss = num_loss
        self.loss_names = loss_names
        self.is_static_dataset = is_data_size_static

        self.current_data_size = 0
        self.data_size = 0
        self.reset_running_loss()
        self.reset_training_loss()

    def reset_epoch(self):
        self.current_epoch = 0

    def increment_epoch(self):
        self.current_epoch += 1

    def is_final_epoch(self) -> bool:
        return self.current_epoch >= self.total_epoch

    def reset_data_size(self):
        self.data_size = 0

    def reset_current_data_size(self):
        self.current_data_size = 0

    def increment_data_size(self, size: int):
        self.data_size += size

    def increment_current_data_size(self, size: int):
        self.current_data_size += size

    def calc_batch_size(self, row: tuple) -> int:
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
        count = 0
        if isinstance(x, torch.Tensor):
            count = x.size(0)
        elif isinstance(x, list | tuple | set):
            count = len(x)

        return count

    def calc_dataset_size(self, dataset: Iterable) -> int:
        self.reset_data_size()

        for row in tqdm(dataset):
            count = self.calc_batch_size(row)
            self.increment_data_size(count)

        return self.data_size

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


class EvalRecorder: ...
