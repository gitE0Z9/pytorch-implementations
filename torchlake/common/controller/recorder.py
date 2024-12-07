from typing import Iterable

import torch


class TrainRecorder:
    def __init__(
        self,
        current_epoch: int = 0,
        total_epoch: int = 1,
        num_loss: int = 1,
        is_data_size_static: bool = True,
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

        self.num_loss = num_loss
        self.is_data_size_static = is_data_size_static

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

    def increment_data_size(self, size: int):
        self.data_size += size

    def calc_row_size(self, row: tuple) -> int:
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

        for row in dataset:
            count = self.calc_row_size(row)
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
                ratio = training_loss[-1] / training_loss[-2] - 1
                ratio = round(10**decimal * ratio)

            ratios.append(ratio)

        return ratios


class EvalRecorder: ...


# class Trainer:
#     def run(
#         self,
#         data: Iterator,
#         model: nn.Module,
#         optimizer: Optimizer,
#         criterion: nn.Module,
#         scheduler: LRScheduler | None = None,
#         recorder: TrainRecorder | None = None,
#         *args,
#         **kwargs,
#     ) -> list[float]:
#         if not hasattr(self, "_predict"):
#             self.build_predict_function_by_data_type(iter(data))

#         model.train()
#         model = model.to(self.device)  # some model extended layer when train
#         for e in range(self.epoches):
#             recorder.reset_running_loss()
#             recorder.reset_data_size()
#             optimizer.zero_grad()

#             for i, row in enumerate(tqdm(data)):
#                 # get x
#                 # case 1: row is a list, e.g. features and labels
#                 # case 2: row is not a list, e.g. features only or features also serve as labels
#                 if isinstance(row, list):
#                     x = row[0]
#                 else:
#                     x = row

#                 # get batch size to calculate dataset size
#                 # hard to choose if running once before or dynamically like this
#                 # since the former will run an empty cycle
#                 # the latter will waste resource when dataset size is fixed
#                 if isinstance(x, torch.Tensor):
#                     recorder.increment_data_size(x.size(0))
#                 elif isinstance(x, list | tuple | set):
#                     recorder.increment_data_size(len(x))

#                 output = self._predict(row, model, *args, **kwargs)
#                 loss = self._calc_loss(output, row, criterion)

#                 if isinstance(loss, tuple):
#                     recorder.increment_running_loss(*loss)
#                 else:
#                     recorder.increment_running_loss(loss)
#                 # first loss must be main loss
#                 # if there are other loss, they will be recorded
#                 # if recorder.num_loss > 1:
#                 #     recorder.
#                 #     other_loss = loss[1:]
#                 #     for loss_i in range(1, self.num_loss):
#                 #         running_loss[loss_i] += other_loss[loss_i].item()
#                 #     loss = loss[0]

#                 loss /= self.acc_iters
#                 assert not torch.isnan(loss)
#                 loss.backward()
#                 running_loss[0] += loss.item()

#                 if (i + 1) % self.acc_iters == 0:
#                     optimizer.step()
#                     optimizer.zero_grad()

#             for loss_i in range(self.num_loss):
#                 mean_loss = running_loss[loss_i] / data_size
#                 training_loss[loss_i].append(mean_loss)

#             main_mean_loss = training_loss[0][-1]

#             if scheduler:
#                 scheduler.step(main_mean_loss)

#             print(f"epoch {e+1} : {main_mean_loss}")

#         return training_loss
