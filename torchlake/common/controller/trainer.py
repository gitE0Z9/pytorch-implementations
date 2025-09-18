from abc import ABC, abstractmethod
from typing import Callable, Iterable, Iterator, Literal

import torch
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from tqdm import tqdm


from ..mixins.controller import PredictFunctionMixin
from .recorder import TrainRecorder


class TrainerBase(PredictFunctionMixin, ABC):
    def __init__(
        self,
        epoches: int = 10,
        device: torch.device = torch.device("cuda:0"),
        acc_iters: int = 1,
        feature_last: bool = False,
        validate_interval: int = 10,
        checkpoint_interval: int = 10,
    ):
        """Base class of trainer

        Args:
            epoches (int, optional): how many epoch to run, if no recorder then use this number as total epoch of this run. Defaults to 10.
            device (torch.device, optional): which device to use. Defaults to torch.device("cuda:0").
            acc_iters (int, optional): how many epoch to finish gradient accumulation. Defaults to 1.
            feature_last (bool, optional): do we need to move index -1 of output to index 1, default value intends to work with image and entropy loss. Defaults to False.
            validate_interval (int, optional): after how many epoch to run validate function. Defaults to 10.
            checkpoint_interval (int, optional): after how many epoch to save checkpint. Defaults to 10.
        """
        self.epoches = epoches
        self.device = torch.device(device)
        self.acc_iters = acc_iters
        self.feature_last = feature_last
        self.validate_interval = validate_interval
        self.checkpoint_interval = checkpoint_interval
        self.recorder = TrainRecorder(total_epoch=self.epoches)

    def get_criterion(self):
        raise NotImplementedError

    # @staticmethod
    # def get_optimizer(name: str, *args, **kwargs) -> Optimizer:
    #     optim_class = getattr(torch.optim, name)
    #     return optim_class(*args, **kwargs)

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
        scaler: GradScaler | None = None,
        recorder: TrainRecorder | None = None,
        validate_func: Callable[[nn.Module], None] | None = None,
        checkpoint_func: (
            Callable[[nn.Module, Optimizer, LRScheduler | None], None] | None
        ) = None,
        at_batch_end_hook: Callable[[int], None] | None = None,
        at_epoch_end_hook: Callable | None = None,
        is_early_stoppable_func: Callable[[], bool] | None = None,
        *args,
        **kwargs,
    ) -> list[float]:
        # predict strategy
        if not hasattr(self, "_predict"):
            self.build_predict_function_by_data_type(iter(data))

        # amp
        torch.set_autocast_enabled(scaler is not None)
        print(f"Enable AMP: {torch.is_autocast_enabled()}")

        if recorder is None:
            recorder = self.recorder
            if recorder.is_static_dataset and recorder.data_size <= 0:
                print("Calculating dataset size...")
                recorder.calc_dataset_size(data)

        model.train()
        # some models have extra layers during training
        # so move again for sure
        model = model.to(self.device)

        print("Training...")
        for e in range(recorder.current_epoch, recorder.total_epoch):
            optimizer.zero_grad()
            recorder.reset_running_loss()
            if not recorder.is_static_dataset:
                recorder.reset_data_size()

            for batch_idx, row in enumerate(tqdm(data)):
                row_size = recorder.calc_batch_size(row)
                if not recorder.is_static_dataset:
                    recorder.increment_data_size(row_size)

                with torch.autocast(
                    device_type=self.device.type,
                    enabled=torch.is_autocast_enabled(),
                ):
                    # predict and loss calculation
                    output = self._predict(row, model, *args, **kwargs)
                    losses = self._calc_loss(output, row, criterion)
                if not isinstance(losses, tuple):
                    losses: tuple[torch.Tensor] = (losses,)

                losses = tuple(loss / self.acc_iters for loss in losses)
                for loss in losses:
                    assert not torch.isnan(loss), f"Loss is singular. Losses: {losses}"

                # gradient backward
                main_loss = losses[0]
                # mixed precision
                if scaler is None:
                    main_loss.backward()
                else:
                    scaler.scale(main_loss).backward()

                # weight update
                if (batch_idx + 1) % self.acc_iters == 0:
                    # mixed precision
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()

                    optimizer.zero_grad()

                recorder.increment_running_loss(
                    *(loss.item() / recorder.data_size for loss in losses)
                )

                if at_batch_end_hook is not None:
                    at_batch_end_hook(batch_idx)

            if at_epoch_end_hook is not None:
                at_epoch_end_hook()

            recorder.enqueue_training_loss()

            last_loss = recorder.training_losses[0][-1]
            if scheduler:
                scheduler.step(last_loss)

            recorder.display_epoch_result()
            recorder.increment_epoch()

            if validate_func is not None and (e + 1) % self.validate_interval == 0:
                print("Validating...")
                validate_func(model)

            if checkpoint_func is not None and (e + 1) % self.checkpoint_interval == 0:
                print("Checkpoint...")
                checkpoint_func(model, optimizer, scheduler)

            if is_early_stoppable_func is not None and is_early_stoppable_func():
                print(f"early stopped at epoch {e+1}")
                break

        return recorder.training_losses[0]


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
    def get_criterion(
        type: Literal["l1", "l2", "smoothl1", "huber"],
    ) -> nn.MSELoss | nn.SmoothL1Loss | nn.HuberLoss | nn.L1Loss:
        return {
            "l1": nn.L1Loss(),
            "l2": nn.MSELoss(),
            "smoothl1": nn.SmoothL1Loss(),
            "huber": nn.HuberLoss(),
        }[type]

    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        return criterion(y_hat, y.float())


class MultiOutputClassificationTrainer(ClassificationTrainer):
    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        _, y = row
        y: torch.Tensor = y.to(self.device)

        return criterion(*y_hat, y.long())


class ShareInputOutputClassificationTrainer(ClassificationTrainer):
    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, _ = row
        x: torch.Tensor = x.to(self.device)

        return criterion(y_hat, x.long())


class ShareInputOutputRegressionTrainer(RegressionTrainer):
    def _calc_loss(
        self,
        y_hat: torch.Tensor,
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, _ = row
        x: torch.Tensor = x.to(self.device)

        return criterion(y_hat, x.float())


class SingleInputMultiOutputClassificationTrainer(ClassificationTrainer):
    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, _ = row
        x: torch.Tensor = x.to(self.device)

        return criterion(*y_hat, x.long())


class SingleInputMultiOutputRegressionTrainer(RegressionTrainer):
    def _calc_loss(
        self,
        y_hat: tuple[torch.Tensor],
        row: tuple[Iterable],
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, _ = row
        x: torch.Tensor = x.to(self.device)

        return criterion(*y_hat, x.float())
