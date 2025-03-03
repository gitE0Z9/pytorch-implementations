from abc import ABC
from typing import Callable, Generator, Iterable
import torch
from tqdm import tqdm
from torch.optim import Optimizer
from torch import nn

from torchlake.common.controller.recorder import TrainRecorder


class GANTrainer(ABC):
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

    def get_batch_size(self, row: list[Iterable]) -> int:
        if isinstance(row, list):
            return len(row[0])

        return len(row)

    def train_discriminator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        valid: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        img, _ = row
        img = img.to(self.device)

        gen_img = generator(noise)
        real_loss = criterion(discriminator(img), valid)
        fake_loss = criterion(discriminator(gen_img.detach()), 1 - valid)
        return (real_loss + fake_loss) / 2

    def train_generator(
        self,
        noise: torch.Tensor,
        valid: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        gen_img = generator(noise)
        return criterion(discriminator(gen_img), valid)

    def run(
        self,
        data,
        noise_generator: Generator[torch.Tensor, None, None],
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        criterion: nn.Module,
        scheduler_g=None,
        scheduler_d=None,
        scaler=None,
        recorder: TrainRecorder | None = None,
        validate_func: Callable[[nn.Module, nn.Module], None] | None = None,
        checkpoint_func: (
            Callable[[nn.Module, nn.Module, Optimizer, Optimizer], None] | None
        ) = None,
        *args,
        **kwargs,
    ):
        # amp
        torch.set_autocast_enabled(scaler is not None)
        print(f"Enable AMP: {torch.is_autocast_enabled()}")

        # TODO: overhead
        if recorder is None:
            print("Calculating dataset size...")
            self.recorder.calc_dataset_size(data)

        for e in range(self.recorder.current_epoch, self.recorder.total_epoch):
            for row in tqdm(data):
                batch_size = self.get_batch_size(row)
                valid = torch.ones(batch_size, 1).to(self.device)

                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                discriminator.train()
                generator.eval()
                d_loss = self.train_discriminator(
                    row,
                    next(noise_generator(batch_size)),
                    valid,
                    generator,
                    discriminator,
                    criterion,
                )
                assert not torch.isnan(d_loss), "Loss is singular"
                d_loss.backward()
                optimizer_d.step()

                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                generator.train()
                discriminator.eval()
                g_loss = self.train_generator(
                    next(noise_generator(batch_size)),
                    valid,
                    generator,
                    discriminator,
                    criterion,
                )
                assert not torch.isnan(g_loss), "Loss is singular"
                g_loss.backward()
                optimizer_g.step()

                recorder.increment_running_loss(
                    *(loss.item() / recorder.data_size for loss in [d_loss, g_loss])
                )

            recorder.enqueue_training_loss()

            print(
                f"epoch {e+1}: "
                f"D: {recorder.training_losses[0][-1]} "
                f"G: {recorder.training_losses[1][-1]}"
            )
            recorder.increment_epoch()

            if validate_func is not None and (e + 1) % self.validate_interval == 0:
                print("Validating...")
                validate_func(generator, discriminator)

            if checkpoint_func is not None and (e + 1) % self.checkpoint_interval == 0:
                print("Checkpoint...")
                checkpoint_func(generator, discriminator, optimizer_g, optimizer_d)

        return recorder.training_losses
