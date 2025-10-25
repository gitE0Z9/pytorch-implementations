from typing import Callable, Generator, Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from torchlake.common.controller.recorder import TrainRecorder


class GANTrainer:
    def __init__(
        self,
        epoches: int = 10,
        device: torch.device = torch.device("cuda:0"),
        acc_iters: int = 1,
        feature_last: bool = False,
        validate_interval: int = 10,
        checkpoint_interval: int = 10,
    ):
        """Trainer of GAN

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
        self.discriminator_cycle = 1

    def set_discriminator_cycle(self, value: int):
        self.discriminator_cycle = value

    def train_discriminator(
        self,
        row: tuple[Iterable],
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, _ = row
        x = x.to(self.device)

        with torch.no_grad():
            xhat = generator(noise)

        return criterion(discriminator(x), discriminator(xhat))

    def train_generator(
        self,
        noise: torch.Tensor,
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        xhat = generator(noise)
        return criterion(discriminator(xhat))

    def run(
        self,
        data,
        noise_generator: Generator[torch.Tensor, None, None],
        generator: nn.Module,
        discriminator: nn.Module,
        optimizer_g: Optimizer,
        optimizer_d: Optimizer,
        criterion_g: nn.Module,
        criterion_d: nn.Module,
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

        if recorder is None:
            recorder = self.recorder
            if recorder.is_static_dataset and recorder.data_size <= 0:
                print("Calculating dataset size...")
                recorder.calc_dataset_size(data)

        for e in range(recorder.current_epoch, recorder.total_epoch):
            for batch_idx, batch in enumerate(tqdm(data)):
                batch_size = recorder.calc_batch_size(batch)
                noise = noise_generator(batch_size)

                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                discriminator.train()
                generator.train()
                d_loss = self.train_discriminator(
                    batch,
                    next(noise),
                    generator,
                    discriminator,
                    criterion_d,
                )
                assert not torch.isnan(d_loss), "Loss is singular"
                d_loss.backward()
                optimizer_d.step()

                if (batch_idx + 1) % self.discriminator_cycle == 0:
                    optimizer_d.zero_grad()
                    optimizer_g.zero_grad()
                    generator.train()
                    discriminator.train()
                    g_loss = self.train_generator(
                        next(noise),
                        generator,
                        discriminator,
                        criterion_g,
                    )
                    assert not torch.isnan(g_loss), "Loss is singular"
                    g_loss.backward()
                    optimizer_g.step()

                    recorder.increment_running_loss(
                        *(loss.item() / recorder.data_size for loss in (d_loss, g_loss))
                    )

            recorder.enqueue_training_loss()

            recorder.display_epoch_result()
            recorder.increment_epoch()

            if validate_func is not None and (e + 1) % self.validate_interval == 0:
                print("Validating...")
                validate_func(generator, discriminator)

            if checkpoint_func is not None and (e + 1) % self.checkpoint_interval == 0:
                print("Checkpoint...")
                checkpoint_func(generator, discriminator, optimizer_g, optimizer_d)

        return recorder.training_losses
