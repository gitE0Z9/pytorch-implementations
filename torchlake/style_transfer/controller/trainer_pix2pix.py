from typing import Callable, Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import tqdm

from torchlake.image_generation.controllers.trainer_gan import GANTrainer
from torchlake.common.controller.recorder import TrainRecorder

__all__ = [
    "Pix2PixTrainer",
]


class Pix2PixTrainer(GANTrainer):

    def train_discriminator(
        self,
        row: tuple[Iterable],
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, w = row
        x = x.to(self.device)
        w = w.to(self.device)

        with torch.no_grad():
            what = generator(x)

        return criterion(discriminator(what, x), discriminator(w, x))

    def train_generator(
        self,
        row: tuple[Iterable],
        generator: nn.Module,
        discriminator: nn.Module,
        criterion: nn.Module,
    ) -> torch.Tensor:
        x, w = row
        x = x.to(self.device)
        w = w.to(self.device)

        what = generator(x)
        return criterion(discriminator(what, x), what, w)

    def run(
        self,
        data,
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
                optimizer_d.zero_grad()
                optimizer_g.zero_grad()
                discriminator.train()
                generator.train()

                d_loss = self.train_discriminator(
                    batch,
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
                        batch,
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
