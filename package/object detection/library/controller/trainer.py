import os
import random
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from adapter.adapter import DetectorLossAdapter, OptimizerAdapter
from albumentations.pytorch.transforms import ToTensorV2
from constants.enums import NetworkStage, NetworkType, OperationMode
from tensorboardX import SummaryWriter
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.train import collate_fn

from controller.controller import Controller


class Trainer(Controller):
    def get_detector_loss_kwargs(self, seen: int) -> dict:
        kwargs = {}

        if self.cfg.MODEL.NAME == "yolov2":
            kwargs = dict(seen=seen)

        return kwargs

    def set_preprocess(self, input_size: int):
        """
        Set preprocessing pipeline.
        Be careful, some transformations might drop targets.
        """
        if self.network_type == NetworkType.DETECTOR.value:
            preprocess = A.Compose(
                [
                    A.ColorJitter(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5, rotate_limit=0),
                    A.augmentations.geometric.resize.SmallestMaxSize(input_size),
                    A.RandomSizedBBoxSafeCrop(input_size, input_size),
                    A.Normalize(),
                    ToTensorV2(),
                ],
                bbox_params=A.BboxParams(
                    format="yolo",
                    # min_area=1024,
                    # min_visibility=0.3,
                ),
            )
        elif self.network_type == NetworkType.CLASSIFIER.value:
            preprocess = A.Compose(
                [
                    A.ColorJitter(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(p=0.5, rotate_limit=0),
                    A.augmentations.geometric.resize.SmallestMaxSize(input_size),
                    A.RandomResizedCrop(input_size, input_size),
                    A.Normalize(),
                    ToTensorV2(),
                ]
            )

        self.data[OperationMode.TRAIN.value]["preprocess"] = preprocess

    def load_loss(self):
        """
        Load loss

        for detector, only yolo loss provided.
        for classifier, only multiclass provided.
        """
        if self.network_type == NetworkType.DETECTOR.value:
            adapter = DetectorLossAdapter(self.get_detector_context())
            self.loss = adapter.get_loss()

        elif self.network_type == NetworkType.CLASSIFIER.value:
            num_classes = self.get_dataset_cfg().NUM_CLASSES
            if num_classes > 1:
                self.loss = nn.CrossEntropyLoss()
            else:
                self.loss = nn.BCEWithLogitsLoss()

    def load_optimizer(self):
        """Load optimizer, only adam & sgd provided."""
        optimizer_cfg = self.get_training_cfg().OPTIM
        self.lr = optimizer_cfg.LR
        self.decay = optimizer_cfg.DECAY
        self.momentum = optimizer_cfg.MOMENTUM

        adapter = OptimizerAdapter(optimizer_cfg)
        self.optimizer = adapter.get_optimizer(self.model.parameters())

    def load_scaler(self):
        """load scaler if amp enabled."""
        if self.cfg.HARDWARE.AMP:
            self.scaler = torch.cuda.amp.GradScaler()

    def train(self, description: str):
        """main function for train"""
        self.writer = SummaryWriter()

        training_cfg = self.get_training_cfg()

        self.prepare_train()

        start_epoch, end_epoch = training_cfg.START_EPOCH, training_cfg.END_EPOCH
        if (
            self.network_type == NetworkType.CLASSIFIER.value
            and self.stage == NetworkStage.FINETUNE.value
        ):
            start_epoch = end_epoch
            end_epoch += training_cfg.FINETUNE.EPOCH

        dataset_size = len(self.data[OperationMode.TRAIN.value]["dataset"])
        seen = start_epoch * dataset_size
        iter_num = start_epoch * dataset_size  # for tensorboard if multiscale enabled

        for e in range(start_epoch, end_epoch):
            print(f"Epoch {e+1}/{end_epoch}")
            running_loss = 0.0
            self.optimizer.zero_grad()

            # multiscale training in YOLOv2
            # compare to every 10 batch in paper, we use 10 or 1 epoch
            self.set_multiscale()

            loader = self.data[OperationMode.TRAIN.value]["loader"]
            for i, (imgs, labels) in enumerate(tqdm(loader)):
                iter_num += 1
                seen += imgs.size(0)

                imgs: torch.Tensor = imgs.to(self.device)

                if self.network_type == NetworkType.CLASSIFIER.value:
                    labels: torch.Tensor = labels.to(self.device)

                with autocast(enabled=self.cfg.HARDWARE.AMP):
                    output: torch.Tensor = self.model(imgs)

                    if self.network_type == NetworkType.DETECTOR.value:
                        kwargs = self.get_detector_loss_kwargs(seen)
                        loss: torch.Tensor = self.loss(output, labels, **kwargs)
                    elif self.network_type == NetworkType.CLASSIFIER.value:
                        loss: torch.Tensor = self.loss(output, labels)

                    loss = loss / self.acc_iter

                if self.scaler is None:
                    loss.backward()
                else:
                    self.scaler.scale(loss).backward()

                # gradient accumulation
                if (i + 1) % self.acc_iter == 0:
                    if self.scaler is None:
                        self.optimizer.step()
                    else:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    self.optimizer.zero_grad()

                running_loss += loss.item()

                # release gpu
                if "cuda" in self.device:
                    imgs = imgs.detach().cpu()

                # if not converge
                assert not np.isnan(running_loss), "loss died"

                # record batch loss
                self.writer.add_scalar(
                    "training loss",
                    loss.item(),
                    iter_num,
                    summary_description=description,
                )

            if self.scheduler:
                self.scheduler.step()

            # save model every 10 epoch
            if (e + 1) % self.save_interval == 0:
                self.save_weight(e + 1)

            print(running_loss / dataset_size)

        self.writer.close()

    def prepare_train(self):
        training_cfg = self.get_training_cfg()

        if self.network_type == NetworkType.DETECTOR.value:
            self.load_detector()
            self.set_preprocess(training_cfg.IMAGE_SIZE)

        elif self.network_type == NetworkType.CLASSIFIER.value:
            assert self.stage in [
                NetworkStage.FINETUNE.value,
                NetworkStage.SCRATCH.value,
            ], 'CLASSIFIER can be trained with stage ["finetune", "scratch"]'
            self.load_classifier(stage=self.stage)
            self.set_preprocess(
                training_cfg.FINETUNE.IMAGE_SIZE
                if self.stage == NetworkStage.FINETUNE.value
                else training_cfg.IMAGE_SIZE,
            )

        self.load_dataset(OperationMode.TRAIN.value)
        self.load_loss()
        self.load_optimizer()

        self.model.train()

    def set_multiscale(self):
        training_cfg = self.get_training_cfg()

        if self.network_type == NetworkType.DETECTOR.value and training_cfg.MULTISCALE:
            random_scale = random.randint(10, 19) * self.cfg.MODEL.SCALE
            self.set_preprocess(random_scale)
            self.load_dataset(OperationMode.TRAIN.value)
            if random_scale > training_cfg.IMAGE_SIZE:
                loader = DataLoader(
                    self.data[OperationMode.TRAIN.value]["dataset"],
                    batch_size=training_cfg.BATCH_SIZE // 2,
                    collate_fn=collate_fn,
                    shuffle=True,
                )
                self.data[OperationMode.TRAIN.value]["loader"] = loader

                self.acc_iter = 64 // training_cfg.BATCH_SIZE

    def save_weight(self, epoch: int):
        file_name = f"{self.cfg.MODEL.NAME}.{self.cfg.MODEL.BACKBONE}.{epoch}.pth"
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save(
            self.model.state_dict(),
            Path(self.save_dir).joinpath(file_name).as_posix(),
        )
