from importlib import import_module

import albumentations as A
import torch
import torch.nn as nn
import torchvision
from albumentations.pytorch.transforms import ToTensorV2
from configs.schema import (
    ClassifierTrainingCfg,
    DatasetCfg,
    DetectorTrainingCfg,
    Setting,
)
from datasets.imagenet.datasets import ImageNetDataset
from datasets.voc.datasets import VOCDatasetFromCSV
from utils.config import load_anchors, load_classes, load_config
from utils.plot import rand_color
from utils.train import collate_fn

from constants.enums import NetworkStage, NetworkType


class Controller:
    def __init__(
        self,
        cfg_path: str,
        dataset_name: str,
        network_type: str,
        stage: str = None,
    ):
        # check network mode
        self.network_type = network_type.upper()
        self.assert_network_type()
        if self.network_type == NetworkType.CLASSIFIER.value:
            warning_message = f'CLASSIFIER need to be assigned a stage in ["finetune", "scratch", "inference"]'
            assert stage.upper() in NetworkStage.__members__, warning_message
            self.stage = stage.lower()

        # cfg
        self.cfg = Setting(**load_config(cfg_path))
        self.device = self.cfg.HARDWARE.DEVICE
        self.dataset_name = dataset_name.upper()

        dataset_cfg = self.get_dataset_cfg()
        self.class_names = load_classes(dataset_cfg.CLASSES_PATH)

        anchor_path = self.cfg.MODEL.ANCHORS_PATH
        self.anchors = (
            load_anchors(anchor_path).to(self.device) if anchor_path else None
        )
        # self.anchors *= self.cfg['TRAIN'][self.mode]['IMAGE_SIZE'] / self.cfg['MODEL']['SCALE']

        self.model = None
        self.data = {
            "train": {"preprocess": None, "dataset": None, "loader": None},
            "test": {"preprocess": None, "dataset": None, "loader": None},
        }
        self.loss = None
        self.optimizer = None
        self.scaler = None
        self.scheduler = None

        training_cfg = self.get_training_cfg()
        self.acc_iter = training_cfg.ACC_ITER
        self.save_interval = training_cfg.SAVE.INTERVAL
        self.save_dir = training_cfg.SAVE.DIR
        self.palette = {name: rand_color() for name in self.class_names}

    def assert_network_type(self):
        warning_message = (
            f"Only {','.join(item.value for item in NetworkType)} supported."
        )
        assert self.network_type in NetworkType.__members__, warning_message

    def get_training_cfg(self) -> DetectorTrainingCfg | ClassifierTrainingCfg:
        return getattr(self.cfg.TRAIN, self.network_type)

    def get_dataset_cfg(self) -> DatasetCfg:
        return getattr(self.cfg.DATA, self.dataset_name)

    def get_classifier_class(self) -> nn.Module:
        backbone = self.cfg.MODEL.BACKBONE

        if backbone.startswith("resnet"):
            model_class = getattr(torchvision.models, backbone)
        else:
            classifier_module = import_module(f"models.{self.cfg.MODEL.NAME}.network")
            model_class = getattr(
                classifier_module, self.cfg.MODEL.BACKBONE.capitalize()
            )

        return model_class

    def get_detector_class(self) -> nn.Module:
        backbone = self.cfg.MODEL.BACKBONE

        if backbone.startswith("resnet"):
            class_name = f"{self.cfg.MODEL.NAME.capitalize()}Resnet"
        else:
            class_name = f"{self.cfg.MODEL.NAME.capitalize()}"

        detector_module = import_module(f"models.{self.cfg.MODEL.NAME}.detector")
        model_class = getattr(detector_module, class_name)

        return model_class

    def load_classifier(self, stage: str):
        """Load classifier with corresponding network stage."""

        model_cfg = self.cfg.MODEL

        model_class = self.get_classifier_class()

        # load trained weight
        if stage == NetworkStage.INFERENCE.value:
            self.model: nn.Module = model_class()
            self.load_weight(model_cfg.CLASSIFIER_PATH)
        # load imagenet weight
        elif stage == NetworkStage.FINETUNE.value:
            if model_cfg.BACKBONE.startswith("resnet"):
                self.model = model_class(weights=True)
            else:
                self.model: nn.Module = model_class()
                self.load_weight(model_cfg.CLASSIFIER_PATH)
        # load random weight
        elif stage == NetworkStage.SCRATCH.value:
            self.model: nn.Module = model_class()

        self.model = self.model.to(self.device)

    def load_detector(self):
        """Load detector"""

        model_cfg = self.cfg.MODEL
        backbone = model_cfg.BACKBONE
        detector_path = model_cfg.DETECTOR_PATH

        number_of_anchors = model_cfg.NUM_ANCHORS
        number_of_classes = self.cfg.DATA.VOC.NUM_CLASSES

        detector_class = self.get_detector_class()

        if backbone.startswith("resnet"):
            # TODO: support higher layer number
            number_of_layers = int(backbone[-2:])

            self.model = detector_class(
                number_of_layers,
                number_of_anchors,
                number_of_classes,
                finetune_weight=model_cfg.CLASSIFIER_PATH,
            )

        else:
            backbone_class = self.get_classifier_class()
            backbone: nn.Module = backbone_class()
            self.model = detector_class(
                backbone,
                number_of_anchors,
                number_of_classes,
                finetune_weight=model_cfg.CLASSIFIER_PATH,
            )
        # else:
        #     raise NotImplementedError(f"{backbone} backbone is not supported.")

        if detector_path:
            self.load_weight(detector_path)

        self.model = self.model.to(self.device)

    def load_weight(self, weight_path: str):
        """Load weight"""
        assert weight_path, "Please provide weight path."

        self.model.load_state_dict(torch.load(weight_path))

    def load_dataset(self, mode: str):
        """
        Load dataset

        args: mode, can be train or test
              dataset, can be voc, imagenet, coco

        """
        if self.dataset_name == "VOC":
            batch = self.cfg.TRAIN.DETECTOR.BATCH_SIZE

            self.data[mode]["dataset"] = VOCDatasetFromCSV(
                root=self.cfg.DATA.VOC.ROOT,
                csv_root=self.cfg.DATA.VOC.CSV_ROOT,
                class_name=self.class_names,
                mode="trainval" if mode == "train" else "test",
                transform=self.data[mode]["preprocess"],
            )

            self.data[mode]["loader"] = torch.utils.data.DataLoader(
                self.data[mode]["dataset"],
                batch_size=batch,
                collate_fn=collate_fn,
                shuffle=mode == "train",
            )

        elif self.dataset_name == "IMAGENET":
            batch = self.cfg.TRAIN.CLASSIFIER.BATCH_SIZE

            self.data[mode]["dataset"] = ImageNetDataset(
                self.cfg.DATA.IMAGENET.ROOT,
                self.class_names,
                "train" if mode == "train" else "val",
                transform=self.data[mode]["preprocess"],
            )

            if mode == "train" and self.stage == NetworkStage.FINETUNE.value:
                batch //= 2

            self.data[mode]["loader"] = torch.utils.data.DataLoader(
                self.data[mode]["dataset"],
                batch_size=batch,
                shuffle=mode == "train",
            )

        elif self.dataset_name == "COCO":
            raise NotImplementedError

    def prepare_inference(self):
        training_cfg = self.get_training_cfg()

        self.set_preprocess(training_cfg.IMAGE_SIZE)

        transform = self.data["test"]["preprocess"]

        if self.network_type == NetworkType.DETECTOR.value:
            self.load_detector()
        elif self.network_type == NetworkType.CLASSIFIER.value:
            self.load_classifier(stage=NetworkStage.INFERENCE.value)

        return transform
