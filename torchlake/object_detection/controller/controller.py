from pathlib import Path
import platform
from importlib import import_module

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

from ..configs.schema import ClassifierTrainingCfg, DetectorTrainingCfg, Setting
from ..constants.enums import NetworkStage, NetworkType, OperationMode
from ..constants.schema import DetectorContext
from ..datasets.coco.datasets import COCODatasetFromCSV
from ..datasets.imagenet.datasets import ImageNetDataset
from ..datasets.schema import DatasetCfg
from ..datasets.voc.datasets import VOCDatasetFromCSV
from ..models.yolov1.decode import yolo_postprocess
from ..utils.config import load_classes, load_config
from ..utils.plot import rand_color
from ..utils.train import collate_fn


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
        self.class_names = load_classes(
            Path(__file__).parent.parent.absolute().joinpath(dataset_cfg.CLASSES_PATH)
        )

        self.model = None
        self.data = {
            OperationMode.TRAIN.value: {
                "preprocess": None,
                "dataset": None,
                "loader": None,
            },
            OperationMode.TEST.value: {
                "preprocess": None,
                "dataset": None,
                "loader": None,
            },
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
        return DatasetCfg(
            **load_config(
                Path(__file__).parent.parent.joinpath(
                    f"datasets/{self.dataset_name.lower()}/config.yml"
                )
            )
        )

    def get_num_workers(self) -> int:
        if platform.system() == "Windows":
            return 0
        else:
            return self.cfg.HARDWARE.NUM_WORKERS

    def get_classifier_class(self) -> nn.Module:
        backbone = self.cfg.MODEL.BACKBONE

        if (
            backbone.startswith("resnet")
            or backbone.startswith("vgg")
            or backbone.startswith("mobile")
        ):
            model_class = getattr(torchvision.models, backbone)
        else:
            classifier_module = import_module(
                f"torchlake.object_detection.models.{self.cfg.MODEL.NAME}.network"
            )
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

        detector_module = import_module(
            f"torchlake.object_detection.models.{self.cfg.MODEL.NAME}.detector"
        )
        model_class = getattr(detector_module, class_name)

        return model_class

    def get_detector_context(self) -> DetectorContext:
        return DetectorContext(
            detector_name=self.cfg.MODEL.NAME,
            dataset=self.dataset_name.lower(),
            device=self.device,
            num_classes=self.get_dataset_cfg().NUM_CLASSES,
            num_anchors=self.cfg.MODEL.NUM_ANCHORS,
            anchors_path=self.cfg.MODEL.ANCHORS_PATH,
        )

    def set_preprocess(self):
        raise NotImplementedError

    def load_classifier(self, stage: str):
        """Load classifier with corresponding network stage."""
        model_cfg = self.cfg.MODEL
        classifier_path = model_cfg.CLASSIFIER_PATH
        model_class = self.get_classifier_class()

        kwargs = {}
        if model_cfg.BACKBONE.startswith("resnet"):
            kwargs["weights"] = True
        self.model: nn.Module = model_class(**kwargs)

        # load trained weight, imagenet weight
        if stage == NetworkStage.INFERENCE.value and classifier_path:
            self.load_weight(classifier_path)
        elif stage == NetworkStage.FINETUNE.value:
            self.load_weight(classifier_path)
        else:
            print("random classifier weight from scratch")

        self.model = self.model.to(self.device)

    def load_detector(self):
        """Load detector"""
        model_cfg = self.cfg.MODEL
        backbone = model_cfg.BACKBONE
        detector_path = model_cfg.DETECTOR_PATH

        dataset_cfg = self.get_dataset_cfg()
        number_of_anchors = model_cfg.NUM_ANCHORS
        number_of_classes = dataset_cfg.NUM_CLASSES

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

        elif backbone.startswith("vgg"):
            self.model = detector_class(
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

    def load_weight(self, weight_path: str | Path):
        """Load weight"""
        assert weight_path and isinstance(
            weight_path, str | Path
        ), "Please provide weight path."

        if isinstance(weight_path, Path):
            weight_path = weight_path.as_posix()

        self.model.load_state_dict(torch.load(weight_path))

    def load_dataset(self, mode: str):
        """
        Load dataset

        args: mode, can be train or test
              dataset, can be voc, imagenet, coco

        """
        num_workers = self.get_num_workers()

        dataset_mapping = {
            "VOC": VOCDatasetFromCSV,
            "IMAGENET": ImageNetDataset,
            "COCO": COCODatasetFromCSV,
        }

        dataset_class = dataset_mapping.get(self.dataset_name)
        self.data[mode]["dataset"] = dataset_class(
            mode=mode,
            transform=self.data[mode]["preprocess"],
        )

        if self.dataset_name in ["VOC", "COCO"]:
            self.data[mode]["loader"] = DataLoader(
                self.data[mode]["dataset"],
                batch_size=self.cfg.TRAIN.DETECTOR.BATCH_SIZE,
                collate_fn=collate_fn,
                shuffle=mode == OperationMode.TRAIN.value,
                # drop_last=True,
                num_workers=num_workers,
                pin_memory=True,
            )
        elif self.dataset_name in ["IMAGENET"]:
            batch = self.cfg.TRAIN.CLASSIFIER.BATCH_SIZE
            if (
                mode == OperationMode.TRAIN.value
                and self.stage == NetworkStage.FINETUNE.value
            ):
                batch //= 2

            self.data[mode]["loader"] = DataLoader(
                self.data[mode]["dataset"],
                batch_size=batch,
                shuffle=mode == OperationMode.TRAIN.value,
                num_workers=num_workers,
                pin_memory=True,
            )
        else:
            raise NotImplementedError

    def load_decoder(self, context: DetectorContext):
        MODEL_NAME = self.cfg.MODEL.NAME

        module = import_module(f"torchlake.object_detection.models.{MODEL_NAME}.decode")

        self.decoder = getattr(module, "Decoder")(context)

    def prepare_inference(self):
        training_cfg = self.get_training_cfg()

        # TODO: implement in each child class
        self.set_preprocess(training_cfg.IMAGE_SIZE)

        transform = self.data[OperationMode.TEST.value]["preprocess"]

        if self.network_type == NetworkType.DETECTOR.value:
            self.load_detector()
            self.load_decoder(self.get_detector_context())
        elif self.network_type == NetworkType.CLASSIFIER.value:
            self.load_classifier(stage=NetworkStage.INFERENCE.value)

        return transform

    def postprocess(
        self,
        output: torch.Tensor,
        img_size: tuple[int, int],
    ) -> list[torch.Tensor]:
        dataset_cfg = self.get_dataset_cfg()

        MODEL_NAME = self.cfg.MODEL.NAME
        NUM_CLASSES = dataset_cfg.NUM_CLASSES

        decoded = self.decoder.decode(output, img_size)

        if "yolo" in MODEL_NAME:
            detected = yolo_postprocess(decoded, NUM_CLASSES, self.cfg.INFERENCE)

        return detected
