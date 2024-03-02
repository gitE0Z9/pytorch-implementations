from importlib import import_module

import torch

from ..configs.schema import OptimizerCfg, Setting
from ..constants.schema import DetectorContext

# class DetectorAdapter:
#     def __init__(self, cfg: Setting, device: str):
#         self.cfg = cfg
#         self.device = device

#     def get_detector(
#         self,
#     ):
#         detector_name = self.cfg.MODEL.NAME

#         if detector_name == "yolo":
#             return YoloLoss()

#         elif detector_name == "yolov2":
#             return YOLOv2Loss(
#                 self.cfg.MODEL.NUM_ANCHORS,
#                 self.device,
#             )


class DetectorLossAdapter:
    def __init__(self, context: DetectorContext):
        self.context = context

    def get_loss(self):
        detector_name = self.context.detector_name

        loss_class_mapping = {
            "yolov1": "yolov1.loss.YOLOLoss",
            "yolov2": "yolov2.loss.YOLOv2Loss",
            "yolo9000": "yolov2.loss.YOLO9000Loss",
            "ssd": "ssd.loss.MultiboxLoss",
        }

        loss_class_path = loss_class_mapping.get(detector_name)
        module_name, loss_class_name = loss_class_path.rsplit(".", 1)
        loss_class = getattr(
            import_module("torchlake.object_detection.models." + module_name),
            loss_class_name,
        )

        return loss_class(self.context)


class OptimizerAdapter:
    def __init__(self, cfg: OptimizerCfg):
        self.cfg = cfg

    def get_optimizer(self, parameter):
        optimizer_name = self.cfg.TYPE

        if optimizer_name == "adam":
            return torch.optim.Adam(
                parameter,
                lr=self.cfg.LR,
                weight_decay=self.cfg.DECAY,
            )
        elif optimizer_name == "sgd":
            return torch.optim.SGD(
                parameter,
                lr=self.cfg.LR,
                weight_decay=self.cfg.DECAY,
                momentum=self.cfg.MOMENTUM,
            )
        else:
            raise NotImplementedError
