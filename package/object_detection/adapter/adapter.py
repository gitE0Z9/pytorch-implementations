import torch

# from object_detection.models.yolov1.detector import Yolov1, Yolov1Resnet
# from object_detection.models.yolov2.detector import Yolov2, Yolov2Resnet
from object_detection.models.yolov1.loss import YOLOLoss
from object_detection.models.yolov2.loss import YOLOv2Loss, YOLO9000Loss
from object_detection.models.ssd.loss import MultiboxLoss
from object_detection.configs.schema import Setting, OptimizerCfg
from object_detection.constants.schema import DetectorContext

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
            "yolov1": YOLOLoss,
            "yolov2": YOLOv2Loss,
            "yolo9000": YOLO9000Loss,
            "ssd": MultiboxLoss,
        }

        loss_class = loss_class_mapping.get(detector_name)

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
