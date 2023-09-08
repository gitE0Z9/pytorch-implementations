import torch

# from models.yolov1.detector import Yolov1, Yolov1Resnet
# from models.yolov2.detector import Yolov2, Yolov2Resnet
from models.yolov1.loss import YoloLoss
from models.yolov2.loss import YOLOv2Loss
from configs.schema import Setting, OptimizerCfg


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
    def __init__(self, cfg: Setting, device: str):
        self.cfg = cfg
        self.device = device

    def get_loss(self):
        detector_name = self.cfg.MODEL.NAME

        if detector_name == "yolov1":
            return YoloLoss(
                self.device,
            )

        elif detector_name == "yolov2":
            return YOLOv2Loss(
                self.cfg.MODEL.NUM_ANCHORS,
                self.device,
            )


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
