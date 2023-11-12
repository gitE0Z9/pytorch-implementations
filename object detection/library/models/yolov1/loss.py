from utils.numerical import safe_sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import IOU, build_targets
from constants.schema import DetectorContext


class YOLOLoss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        lambda_coord: float = 5,
        lambda_noobject: float = 0.5,
        iou_threshold: float = 0.5,
    ):
        super(YOLOLoss, self).__init__()

        self.device = context.device
        self.num_bboxes = context.num_anchors
        self.num_classes = context.num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobject = lambda_noobject
        self.epsilon = 1e-5
        self.iou_threshold = iou_threshold

    def responsible_iou(
        self,
        prediction: torch.Tensor,
        groundtruth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # N, 2, 1, 7, 7
        ious = IOU(prediction[:, :, 0:4, :, :], groundtruth[:, :, 0:4, :, :])
        max_iou, best_box = ious.max(dim=1, keepdim=True)  # N, 1, 1, 7, 7

        # N, 2, 1, 7, 7
        best_box = torch.zeros_like(ious)
        best_box = torch.cat([best_box.eq(b).int() for b in range(self.num_bboxes)], 1)

        return max_iou, best_box

    def forward(
        self,
        prediction: torch.Tensor,
        groundtruth: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, grid_y, grid_x = prediction.shape
        prediction = prediction.unsqueeze(1)
        coord_prediction = prediction[:, :, : self.num_bboxes * 5, :, :].view(
            batch_size, self.num_bboxes, 5, grid_y, grid_x
        )
        groundtruth = build_targets(
            groundtruth,
            (batch_size, 1, 5 + self.num_classes, grid_y, grid_x),
        ).to(self.device)

        # obj indicator
        obj_here = groundtruth[:, :, 4:5, :, :]  # N, 1, 1, 7, 7

        # iou indicator
        # left only 49 predictors
        # N, 1, 1, 7, 7 | N, 2, 1, 7, 7
        ious, best_box = self.responsible_iou(coord_prediction, groundtruth)

        positives = obj_here * best_box

        # class loss / objecness loss / xywh loss
        # indicator has to be inside the loss function
        cls_loss = F.mse_loss(
            obj_here * prediction[:, :, self.num_bboxes * 5 :, :, :],
            groundtruth[:, :, 5:, :, :],
            reduction="sum",
        )

        obj_loss = F.mse_loss(
            positives * coord_prediction[:, :, 4:5, :, :],
            positives * ious,
            reduction="sum",
        )

        xy_loss = F.mse_loss(
            positives * coord_prediction[:, :, 0:2, :, :],
            positives * groundtruth[:, :, 0:2, :, :],
            reduction="sum",
        )

        # sqrt numerical issue
        wh_loss = F.mse_loss(
            positives * safe_sqrt(coord_prediction[:, :, 2:4, :, :]),
            positives * groundtruth[:, :, 2:4, :, :].sqrt(),
            reduction="sum",
        )

        # clean the other bbox block with wrong confidence
        noobj_loss = F.mse_loss(
            (1 - positives) * coord_prediction[:, :, 4:5, :, :],
            positives * 0,  # all zeros
            reduction="sum",
        )

        total_loss = (
            cls_loss
            + self.lambda_noobject * noobj_loss
            + obj_loss
            + self.lambda_coord * (xy_loss + wh_loss)
        )

        return total_loss
