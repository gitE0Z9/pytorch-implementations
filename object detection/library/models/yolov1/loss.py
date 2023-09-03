import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import IOU


class YoloLoss(nn.Module):
    def __init__(
        self,
        lambda_coord: float = 5,
        lambda_noobject: float = 0.5,
        num_boxes: int = 2,
    ):
        super(YoloLoss, self).__init__()

        self.num_bbox = num_boxes
        self.lambda_coord = lambda_coord
        self.lambda_noobject = lambda_noobject
        self.epsilon = 1e-5

    def responsible_iou(
        self,
        prediction: torch.Tensor,
        groundtruth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ious = [
            IOU(
                prediction[:, (1 + 5 * b) : (5 + 5 * b), :, :],
                groundtruth[:, 1:5, :, :],
            )
            for b in range(self.num_bbox)
        ]
        ious = torch.cat(ious, dim=1)  # N, boxes, 7, 7
        max_iou, best_box = ious.max(dim=1, keepdim=True)  # N,1,7,7
        return max_iou, best_box

    def forward(
        self,
        prediction: torch.Tensor,
        groundtruth: torch.Tensor,
    ) -> torch.Tensor:
        groundtruth = groundtruth.float()

        # obj indicator
        obj_here = groundtruth[:, 0:1, :, :]  # [N,1,7,7]

        # iou indicator
        ious, best_box = self.responsible_iou(prediction, groundtruth)  # N, 1, 7, 7

        # left only 49 predictors
        box_pred = torch.zeros_like(prediction[:, :5, :, :])  # N, 5, 7, 7
        for b in range(self.num_bbox):
            box_pred += best_box.eq(b) * prediction[:, b * 5 : (b + 1) * 5, :, :]
        wh_pred = (
            box_pred[:, 3:5, :, :].sign()
            * (box_pred[:, 3:5, :, :].abs() + self.epsilon).sqrt()
        )  # sqrt the value then plus sign back

        cls_pred = prediction[:, self.num_bbox * 5 :, :, :]

        # class loss / objecness loss / xywh loss
        # indicator has to be inside the loss function
        cls_loss = F.mse_loss(
            obj_here * cls_pred, groundtruth[:, 5:, :, :], reduction="sum"
        )
        obj_loss = F.mse_loss(
            obj_here * box_pred[:, 0:1, :, :], obj_here, reduction="sum"
        )
        xy_loss = F.mse_loss(
            obj_here * box_pred[:, 1:3, :, :],
            groundtruth[:, 1:3, :, :],
            reduction="sum",
        )
        wh_loss = F.mse_loss(
            obj_here * wh_pred, groundtruth[:, 3:5, :, :].sqrt(), reduction="sum"
        )

        # not mention in the original papaer, but in aladdin and clean the other bbox block with wrong confidence
        noobj_loss = 0.0
        for b in range(self.num_bbox):
            noobj_loss += F.mse_loss(
                (1 - obj_here) * prediction[:, 0 + 5 * b : 1 + 5 * b, :, :],
                obj_here * 0,
                reduction="sum",
            )  # weird part

        total_loss = (
            cls_loss
            + self.lambda_noobject * noobj_loss
            + obj_loss
            + self.lambda_coord * (xy_loss + wh_loss)
        )

        return total_loss
