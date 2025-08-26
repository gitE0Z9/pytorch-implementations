import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlake.common.utils.numerical import safe_sqrt

from ...constants.schema import DetectorContext
from ...utils.train import IOU, build_grid_targets


class YOLOLoss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        lambda_coord: float = 5,
        lambda_noobject: float = 0.5,
        iou_threshold: float = 0.5,
        return_all_loss: bool = False,
    ):
        super().__init__()

        self.device = context.device
        self.num_bboxes = context.num_anchors
        self.num_classes = context.num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobject = lambda_noobject
        self.epsilon = 1e-5
        self.iou_threshold = iou_threshold
        self.return_all_loss = return_all_loss

    def match(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """find responsible bbox in each cell

        Args:
            pred (torch.Tensor): coord & class logit, shape is (batch size, num anchor, 4+1+num class, h, w)
            gt (torch.Tensor): groundtruth, shape is (batch size, 1, 4+1+num class, h, w)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: matched ious, responsible bbox, shape is (batch size, num anchor, 1, h, w)
        """
        # N, 2, 1, 7, 7
        ious = IOU(pred[:, :, 0:4, :, :], gt[:, :, 0:4, :, :])

        # N, 1, 1, 7, 7
        best_box = ious.argmax(dim=1, keepdim=True)

        # N, 2, 1, 7, 7
        best_box = torch.ones_like(ious).scatter_(1, best_box, 0)

        return ious, best_box

    def forward(self, pred: torch.Tensor, gt: list[list[list[int]]]) -> torch.Tensor:
        """compute classification loss, coord loss, obj loss, nonobject loss

        Args:
            pred (torch.Tensor): coord & class logit, shape is (batch size, num bbox*(4+1) + num class, h, w)
            gt (list[list[list[int]]]): bacth of groundtruth, a row is like [cx, cy, w, h, c]

        Returns:
            torch.Tensor: loss
        """
        batch_size, _, grid_y, grid_x = pred.shape
        # N, 2, 5, 7, 7
        coord_pred = pred[:, : self.num_bboxes * 5, :, :].unflatten(
            1, (self.num_bboxes, 5)
        )
        # N, 1, 2*5+C, 7, 7
        pred = pred.unsqueeze(1)
        # N, 1, 5+C, 7, 7
        gt = build_grid_targets(
            gt,
            (batch_size, 1, 5 + self.num_classes, grid_y, grid_x),
        ).to(self.device)

        # iou indicator
        # left only 49 predictors
        # N, 2, 1, 7, 7 | N, 2, 1, 7, 7
        with torch.no_grad():
            ious, best_box = self.match(coord_pred, gt)

        # obj indicator
        # N, 1, 1, 7, 7
        obj_here = gt[:, :, 4:5, :, :]

        # N, 2, 1, 7, 7
        positives = best_box * obj_here

        # class loss / objecness loss / xywh loss
        # indicator has to be inside the loss function
        cls_loss = F.mse_loss(
            obj_here * pred[:, :, self.num_bboxes * 5 :, :, :],
            obj_here * gt[:, :, 5:, :, :],
            reduction="sum",
        )

        obj_loss = F.mse_loss(
            positives * coord_pred[:, :, 4:5, :, :],
            positives * ious,
            reduction="sum",
        )

        # clean the other bbox block with wrong confidence
        noobj_loss = F.mse_loss(
            (1 - positives) * coord_pred[:, :, 4:5, :, :],
            torch.zeros_like(positives),  # all zeros
            reduction="sum",
        )

        xy_loss = F.mse_loss(
            positives * coord_pred[:, :, 0:2, :, :],
            positives * gt[:, :, 0:2, :, :],
            reduction="sum",
        )

        # sqrt numerical issue
        wh = coord_pred[:, :, 2:4, :, :]
        wh_loss = F.mse_loss(
            positives * wh.sign() * safe_sqrt(wh),
            positives * safe_sqrt(gt[:, :, 2:4, :, :]),
            reduction="sum",
        )

        # https://github.com/pjreddie/darknet/blob/f6afaabcdf85f77e7aff2ec55c020c0e297c77f9/src/detection_layer.c#L179
        # iou_loss = F.mse_loss(
        #     positives * ious,
        #     positives,
        #     reduction="mean",
        # )

        total_loss = (
            # iou_loss
            cls_loss
            + self.lambda_noobject * noobj_loss
            + obj_loss
            + self.lambda_coord * (xy_loss + wh_loss)
        )

        if self.return_all_loss:
            return total_loss, cls_loss, noobj_loss, obj_loss, xy_loss, wh_loss
        else:
            return total_loss
