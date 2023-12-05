import torch
import torch.nn as nn
import torch.nn.functional as F
from object_detection.constants.schema import DetectorContext
from object_detection.models.yolov2.anchor import PriorBox
from torchvision.ops import box_convert, box_iou
from object_detection.utils.train import IOU, generate_grid_train


class YOLOv2Loss(nn.Module):
    def __init__(
        self,
        context: DetectorContext,
        lambda_obj: float = 5,
        lambda_prior: float = 0.01,
        iou_threshold: float = 0.6,
    ):
        super(YOLOv2Loss, self).__init__()

        self.num_anchors = context.num_anchors
        self.num_classes = context.num_classes
        self.device = context.device
        self.lambda_obj = lambda_obj
        self.lambda_prior = lambda_prior
        self.iou_threshold = iou_threshold
        self.epsilon = 1e-5

        self.anchors = PriorBox(context.num_anchors, context.dataset).anchors
        self.anchors = self.anchors.to(context.device)

    def match(self, groundtruth: list, grids: torch.Tensor) -> torch.Tensor:
        batch_size = len(groundtruth)
        _, num_anchors, _, grid_y, grid_x = grids.shape

        target = torch.zeros((batch_size, num_anchors * grid_y * grid_x, 5)).to(
            self.device
        )

        # 1, 5, 2, 1, 1
        default_boxes = (
            torch.cat([grids, self.anchors.repeat(1, 1, 1, grid_y, grid_x)], 2)
            .reshape(1, self.num_anchors, 4, -1)
            .transpose(2, 3)
            .reshape(-1, 4)
        )

        for batch_index, gt in enumerate(groundtruth):
            gt = torch.Tensor(gt).to(self.device)

            # ?, 8732
            overlaps = box_iou(
                box_convert(gt[:, :4], "xywh", "xyxy"),
                box_convert(default_boxes, "xywh", "xyxy"),
            )

            # A, assign gt to rest of anchor
            best_gt_overlap, best_gt_idx = overlaps.max(0)
            over_threshold = best_gt_overlap > self.iou_threshold

            target[batch_index, over_threshold] = gt[best_gt_idx[over_threshold]]

            # ?, assign best anchor
            best_prior_idx = overlaps.argmax(1)

            # ?,4
            target[batch_index, best_prior_idx] = gt

        target = torch.cat(
            [
                target,
                F.one_hot(target[:, :, 4].long(), num_classes=self.num_classes),
            ],
            2,
        )

        target = (
            target.reshape(
                batch_size, self.num_anchors, grid_y * grid_x, 5 + self.num_classes
            )
            .transpose(2, 3)
            .reshape(batch_size, self.num_anchors, 5 + self.num_classes, grid_y, grid_x)
        )

        return target

    def forward(
        self,
        prediction: torch.Tensor,
        groundtruth: list,
        seen: int,
    ) -> torch.Tensor:
        """forward function of YOLOv2Loss
        Some extra rules
        positive anchors: x,y,w,h,c,p loss
        before 12800, negative anchors : use anchors as truths
        best matched, iou lower than threshold: noobject loss
        best matched, iou over threshold: no loss

        p.s. match with fixed anchor and no overlapping groundtruth(?

        Args:
            prediction (torch.Tensor): prediction
            groundtruth (list): groundtruth
            anchors (torch.Tensor): anchors
            seen (int): had seen how many images, 12800 is the threshold in the paper

        Returns:
            torch.Tensor: loss
        """

        batch_size, channel, grid_y, grid_x = prediction.shape
        prediction = prediction.reshape(
            batch_size,
            self.num_anchors,
            channel // self.num_anchors,
            grid_y,
            grid_x,
        )
        # transform
        prediction[:, :, 0:2, :, :] = prediction[:, :, 0:2, :, :].sigmoid()
        prediction[:, :, 2:4, :, :] = prediction[:, :, 2:4, :, :].exp() * self.anchors
        prediction[:, :, 4:5, :, :] = prediction[:, :, 4:5, :, :].sigmoid()

        # find iou and noobject loss indicator
        # N, 5, 4, 13, 13 ; N, [?, 4]

        # 1, 5, 2, 13, 13
        grids = (
            generate_grid_train(grid_x, grid_y, center=True)
            .repeat(1, self.num_anchors, 1, 1, 1)
            .to(self.device)
        )
        target = self.match(groundtruth, grids)

        # B, A, 1, 13, 13
        ious = IOU(prediction[:, :, :4, :, :], target[:, :, :4, :, :])

        # no object loss for lower than threshold
        noobject_indicator = ious.le(self.iou_threshold)
        noobj_loss = F.mse_loss(
            noobject_indicator * prediction[:, :, 4:5, :, :],
            0 * target[:, :, 4:5, :, :],
            reduction="sum",
        )

        # before 12800 iter, prior as truth
        positive_indicator = target[:, :, 4:5, :, :] > 0
        prior_loss = 0
        if seen < 12800:
            # N, 5, 4, 13, 13
            box_pred = ~positive_indicator * prediction[:, :, 0:4, :, :]
            anchors_truth = torch.zeros_like(box_pred).to(self.device)
            anchors_truth[:, :, 0:1, :, :] = 0.5 / grid_x
            anchors_truth[:, :, 1:2, :, :] = 0.5 / grid_y
            anchors_truth[:, :, 2:4, :, :] = self.anchors.tile(
                batch_size, 1, 1, grid_y, grid_x
            )
            anchors_truth = ~positive_indicator * anchors_truth
            prior_loss = F.mse_loss(box_pred, anchors_truth, reduction="sum")

        # high iou predictors
        positive = prediction * positive_indicator  # N, 5, 25, 13, 13
        x_pred = positive[:, :, 0:1, :, :]
        y_pred = positive[:, :, 1:2, :, :]
        wh_pred = positive[:, :, 2:4, :, :]
        obj_pred = positive[:, :, 4:5, :, :]
        cls_pred = positive[:, :, 5:, :, :].softmax(2)

        # class loss / objecness loss / xywh loss
        x_loss = F.mse_loss(x_pred, target[:, :, 0:1, :, :], reduction="sum")
        y_loss = F.mse_loss(y_pred, target[:, :, 1:2, :, :], reduction="sum")
        wh_loss = F.mse_loss(wh_pred, target[:, :, 2:4, :, :], reduction="sum")
        box_loss = x_loss + y_loss + wh_loss
        obj_loss = F.mse_loss(
            obj_pred,
            ious * positive_indicator,
            reduction="sum",
        )
        cls_loss = F.mse_loss(cls_pred, target[:, :, 5:, :, :], reduction="sum")

        total_loss = (
            cls_loss
            + 1 * noobj_loss
            + self.lambda_obj * obj_loss
            + 1 * box_loss
            + self.lambda_prior * prior_loss
        )

        return total_loss


class YOLO9000Loss(nn.Module):
    ...
