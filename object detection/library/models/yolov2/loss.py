from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train import IOU, xywh_to_xyxy
from torchvision.ops import box_iou, box_convert


class YOLOv2Loss(nn.Module):
    def __init__(
        self,
        num_anchors: int,
        device: str,
        lambda_obj: float = 5,
        iou_threshold: float = 0.6,
    ):
        super(YOLOv2Loss, self).__init__()

        self.num_anchors = num_anchors
        self.device = device
        self.lambda_obj = lambda_obj
        self.lambda_prior = 0.01
        self.iou_threshold = iou_threshold
        self.epsilon = 1e-5

    def iou_box(
        self,
        prediction: torch.Tensor,
        groundtruth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, na, _, _, _ = prediction.shape

        ious = IOU(prediction, groundtruth)  # N,5,1,13,13

        best_ious, best_box = ious.max(1, keepdim=True)  # N,1,1,13,13
        best_box = torch.cat(
            [best_box.eq(a).int() for a in range(na)], 1
        )  # indicator N,5,1,13,13
        # best_ious = best_ious * best_box # N, 5, 1, 13, 13

        # best_box = best_ious.gt(self.iou_thresh).int()
        # best_ious = best_ious * best_box
        return best_ious, best_box

    def match_anchor(
        self,
        anchors: torch.Tensor,
        groundtruth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ious = IOU(anchors, groundtruth)  # N,5,1,13,13 # TODO: cant use
        _, best_anchors = ious.max(1, keepdim=True)  # N,1,1,13,13
        best_anchors = torch.cat(
            [best_anchors.eq(b).int() for b in range(self.num_anchors)], 1
        )
        return best_anchors

    # def forward(self, prediction: torch.Tensor, groundtruth: list, anchors: torch.Tensor, seen: int) -> torch.Tensor:
    #     """"
    #     positive anchors: x,y,w,h,c,p loss
    #     before 12800, negative anchors : uses anchors as truths
    #     best predictd iou lower than threshold: noobject loss
    #     best predicted iou over threshold: no loss

    #     not match to anchor
    #     """
    #     b, c, grid_y, grid_x = prediction.shape
    #     prediction = prediction.reshape(b, self.num_anchors, c//self.num_anchors, grid_y, grid_x)
    #     groundtruth = build_targets(groundtruth, (b, 1, c//self.num_anchors, grid_y, grid_x)) # N, 1, 25, 13, 13
    #     # groundtruth[:,:,2:4,:,:] = groundtruth[:,:,2:4,:,:] * grid_x
    #     groundtruth = groundtruth.tile(1, self.num_anchors, 1, 1, 1)
    #     groundtruth = groundtruth.to('cuda:0') if prediction.is_cuda else groundtruth.to('cpu')

    #     # transform
    #     prediction[:, :, 0:2, :, :] = prediction[:, :, 0:2, :, :].sigmoid()
    #     prediction[:, :, 2:4, :, :] = prediction[:, :, 2:4, :, :].exp() * anchors
    #     prediction[:, :, 4:5, :, :] = prediction[:, :, 4:5, :, :].sigmoid()

    #     # iou indicator
    #     ious, good_box = self.iou_box(prediction[:,:,:4,:,:], groundtruth[:,:,:4,:,:]) # N, 5, 1, 13, 13 # TODO: bias match

    #     # obj indicator
    #     obj_here = groundtruth[:,:,4:5,:,:] # N,1,1,13,13
    #     positive_index = good_box * obj_here # N, 5, 1, 13, 13

    #     # poitive anchor TODO
    #     # negative anchor TODO
    #     # max iou > thresh TODO

    #     # no object loss for lower than threshold
    #     noobj_pred = (1 - positive_index) * prediction[:,:,4:5,:,:]
    #     noobj_loss = F.mse_loss(noobj_pred, noobj_pred*0, reduction="sum") # TODO: loss conflict with positive

    #     # before 12800 iter, prior as truth
    #     prior_loss = 0
    #     if seen < 12800:
    #         box_pred = (1- positive_index) * prediction[:,:,0:4,:,:] # N, 5, 4, 13, 13
    #         d = 'cuda:0' if box_pred.is_cuda else 'cpu'
    #         anchors_truth = torch.zeros_like(box_pred).to(d)
    #         anchors_truth[:,:,0:1,:,:] = 0.5 / grid_x
    #         anchors_truth[:,:,1:2,:,:] = 0.5 / grid_y
    #         anchors_truth[:,:,2:4,:,:] = anchors.tile(b, 1, 1, grid_y, grid_x)
    #         prior_loss = F.mse_loss(box_pred, (1 - positive_index) * anchors_truth, reduction="sum") # TODO: loss conflict with positive

    #     # high iou predictors
    #     positive = prediction * positive_index # N, 5, 25, 13, 13
    #     x_pred = positive[:,:,0:1,:,:]
    #     y_pred = positive[:,:,1:2,:,:]
    #     wh_pred = positive[:,:,2:4,:,:]
    #     obj_pred = positive[:,:,4:5,:,:]
    #     cls_pred = positive[:,:,5:,:,:].softmax(2)

    #     # class loss / objecness loss / xywh loss
    #     x_loss = F.mse_loss(x_pred, positive_index * groundtruth[:,:,0:1,:,:], reduction="sum")
    #     y_loss = F.mse_loss(y_pred, positive_index * groundtruth[:,:,1:2,:,:], reduction="sum")
    #     wh_loss = F.mse_loss(wh_pred, positive_index * groundtruth[:,:,2:4,:,:], reduction="sum")
    #     box_loss = x_loss + y_loss + wh_loss
    #     obj_loss = F.mse_loss(obj_pred, positive_index * ious, reduction="sum")
    #     cls_loss = F.mse_loss(cls_pred, positive_index * groundtruth[:,:,5:,:,:], reduction="sum") # TODO: softmax testing

    #     total_loss = self.lambda_obj * cls_loss + 1 * noobj_loss + self.lambda_obj * obj_loss + self.lambda_obj * box_loss + self.lambda_prior * prior_loss

    #     return total_loss

    def forward(
        self,
        prediction: torch.Tensor,
        groundtruth: list,
        anchors: torch.Tensor,
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
        groundtruth = [torch.Tensor(gt).to(self.device) for gt in groundtruth]

        # transform
        prediction[:, :, 0:2, :, :] = prediction[:, :, 0:2, :, :].sigmoid()
        prediction[:, :, 2:4, :, :] = prediction[:, :, 2:4, :, :].exp() * anchors
        prediction[:, :, 4:5, :, :] = prediction[:, :, 4:5, :, :].sigmoid()

        # find iou and noobject loss indicator
        # N, 5, 4, 13, 13 ; N, [?, 4]

        with torch.no_grad():
            noobject_indicator = torch.zeros(
                batch_size,
                self.num_anchors,
                1,
                grid_y,
                grid_x,
            ).to(self.device)
            positive_indicator = torch.zeros(
                batch_size,
                self.num_anchors,
                1,
                grid_y,
                grid_x,
            ).to(self.device)
            target = torch.zeros(
                batch_size,
                self.num_anchors,
                channel // self.num_anchors,
                grid_y,
                grid_x,
            ).to(self.device)

            recover_prediction_loc = xywh_to_xyxy(prediction[:, :, :4, :, :])
            for batch in range(batch_size):
                cur_prediction_loc = recover_prediction_loc[batch]
                cur_prediction_loc = (
                    cur_prediction_loc.transpose(0, 1).reshape(4, -1).transpose(0, 1)
                )
                cur_gt = groundtruth[batch]
                if len(cur_gt) == 0:
                    continue
                cur_gt = cur_gt.view(-1, 6)
                cur_gt_loc = cur_gt[:, :4]  # ?, 4

                cur_gt_loc = box_convert(cur_gt_loc, "cxcywh", "xyxy")
                ious = box_iou(cur_prediction_loc, cur_gt_loc)
                iou_indicator = (
                    ious.max(1)[0]
                    .le(self.iou_threshold)
                    .view(self.num_anchors, 1, grid_y, grid_x)
                )
                noobject_indicator[
                    batch
                ] = iou_indicator  # take care if loss conflict with positive

                # build target and bias_match
                for i, gt in enumerate(cur_gt):
                    x_ind, y_ind, anchor_ind, cls_ind = (
                        int(gt[0] * grid_x),
                        int(gt[1] * grid_y),
                        int(gt[5]),
                        int(gt[4]),
                    )
                    positive_indicator[batch, anchor_ind, 0, y_ind, x_ind] = 1
                    noobject_indicator[batch, anchor_ind, 0, y_ind, x_ind] = 0
                    target[batch, anchor_ind, 0, y_ind, x_ind] = gt[0] % (1 / grid_x)
                    target[batch, anchor_ind, 1, y_ind, x_ind] = gt[1] % (1 / grid_y)
                    target[batch, anchor_ind, 2, y_ind, x_ind] = gt[2]
                    target[batch, anchor_ind, 3, y_ind, x_ind] = gt[3]
                    target[batch, anchor_ind, 4, y_ind, x_ind] = ious[
                        x_ind + y_ind * grid_x + anchor_ind * grid_y * grid_x, i
                    ]  # only useful, others are waste computation
                    target[batch, anchor_ind, 5 + cls_ind, y_ind, x_ind] = 1

        # no object loss for lower than threshold
        noobj_loss = F.mse_loss(
            noobject_indicator * prediction[:, :, 4:5, :, :],
            0 * prediction[:, :, 4:5, :, :],
            reduction="sum",
        )

        # before 12800 iter, prior as truth
        prior_loss = 0
        if seen < 12800:
            box_pred = (1 - positive_indicator) * prediction[
                :, :, 0:4, :, :
            ]  # N, 5, 4, 13, 13
            anchors_truth = torch.zeros_like(box_pred).to(self.device)
            anchors_truth[:, :, 0:1, :, :] = 0.5 / grid_x
            anchors_truth[:, :, 1:2, :, :] = 0.5 / grid_y
            anchors_truth[:, :, 2:4, :, :] = anchors.tile(
                batch_size, 1, 1, grid_y, grid_x
            )
            anchors_truth = (1 - positive_indicator) * anchors_truth
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
        obj_loss = F.mse_loss(obj_pred, target[:, :, 4:5, :, :], reduction="sum")
        cls_loss = F.mse_loss(cls_pred, target[:, :, 5:, :, :], reduction="sum")

        total_loss = (
            cls_loss
            + 1 * noobj_loss
            + self.lambda_obj * obj_loss
            + 1 * box_loss
            + self.lambda_prior * prior_loss
        )

        return total_loss


class Yolo9000Loss(nn.Module):
    ...
