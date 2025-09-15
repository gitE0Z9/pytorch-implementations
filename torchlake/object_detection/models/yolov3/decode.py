from typing import Literal, Sequence

import torch

from torchlake.common.utils.numerical import generate_grid

from ...constants.schema import DetectorContext
from ...mixins.decode_yolo import YOLODecodeMixin


class Decoder(YOLODecodeMixin):
    def __init__(
        self,
        anchors: torch.Tensor,
        context: DetectorContext,
        cls_loss_type: Literal["softmax", "sigmoid"] = "sigmoid",
    ):
        self.context = context
        self.anchors = anchors
        self.cls_loss_type = cls_loss_type

    def decode(
        self,
        feature_maps: Sequence[torch.Tensor],
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """decode feature map to original size

        Args:
            feature_maps (Sequence[torch.Tensor]): length is number of scales, shape is (#batch, #anchor*(5 + #class), grid_y, grid_x)
            image_size (tuple[int, int]): (image_y, image_x)

        Returns:
            torch.Tensor: shape (#batch, #box = sum of #anchor * #grid, 5 + #class), in format of (x,y,w,h)
        """
        input_h, input_w = image_size
        decoded_all = []
        anchor_offset = 0
        for feature_map, num_anchor in zip(feature_maps, self.context.num_anchors):
            batch_size, channel_size, fm_h, fm_w = feature_map.shape
            stride_x, stride_y = input_w / fm_w, input_h / fm_h

            channel_size_per_anchor = channel_size // num_anchor
            num_classes = channel_size_per_anchor - 5
            feature_map = feature_map.unflatten(
                1, (num_anchor, channel_size_per_anchor)
            )

            grid_x, grid_y = generate_grid(fm_w, fm_h)

            anchors = self.anchors[:, anchor_offset : anchor_offset + num_anchor]

            cx = (
                feature_map[:, :, 0, :, :]
                .sigmoid()
                .multiply(input_w)
                .add(grid_x * stride_x)
                .view(batch_size, -1, 1)
            )
            cy = (
                feature_map[:, :, 1, :, :]
                .sigmoid()
                .multiply(input_h)
                .add(grid_y * stride_y)
                .view(batch_size, -1, 1)
            )
            w = (
                feature_map[:, :, 2, :, :]
                .exp()
                .multiply(anchors[:, :, 0, :, :])
                .multiply(input_w)
                .view(batch_size, -1, 1)
            )
            h = (
                feature_map[:, :, 3, :, :]
                .exp()
                .multiply(anchors[:, :, 1, :, :])
                .multiply(input_h)
                .view(batch_size, -1, 1)
            )
            conf = feature_map[:, :, 4, :, :].sigmoid().view(batch_size, -1, 1)
            prob = (
                feature_map[:, :, 5:, :, :]
                .float()
                .permute(0, 1, 3, 4, 2)
                .contiguous()
                .view(batch_size, -1, num_classes)
            )
            if self.cls_loss_type == "softmax":
                prob = prob.softmax(-1)
            else:
                prob = prob.sigmoid()
            x = cx - w / 2
            y = cy - h / 2

            # batch_size, boxes * grid_y * grid_x, 5+C
            decoded_all.append(torch.cat([x, y, w, h, conf, prob], 2))

            anchor_offset += num_anchor

        return torch.cat(decoded_all, 1)
