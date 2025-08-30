import torch
from torchlake.common.utils.numerical import generate_grid
from torchlake.object_detection.constants.schema import DetectorContext

from ...mixins.decode_yolo import YOLODecodeMixin


class Decoder(YOLODecodeMixin):
    def __init__(self, context: DetectorContext):
        self.context = context

    def decode(
        self,
        pred: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """decode feature map to original size

        Args:
            pred (torch.Tensor): shape (#batch, 5 + #class, grid_y, grid_x)
            image_size (tuple[int, int]): (image_y, image_x)

        Returns:
            torch.Tensor: shape (#batch, #anchor * #grid, 5 + #class), in format of (x,y,w,h)
        """
        num_anchors = self.context.num_anchors
        num_classes = self.context.num_classes
        batch_size, _, fm_h, fm_w = pred.shape

        # batch_size, boxes, 5, grid_y, grid_x
        feature_map_coord = pred[:, : 5 * num_anchors, :, :].unflatten(
            1, (num_anchors, 5)
        )

        input_h, input_w = image_size
        stride_x, stride_y = input_w / fm_w, input_h / fm_h
        grid_x, grid_y = generate_grid(fm_w, fm_h)

        # batch_size, 1, boxes * grid_y * grid_x
        cx = (
            feature_map_coord[:, :, 0, :, :]
            .multiply(input_w)
            .add(grid_x * stride_x)
            .view(batch_size, 1, -1)
        )
        cy = (
            feature_map_coord[:, :, 1, :, :]
            .multiply(input_h)
            .add(grid_y * stride_y)
            .view(batch_size, 1, -1)
        )
        w = feature_map_coord[:, :, 2, :, :].multiply(input_w).view(batch_size, 1, -1)
        h = feature_map_coord[:, :, 3, :, :].multiply(input_h).view(batch_size, 1, -1)
        x = cx - w / 2
        y = cy - h / 2

        conf = feature_map_coord[:, :, 4, :, :].contiguous().view(batch_size, 1, -1)
        prob = (
            pred[:, 5 * num_anchors :, :, :]
            .unsqueeze(2)
            .repeat(1, 1, num_anchors, 1, 1)
            .view(batch_size, num_classes, -1)
        )

        # batch_size, boxes * grid_y * grid_x, 5+C
        return torch.cat([x, y, w, h, conf, prob], 1).transpose(1, 2)
