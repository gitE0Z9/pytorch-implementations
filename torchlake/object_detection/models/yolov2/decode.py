import torch
from torchlake.common.utils.numerical import generate_grid

from ...constants.schema import DetectorContext
from ...mixins.decode_yolo import YOLODecodeMixin


class Decoder(YOLODecodeMixin):
    def __init__(self, anchors: torch.Tensor, context: DetectorContext):
        self.context = context
        self.anchors = anchors

    def decode(
        self,
        feature_map: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor:
        """decode feature map to original size

        Args:
            feature_map (torch.Tensor): shape (#batch, #anchor*(5 + #class), grid_y, grid_x)
            image_size (tuple[int, int]): (image_y, image_x)

        Returns:
            torch.Tensor: shape (#batch, #anchor * #grid, 5 + #class), in format of (x,y,w,h)
        """
        batch_size, channel_size, fm_h, fm_w = feature_map.shape
        input_h, input_w = image_size
        stride_x, stride_y = input_w / fm_w, input_h / fm_h

        num_anchors = self.anchors.size(1)
        channel_size_per_anchor = channel_size // num_anchors
        num_classes = channel_size_per_anchor - 5
        feature_map = feature_map.unflatten(1, (num_anchors, channel_size_per_anchor))

        grid_x, grid_y = generate_grid(fm_w, fm_h)

        cx = (
            feature_map[:, :, 0, :, :]
            .sigmoid()
            .multiply(input_w)
            .add(grid_x * stride_x)
            .reshape(batch_size, -1, 1)
        )
        cy = (
            feature_map[:, :, 1, :, :]
            .sigmoid()
            .multiply(input_h)
            .add(grid_y * stride_y)
            .reshape(batch_size, -1, 1)
        )
        w = (
            feature_map[:, :, 2, :, :]
            .exp()
            .multiply(self.anchors[:, :, 0, :, :])
            .multiply(input_w)
            .reshape(batch_size, -1, 1)
        )
        h = (
            feature_map[:, :, 3, :, :]
            .exp()
            .multiply(self.anchors[:, :, 1, :, :])
            .multiply(input_h)
            .reshape(batch_size, -1, 1)
        )
        conf = feature_map[:, :, 4, :, :].sigmoid().reshape(batch_size, -1, 1)
        prob = (
            feature_map[:, :, 5:, :, :]
            .float()
            .softmax(2)
            .permute(0, 1, 3, 4, 2)
            .reshape(batch_size, -1, num_classes)
        )
        x = cx - w / 2
        y = cy - h / 2

        # batch_size, boxes * grid_y * grid_x, 5+C
        return torch.cat([x, y, w, h, conf, prob], 2)
