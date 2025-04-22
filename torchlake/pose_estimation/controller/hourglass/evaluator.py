import numpy as np
import torch
import torch.nn.functional as F
from torchlake.common.controller.evaluator import ClassificationEvaluator


class PoseEstimationEvaluator(ClassificationEvaluator):
    def _decode_output(
        self,
        output: torch.Tensor | tuple[torch.Tensor],
        *args,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor]:
        x, _ = kwargs.pop("row")
        # B, A, C, H, W
        B, _, C, H, W = output.shape

        # take final stack
        output = output[:, -1].float()
        output = F.interpolate(output, size=x.shape[-2:]).cpu().numpy()
        keypoints = np.column_stack(
            np.unravel_index(
                output.view(B * C, -1).argmax(-1),
                output.shape[-2:],
            )
        )
        keypoints = torch.Tensor(keypoints).view(B, C, 2)

        return keypoints
