import torch
import numpy as np


def img_tensor_to_np(
    tensor: torch.Tensor,
    scale: float = 1,
    output_type=np.float32,
) -> np.ndarray:
    if tensor.dim() == 3:
        tensor: torch.Tensor = tensor.unsqueeze(0)

    tensor = tensor.permute(0, 2, 3, 1)

    if tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()

    if scale != 1:
        tensor *= scale

    arr = tensor.numpy().astype(output_type)

    return np.ascontiguousarray(arr)
