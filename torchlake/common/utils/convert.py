import torch
import numpy as np


def img_tensor_to_np(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 3:
        tensor: torch.Tensor = tensor.unsqueeze(0)

    tensor = tensor.permute(0, 2, 3, 1)

    if tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    if tensor.device.type != "cpu":
        tensor = tensor.detach().cpu()

    return tensor.numpy()
