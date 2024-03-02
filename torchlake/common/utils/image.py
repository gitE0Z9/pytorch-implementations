from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


@lru_cache
def load_image(
    path: str | Path,
    is_numpy: bool = False,
    is_tensor: bool = False,
) -> np.ndarray | torch.Tensor | Image.Image:
    """load image to numpy array, torch tensor, PIL image

    Args:
        path (str | Path): path to image
        is_numpy (bool, optional): return numpy array. Defaults to False.
        is_tensor (bool, optional): return torch tensor. Defaults to False.

    Returns:
        np.ndarray | torch.Tensor | Image.Image: returned type
    """
    if isinstance(path, str):
        path = Path(path)
    assert path.exists(), f"Image {path} does't exist."
    path = path.as_posix()

    # numpy array
    if is_numpy:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    image = Image.open(path)
    # PIL image
    if not is_tensor:
        return image
    # pytorch tensor
    else:
        return transforms.ToTensor()(image)


def save_img_array(array: np.ndarray, file_path: str):
    if array.dtype == np.float32:
        array = (array.clip(0, 1) * 255).clip(0, 255).astype(np.uint8)
    elif array.dtype == np.uint8:
        array = array.clip(0, 255)
    else:
        raise NotImplementedError

    cv2.imwrite(file_path, array)


def decode_segmap(image: np.ndarray, colors: np.ndarray) -> np.ndarray:
    r: np.ndarray = np.zeros_like(image).astype(np.uint8)
    g = r.copy()
    b = r.copy()

    for class_idx in range(len(colors)):
        idx = image == class_idx
        r[idx] = colors[class_idx, 0]
        g[idx] = colors[class_idx, 1]
        b[idx] = colors[class_idx, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def yiq_transform(x: torch.Tensor) -> torch.Tensor:
    A = torch.Tensor(
        [
            [
                [0.299, 0.587, 0.114],
                [0.596, -0.274, -0.322],
                [0.211, -0.523, 0.312],
            ]
        ]
    ).to(x.device)

    return torch.bmm(A, torch.flatten(x, start_dim=2)).reshape(x.shape)


def yiq_inverse_transform(x: torch.Tensor) -> torch.Tensor:
    A = torch.Tensor(
        [
            [
                [0.299, 0.587, 0.114],
                [0.596, -0.274, -0.322],
                [0.211, -0.523, 0.312],
            ]
        ]
    ).to(x.device)

    return torch.bmm(A.inverse(), torch.flatten(x, start_dim=2)).reshape(x.shape)


def luminance_transfer(content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
    content_yiq = yiq_transform(content)
    style_yiq = yiq_transform(style)

    content_yiq[:, 1:3, :, :] = style_yiq[:, 1:3, :, :]

    return yiq_inverse_transform(content_yiq)


def color_histogram_matching(
    content: torch.Tensor,
    style: torch.Tensor,
) -> torch.Tensor:
    n, c, _, _ = style.shape

    x, y = torch.flatten(style, start_dim=2), torch.flatten(content, start_dim=2)

    mu_x, mu_y = x.mean(2, keepdim=True), y.mean(2, keepdim=True)
    cov_x = torch.bmm(x - mu_x, (x - mu_x).transpose(1, 2)) / x.numel()
    cov_y = torch.bmm(y - mu_y, (y - mu_y).transpose(1, 2)) / y.numel()

    eta_x, eig_x = torch.linalg.eig(cov_x)
    eta_y, eig_y = torch.linalg.eig(cov_y)

    cov_half_x = torch.bmm(
        torch.bmm(eig_x, eta_x.diag_embed().sqrt()),
        eig_x.transpose(1, 2),
    )
    cov_half_y = torch.bmm(
        torch.bmm(eig_y, eta_y.diag_embed().sqrt()),
        eig_y.transpose(1, 2),
    )

    A = torch.bmm(cov_half_y, cov_half_x.inverse()).real

    print(A)

    return (
        torch.bmm(A, x).reshape(style.shape)
        + mu_y.unsqueeze(3)
        - torch.bmm(A, mu_x).unsqueeze(3)
    )
