from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def load_image(path: str, is_numpy: bool = False, is_tensor: bool = False):
    if isinstance(path, Path):
        path = path.as_posix()

    if is_numpy:
        return cv2.imread(path)[:, :, ::-1]

    image = Image.open(path)
    if is_tensor:
        image = transforms.ToTensor()(image)

    return image


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
