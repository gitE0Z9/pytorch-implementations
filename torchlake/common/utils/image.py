import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


def load_image(path: str, is_numpy: False, is_tensor: bool = False):
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
