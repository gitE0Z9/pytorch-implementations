from PIL import Image
import numpy as np


def save_img_array(array: np.ndarray, file_path: str):
    if array.dtype == np.float32:
        array = (array.clip(0, 1) * 255).clip(0, 255).astype(np.uint8)
    elif array.dtype == np.uint8:
        array = array.clip(0, 255)
    else:
        raise NotImplementedError

    image = Image.fromarray(array)
    image.save(file_path)
