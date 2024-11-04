import matplotlib.pyplot as plt
import numpy as np

from .random import rand_color


def rand_color_map(class_names: list[str]) -> dict[str, list[int]]:
    return {class_name: rand_color() for class_name in class_names}


def plot_grids(
    images: list[np.ndarray],
    titles: list[str] | None = None,
    num_row: int = 1,
    num_col: int = 1,
    figsize: tuple[int] = (8, 8),
    is_gray_scale: bool = False,
):
    plt.figure(figsize=figsize)

    for idx, image in enumerate(images):
        plt.subplot(num_row, num_col, idx + 1)
        if titles:
            title = titles[idx]
            plt.title(title)
        plt.axis("off")

        if is_gray_scale:
            plt.imshow(image, cmap="gray")
        else:
            plt.imshow(image)
