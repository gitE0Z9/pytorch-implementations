import matplotlib.pyplot as plt
import numpy as np


def plot_grids(
    images: list[np.ndarray],
    titles: list[str],
    num_row: int = 1,
    num_col: int = 1,
    figsize: tuple[int] = (8, 8),
):
    plt.figure(figsize=figsize)

    if not titles:
        titles = [0 for _ in range(len(images))]

    for idx, (image, title) in enumerate(images, titles):
        plt.subplot(num_row, num_col, idx + 1)
        if title:
            plt.title(title)
        plt.axis("off")
        plt.imshow(image)
