import torch
from tqdm import tqdm
from pprint import pprint
from yolov2.datasets import VOC_dataset


def show_gt_num(loader):
    """show the number of ground truth in a dataset"""
    gt_num = 0
    for _, label in tqdm(loader):
        gt_num += label[:, :, 5:, :, :].sum((0, 1, 3, 4))

    pprint(gt_num)


if __name__ == "__main__":
    dataset = VOC_dataset(
        root="D://research/pytorch-implementations/data", year="2007", mode="test"
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=32, num_workers=4, pin_memory=True
    )
    show_gt_num(loader)
