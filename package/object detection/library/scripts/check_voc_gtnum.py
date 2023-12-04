from pprint import pprint
from datasets.voc.datasets import VOCDatasetFromCSV
from utils.config import load_classes


def show_gt_num(dataset):
    """show the number of ground truth in a dataset"""
    mapping = dataset.table.groupby("class_id")["name"].count().to_dict()
    pprint({dataset.class_name[k]: v for k, v in mapping.items()})


if __name__ == "__main__":
    dataset = VOCDatasetFromCSV(
        root=""
        csv_root="config",
        class_name=load_classes("config/voc_classes.txt"),
        mode="test",
        transform=None,
    )
    show_gt_num(dataset)
