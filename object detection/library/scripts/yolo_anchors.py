from pathlib import Path

import click
import torch
from constants.enums import OperationMode
from datasets.coco.datasets import COCODatasetFromCSV
from datasets.voc.datasets import VOCDatasetFromCSV
from utils.anchors import dist_metric, kmeans
from utils.config import load_classes, load_config
from utils.plot import show_anchors


@click.command()
@click.option(
    "-c",
    "--config",
    "config",
    type=str,
    help="which config to use",
    required=True,
)
@click.option(
    "-d",
    "--dataset",
    "dataset",
    type=str,
    help="which dataset to use",
    required=True,
)
@click.option(
    "-show",
    "--show",
    "show",
    is_flag=True,
    help="whether to show the result or not",
)
@click.option(
    "--save-dir",
    "save_dir",
    type=str,
    help="where to save the result",
    default="",
)
def main(
    config: str,
    dataset: str,
    show: bool,
    save_dir: str,
):
    dataset = dataset.upper()

    cfg = load_config(config)
    class_names = load_classes(cfg["DATA"][dataset]["CLASSES_PATH"])

    cluster_num = cfg["MODEL"]["NUM_ANCHORS"]

    dataset_class_mapping = {
        "VOC": VOCDatasetFromCSV,
        "COCO": COCODatasetFromCSV,
    }

    # comment lines of raw data
    dataset = dataset_class_mapping.get(dataset)(
        root=cfg["DATA"][dataset]["ROOT"],
        csv_root=cfg["DATA"][dataset]["CSV_ROOT"],
        class_names=class_names,
        mode=OperationMode.TRAIN.value,
        # transform=A.Compose(
        #     [
        #         A.Resize(100, 100),
        #         A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        #         ToTensorV2(),
        #     ]
        # ),
    )

    # dataloader = torch.utils.data.DataLoader(
    #     dataset,
    #     batch_size=32,
    #     collate_fn=collate_fn,
    #     shuffle=False,
    # )

    wh = torch.from_numpy(dataset.table[["w", "h"]].to_numpy())

    # wh = torch.cat(
    #     [
    #         torch.Tensor(label)[:, 2:4]
    #         for _, labels in tqdm(dataloader)
    #         for label in labels
    #     ],
    #     0,
    # )
    print("gt shape: ", wh.shape)

    # debug
    # wh = torch.rand(100,2)

    group_index, anchor_wh = kmeans(wh, cluster_num)

    final_iou = (
        sum(
            [
                (1 - dist_metric(wh[group_index == i], anchor_wh))[:, i].mean().item()
                for i in range(cluster_num)
            ]
        )
        / cluster_num
    )
    print("final mean IOU: ", final_iou)

    print(
        "member number in each group",
        (group_index.view(-1, 1) == torch.arange(cluster_num).view(1, -1)).sum(0),
    )

    with open(Path(save_dir).joinpath("anchors.txt"), "w") as f:
        for w, h in anchor_wh.tolist():
            print(f"{w},{h}", file=f)

    if show:
        show_anchors("anchors.txt")
