import torch

from tqdm import tqdm
from datasets.voc.datasets import VOCDatasetRaw
from datasets.coco.datasets import COCODatasetRaw
from utils.config import load_config, load_classes
from utils.anchors import kmeans, dist_metric
from constants.enums import OperationMode

if __name__ == "__main__":
    cfg = load_config("configs/yolov2/resnet18.yml")
    cls_name = load_classes("datasets/voc/voc_classes.txt")

    cluster_num = cfg["MODEL"]["NUM_ANCHORS"]

    dataset = VOCDatasetRaw(
        root=cfg["DATA"]["VOC"]["ROOT"],
        class_name=cls_name,
        mode=OperationMode.TRAIN.value,
    )
    wh = torch.cat([torch.Tensor(label)[:, 2:4] for _, label in tqdm(dataset)], 0)
    print("gt shape: ", wh.shape)

    # # wh = torch.rand(100,2) # debug

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
    print("final avg IOU: ", final_iou)

    print(
        "group member number",
        (group_index.view(-1, 1) == torch.arange(cluster_num).view(1, -1)).sum(0),
    )

    with open("anchors.txt", "w") as f:
        for w, h in anchor_wh.tolist():
            print(f"{w},{h}", file=f)
