import torch

from tqdm import tqdm
from yolov2.datasets import VOC_dataset_raw
from yolov2.utils.config import load_config, load_classes


def dist_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """compute iou to each centroid"""
    # compute pairwise iou
    iou = []
    for i in range(y.size(0)):
        intersection_wh = torch.min(x, y[i])  # N, 2
        intersection = intersection_wh[:, 0]*intersection_wh[:, 1]  # N
        union = x[:, 0]*x[:, 1] + y[i, 0]*y[i, 1] - intersection  # N
        iou.append(intersection / union)
    iou = torch.stack(iou, 1)

    # convert to distance
    distance = 1 - iou
    return distance

def avg_iou(x: torch.Tensor, group: torch.Tensor, center: torch.Tensor) -> float:
    cluster_num = center.size(0)
    within_cluster_iou = [(1-dist_metric(x[group==i], center))[:,i].mean().item() for i in range(cluster_num)]
    iou = sum(within_cluster_iou)/cluster_num
    return iou 

def kmeans(x: torch.Tensor, cluster_num: int = 5, total_iter: int = 300) -> torch.Tensor:
    """generate kmeans index and center"""
    # init group
    init_point = torch.rand(cluster_num, 2)
    distance = dist_metric(x, init_point)
    group_index = distance.argmin(1)

    # iteratively update group
    prev_avg_iou = avg_iou(x, group_index, init_point)
    print('init mean IOU: ', prev_avg_iou)

    for _ in range(total_iter):
        for i in range(cluster_num):
            init_point[i] = x[group_index.eq(i)].mean(0)
        distance = dist_metric(x, init_point)
        group_index = distance.argmin(1)
        
        new_avg_iou = avg_iou(x, group_index, init_point)
        print('mean IOU: ', new_avg_iou)

        # early stopping
        if new_avg_iou - prev_avg_iou > 1e-2:
            prev_avg_iou = new_avg_iou
        else:
            break

    # final group and cluster center
    return group_index, init_point


if __name__ == "__main__":

    cfg, cls_name = load_config('config/resnet18.yml'), load_classes('config/voc_classes.txt')

    cluster_num = cfg['MODEL']['NUM_ANCHORS']

    dataset = VOC_dataset_raw(root=cfg['DATA']['VOC']['ROOT'], class_name=cls_name)
    wh = torch.cat([torch.Tensor(label)[:,2:4] for _, label in tqdm(dataset)],0)
    print('gt shape: ', wh.shape)

    # # wh = torch.rand(100,2) # debug

    group_index, anchor_wh = kmeans(wh, cluster_num)

    final_iou = sum([(1-dist_metric(wh[group_index==i], anchor_wh))[:,i].mean().item() for i in range(cluster_num)])/cluster_num
    print('final avg IOU: ', final_iou)

    print('group member number',(group_index.view(-1,1) == torch.arange(cluster_num).view(1,-1)).sum(0))

    with open('anchors.txt', 'w') as f:
        for w, h in anchor_wh.tolist():
            print(f'{w},{h}', file=f)
