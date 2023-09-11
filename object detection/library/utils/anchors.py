import torch


def dist_metric(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """compute iou to each centroid"""
    # compute pairwise iou
    iou = []
    for i in range(y.size(0)):
        intersection_wh = torch.min(x, y[i])  # N, 2
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]  # N
        union = x[:, 0] * x[:, 1] + y[i, 0] * y[i, 1] - intersection  # N
        iou.append(intersection / union)
    iou = torch.stack(iou, 1)

    # convert to distance
    distance = 1 - iou
    return distance


def avg_iou(x: torch.Tensor, group: torch.Tensor, center: torch.Tensor) -> float:
    cluster_num = center.size(0)
    within_cluster_iou = [
        (1 - dist_metric(x[group == i], center))[:, i].mean().item()
        for i in range(cluster_num)
    ]
    iou = sum(within_cluster_iou) / cluster_num
    return iou


def kmeans(
    x: torch.Tensor,
    cluster_num: int = 5,
    total_iter: int = 300,
    error_acceptance: float = 1e-2,
) -> torch.Tensor:
    """generate kmeans index and center"""
    # init group
    init_point = torch.rand(cluster_num, 2)
    distance = dist_metric(x, init_point)
    group_index = distance.argmin(1)

    # iteratively update group
    prev_avg_iou = avg_iou(x, group_index, init_point)
    print("init mean IOU: ", prev_avg_iou)

    for _ in range(total_iter):
        for i in range(cluster_num):
            init_point[i] = x[group_index.eq(i)].mean(0)
        distance = dist_metric(x, init_point)
        group_index = distance.argmin(1)

        new_avg_iou = avg_iou(x, group_index, init_point)
        print("mean IOU: ", new_avg_iou)

        # early stopping
        if new_avg_iou - prev_avg_iou > error_acceptance:
            prev_avg_iou = new_avg_iou
        else:
            break

    # final group and cluster center
    return group_index, init_point
