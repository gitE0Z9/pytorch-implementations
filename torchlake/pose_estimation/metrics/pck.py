import torch


class PCK:

    def __init__(self, output_size: int, threshold: float = 0.5):
        self.output_size = output_size
        self.threshold = threshold
        self.hits = torch.zeros(output_size, 2)

    def update(self, gt: torch.Tensor, pred: torch.Tensor):
        """update hits

        Args:
            gt (torch.Tensor): shape B, C, 2/3
            pred (torch.Tensor): shape B, C, 2
        """
        dist = torch.norm(gt[:, :, :2].float() - pred.float(), p=2, dim=-1)
        hits = dist.lt(self.threshold)
        self.hits[:, 0] += hits.sum(0)
        self.hits[:, 1] += (1 - hits).sum(0)
