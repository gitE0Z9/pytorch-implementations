from typing import Callable, Literal

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean


class KMeans(nn.Module):
    def __init__(
        self,
        k: int,
        total_iter: int = 300,
        error_acceptance: float = 1e-2,
        dist_metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        eval_metric: (
            Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float] | None
        ) = None,
        init_method: Literal["uniform", "random", "kmeans++"] = "kmeans++",
    ):
        """KMeans

        Args:
            k (int): number of clusters
        """
        super(KMeans, self).__init__()
        assert k > 1, "number of clusters should be larger than 1"

        self.k = k
        self.total_iter = total_iter
        self.error_acceptance = error_acceptance
        self.dist_metric = dist_metric or self.build_dist_metric()
        self.eval_metric = eval_metric or self.build_eval_metric()
        self.init_method = init_method
        self.centroids = None

    def build_dist_metric(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        return lambda x, y: torch.cdist(x, y, p=2)

    def build_eval_metric(
        self,
    ) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], float]:
        return lambda x, i, c: F.mse_loss(x, c[i])

    def init_centroids(self, x: torch.Tensor):
        n, c = x.shape

        if self.init_method == "uniform":
            self.centroids = torch.rand(self.k, c)
        elif self.init_method == "random":
            indices = torch.multinomial(
                torch.ones_like(x[:, 0]),
                num_samples=self.k,
                replacement=False,
            )
            self.centroids = x[indices]
        elif self.init_method == "kmeans++":
            visited = torch.randint(0, n, (1,))
            for _ in range(1, self.k):
                prob = self.dist_metric(x, x[visited]).min(1)[0]
                prob[visited] = 0

                index = torch.multinomial(prob, num_samples=1)
                visited = torch.cat([visited, index])

            self.centroids = x[visited]
        else:
            raise NotImplementedError

    def fit(self, x: torch.Tensor) -> torch.Tensor:
        """fit a kmeans model

        Args:
            x (torch.Tensor): input tensor, shape is (...other shapes, channel)

        Returns:
            torch.Tensor: group indices
        """
        original_shape, channel = x.shape[:-1], x.size(-1)
        # n, c
        x = x.view(-1, channel)
        # k, c
        self.init_centroids(x)

        # init group

        # n
        i = self.predict(x)

        # iteratively update group
        prev_score = self.score(x, i)
        print("init evaluation score:", prev_score)
        for _ in range(self.total_iter):
            # k, c
            self.centroids = scatter_mean(
                x,
                i,
                dim=0,
                out=torch.zeros_like(self.centroids),
            )
            # n
            i = self.predict(x)

            new_score = self.score(x, i)
            print("new evaluation score:", new_score)

            # early stopping
            if new_score - prev_score > self.error_acceptance:
                prev_score = new_score
            else:
                break

        # n
        return i.view(original_shape)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        # n, c x k, c => n, c
        distance = self.dist_metric(x, self.centroids)
        # n
        return distance.argmin(1)

    def score(self, x: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        return self.eval_metric(x, i, self.centroids)

    def transform(self, i: torch.Tensor) -> torch.Tensor:
        return self.centroids[i]

    def predict_and_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform(self.predict(x))
