import math
from operator import itemgetter
from typing import Sequence

import torch
from torch import nn
from torch_scatter import scatter_add


class PermutohedralLattice(nn.Module):
    def __init__(self, sigmas: Sequence[float]):
        """
        Initialize permutohedral lattice for high-dimensional Gaussian filtering.

        Args:
            feature_dim (int): Dimension of feature vectors (e.g., 2 for spatial, 5 for spatial+RGB).
            sigmas (Sequence[float]): Standard deviation for Gaussian kernel.
        """
        super().__init__()
        self.sigmas = sigmas
        # registry of lattice points
        # key: point, value: index
        self.registry: dict[tuple[int], int] = {}
        # simplices => N, D+1, as indices in the registry
        self.simplices = None
        # neighbors => N, 2(D+1), as indices in the registry
        self.neighbors = None
        # barycentric weights => N, D+1
        self.barycentric_weights = None
        # norms => N, 1
        self.norms = None

    def canonical_simplex(self, d: int) -> torch.Tensor:
        """canonical simplex

        Args:
            d (int): dimension of position vectors, denoted as D

        Returns:
            torch.Tensor: in shape of (D+1, D+1)
        """
        return (
            torch.Tensor(
                [[*([i] * (d + 1 - i)), *([-(d + 1 - i)] * i)] for i in range(d + 1)]
            )
            .long()
            .T
        )

    def lattice_basis(self, d: int) -> torch.Tensor:
        """lattice basis

        Args:
            d (int): dimension of position vectors, denoted as D

        Returns:
            torch.Tensor: in shape of (D+1, D+1)
        """
        ED = d + 1
        return ED * torch.eye(ED) - torch.ones(ED, ED)

    def projection_matrix(self, d: int) -> torch.Tensor:
        """projection matrix

        Args:
            d (int): dimension of position vectors, denoted as D

        Returns:
            torch.Tensor: in shape of (D+1, D)
        """
        # a: d+1 x d * b: d x d => e: d+1 x d
        a = torch.ones(d, d).triu_(1) - torch.arange(d).add(1).diag()
        a = torch.cat((torch.ones(1, d), a), 0)
        b = (1 / (torch.arange(d).add(1) * torch.arange(d).add(2)).sqrt()).diag()
        e = a @ b

        return e

    def fit(self, x: torch.Tensor):
        """
        Build permutohedral lattice with features.
        project position vectors onto lattice
        each position vector is enclosed by a simplex which composed of D+1 vertices
        each vertex is a lattice point belongs to Z^(D+1)
        then compute barycentric weights for splat and slice

        Args:
            x (torch.Tensor): Feature vectors, shape [N, D].
        """
        N, D = x.shape
        # elevated dimension
        ED = D + 1
        device = x.device

        ## step 1: scale
        scaled_x = x / torch.Tensor(self.sigmas).view(1, D).to(device)
        # expected variance of blur is 2/3 * (D+1)^2
        blur_std = math.sqrt(2 / 3) * ED
        scaled_x = scaled_x / blur_std

        ## step 2: embed x onto e
        e = self.projection_matrix(D).to(device)
        p = scaled_x @ e.T

        ## embed => recurrence version
        # p = torch.cat((scaled_x.clone(), torch.empty(B, 1)), 1)
        # alpha_i = (D / (D + 1)) ** 0.5
        # p[:, D] = -p[:, D - 1] * alpha_i
        # for i in range(D - 1, 0, -1):
        #     alpha_j = (i / (i + 1)) ** 0.5
        #     p[:, i] = -p[:, i - 1] * alpha_j + p[:, i] / alpha_i + p[:, i + 1]

        #     # update for next step
        #     alpha_i = alpha_j
        # p[:, 0] = p[:, 0] / alpha_i + p[:, 1]

        ## step 3: find the nearest remainder-0 lattice point
        # rounding
        higher_l0 = (p // ED).ceil() * ED
        lower_l0 = (p // ED).floor() * ED
        l0 = torch.where(p - lower_l0 < higher_l0 - p, lower_l0, higher_l0)
        # sort p - l0 to determine simplex
        residual = p - l0
        _, indices = residual.sort(1, descending=True)
        _, ranks = indices.sort(1)
        # otherwise greedily trackback
        # dont understand
        # https://github.com/lucasb-eyer/pydensecrf/blob/da2c12260e99ed4b9e3f72f4994a49b60c1decea/densecrf/src/permutohedral.cpp#L392
        greedy_check = ranks + l0.sum(1, keepdim=True) / ED
        ranks = torch.where(
            greedy_check < 0,
            greedy_check + ED,
            torch.where(greedy_check > D, greedy_check - ED, greedy_check),
        )
        l0 = torch.where(
            greedy_check < 0,
            l0 + ED,
            torch.where(greedy_check > D, l0 - ED, l0),
        )
        # check Hd property Ex = 0
        assert l0.sum(1).eq(0).all()
        # check permutation property
        # assert (values[:, 0] - values[:, -1]).lt(ED).all()

        ## step 4: barycentric weights in shape of (N, D+1)
        residual = (p - l0) / ED
        b = residual.gather(1, ranks.argsort(dim=1, descending=True)).diff(1, dim=1)
        b = torch.cat((1 - b.sum(1, keepdim=True), b), 1)
        # check b bounds in [0, 1]
        assert b.sum(1).sub(1).lt(1e-5).all()
        self.barycentric_weights = b

        ## step 5: register vertices of simplices
        # D+1, D+1
        canonical_simplex = self.canonical_simplex(D).to(device)
        # N, D+1, D+1
        simplices = l0.unsqueeze(1) + canonical_simplex[None, ...].expand(
            N, ED, ED
        ).gather(2, ranks.unsqueeze(1).expand(N, ED, ED).long())
        points = tuple(
            tuple(point)
            for point in simplices.view(-1, ED).long().detach().cpu().tolist()
        )
        registry = {}
        count = 0
        for point in points:
            if point not in registry:
                registry[point] = count
                count += 1
        self.registry = registry
        M = len(registry)
        self.simplices = (
            torch.Tensor(itemgetter(*points)(self.registry))
            .view(N, ED)
            .long()
            .to(device)
        )

        ## step 6: find neighbors
        # M, 1, D+1
        points = torch.Tensor(tuple(registry.keys())).view(M, 1, ED).to(device)
        # 1, D+1, D+1
        offset = self.lattice_basis(D)[None, ...].to(device)
        # M, 2, D+1, D+1 => number of unique points, neighbors of each point, dimension
        neighbors = torch.stack((points + offset, points - offset), 1)
        # store neighbors in shape of (M, 2, D+1), as indices in the registry
        # if not in the registry, reset to -1
        # they will be ignored during filtering
        self.neighbors = (
            torch.Tensor(
                [
                    self.registry.get(tuple(neighbor), -1)
                    for neighbor in neighbors.view(-1, ED).detach().cpu().tolist()
                ]
            )
            .view(M, 2, ED)
            .long()
            .to(device)
        )
        # check if at least one neighbor in the registry
        assert self.neighbors.ne(-1).any()

        ## step 7: get norm, important
        self.norms = (
            1 / (self._predict(torch.ones(N, 1).to(device)) + 1e-20).sqrt()
        ).to(device)

    def predict(self, y: torch.Tensor) -> torch.Tensor:
        y = y * self.norms
        y = self._predict(y)
        y = y * self.norms

        return y

    def _predict(self, y: torch.Tensor) -> torch.Tensor:
        """
        Apply Gaussian filtering using permutohedral lattice.

        Args:
            y (torch.Tensor): Values to filter, shape [N, C].

        Returns:
            torch.Tensor: Filtered values, shape [N, C].
        """
        N, C = y.shape
        # elevated dimension
        _, ED = self.barycentric_weights.shape
        D = ED - 1
        M = len(self.registry)
        device = y.device

        # step 1: splat
        # separate value onto each lattice points of canonical simplex

        # N, C => N, D+1, C
        y = torch.einsum("bi, bj -> bij", self.barycentric_weights, y)

        # N, D+1, C => M, C
        y = scatter_add(y.view(-1, C), self.simplices.view(-1), dim=0)

        ## add a slot for outbound neighbors

        # M+1, C
        y = torch.cat((torch.zeros(1, C).to(device), y), 0)

        # step 2: blur
        # convolve along each direction of the lattice with weights [1,2,1]
        for d in range(ED):
            # M, 2 => M, 2, C => M, C
            # M, C + M, C => M, C
            y[1:] += y[(self.neighbors[:, :, d] + 1).view(-1)].view(M, 2, C).mean(dim=1)

        # step 3: slice
        # use precomputed barycentric weights to reproduce position vectors

        ## transform back by barycentric weights

        # M, C => N, D+1, C
        y = y[self.simplices.view(-1) + 1].view(N, ED, C)

        # (N, D+1, C) x (N, D+1) => N, C
        y = torch.einsum("bij, bi -> bj", y, self.barycentric_weights)

        # magic
        # https://github.com/lucasb-eyer/pydensecrf/blob/da2c12260e99ed4b9e3f72f4994a49b60c1decea/densecrf/src/permutohedral.cpp#L509
        alpha = 1 / (1 + 2 ** (-D))
        y *= alpha

        return y
