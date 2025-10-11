import torch
from torch import nn
import torch.nn.functional as F
from torchlake.common.utils.numerical import generate_grid

from torchlake.semantic_segmentation.models.densecrf.network import (
    PermutohedralLattice,
)


class DenseCRF(nn.Module):

    def __init__(
        self,
        num_class: int,
        appearance_weight: float,
        smoothness_weight: float,
        proximity_std: float,
        color_std: float,
        smoothness_std: float,
        iters: int = 1,
        potts_model: bool = True,
    ):
        super().__init__()
        self.num_class = num_class
        self.appearance_weight = appearance_weight
        self.smoothness_weight = smoothness_weight
        self.proximity_std = proximity_std
        self.color_std = color_std
        self.smoothness_std = smoothness_std
        self.iters = iters

        self.bilateral_lattice = PermutohedralLattice(
            sigmas=(
                self.proximity_std,
                self.proximity_std,
                self.color_std,
                self.color_std,
                self.color_std,
            ),
        )
        self.smoothness_lattice = PermutohedralLattice(
            sigmas=(
                self.smoothness_std,
                self.smoothness_std,
            )
        )

        # build incompatibility
        mu = self.build_incompatibilty(num_class, potts_model)
        if not potts_model:
            self.mu = nn.Parameter(mu)
        else:
            self.register_buffer("mu", mu, persistent=False)

    def build_incompatibilty(
        self,
        num_class: int,
        potts_model: bool = True,
    ) -> torch.Tensor:
        """_summary_

        Args:
            yhat (torch.Tensor): output label, in shape of (O, H, W)
            potts_model (bool, optional): use potts model, if False, use learnable weights. Defaults to True.

        Returns:
            torch.Tensor: incompatibility matrix
        """
        if potts_model:
            return torch.ones(num_class) - torch.eye(num_class)
        else:
            return torch.rand(num_class, num_class)

    def forward(self, x: torch.Tensor, yhat: torch.Tensor) -> torch.Tensor:
        """iteratively minimize energy of CRF

        Args:
            x (torch.Tensor): input image
            yhat (torch.Tensor): output logit

        Returns:
            torch.Tensor: output logit
        """
        _, H, W = x.shape
        C, _, _ = yhat.shape

        # 2, h, w
        pos = torch.stack(generate_grid(H, W)).to(x.device)

        # N, 5
        self.bilateral_lattice.fit(torch.cat((pos, x), 0).view(5, -1).T)

        # N, 2
        self.smoothness_lattice.fit(pos.view(2, -1).T)

        # C, N
        unary_potential = yhat.reshape(C, -1)
        # C, N
        output = unary_potential.softmax(0)
        # convergenece criterion can also be threshold?
        for _ in range(self.iters):
            # apperance kernel
            # N
            appearance_kernel = self.bilateral_lattice.predict(output.T)

            # smoothness kernel
            # N
            smoothness_kernel = self.smoothness_lattice.predict(output.T)

            # N
            pairwise_potential = (
                self.appearance_weight * appearance_kernel
                + self.smoothness_weight * smoothness_kernel
            )

            # C, N
            pairwise_potential = torch.einsum(
                "ij, bi -> bj",
                self.mu,
                pairwise_potential,
            ).T

            # TODO: if incompatibility is learnable

            # C, N
            unary_potential = unary_potential + pairwise_potential
            output = unary_potential.softmax(0)

        return output.view(C, H, W)
