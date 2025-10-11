import torch

from torchlake.common.utils.numerical import generate_grid

from ..models.densecrf.model import DenseCRF
from ..models.densecrf.network import PermutohedralLattice

B = 4
H = 300
W = 150
N = H * W
C = 20


class TestNetwork:
    def test_permutohedral_lattice_fit_smoothness_kernel(self):
        xy = torch.stack(generate_grid(H, W))

        m = PermutohedralLattice(sigmas=(1, 1))
        m.fit(xy.view(2, -1).T)

    def test_permutohedral_lattice_fit_appearance_kernel(self):
        xy = torch.stack(generate_grid(H, W))
        rgb = torch.randint(0, 256, (3, H, W))

        m = PermutohedralLattice(sigmas=(1, 1, 30, 30, 30))
        m.fit(torch.cat((xy.view(2, -1), rgb.view(3, -1)), 0).T)

    def test_permutohedral_lattice_predict_smoothness_kernel(self):
        xy = torch.stack(generate_grid(H, W))
        x = xy.view(2, -1).T
        y = torch.rand(N, C)

        m = PermutohedralLattice(sigmas=(1, 1))
        m.fit(x)
        m.predict(y)

    def test_permutohedral_lattice_predict_appearance_kernel(self):
        xy = torch.stack(generate_grid(H, W))
        rgb = torch.randint(0, 256, (3, H, W))
        x = torch.cat((xy.view(2, -1), rgb.view(3, -1)), 0).T
        y = torch.rand(N, C)

        m = PermutohedralLattice(sigmas=(1, 1, 30, 30, 30))
        m.fit(x)
        m.predict(y)


class TestModel:
    def test_densecrf_forward_shape(self):
        x = torch.randint(0, 256, (B, 3, H, W))
        yhat = torch.rand(B, C, H, W)

        for img, pred in zip(x, yhat):
            m = DenseCRF(C, 4, 3, 121, 5, 3, 1, True)
            y = m(img, pred)

            assert y.shape == torch.Size((C, H, W))
