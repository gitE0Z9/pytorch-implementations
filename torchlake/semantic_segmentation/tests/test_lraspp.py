import torch

from ..models.lr_aspp import LRASPP, MobileNetV3Seg


class TestMobileNetV3Seg:
    def setUp(self):
        self.x = torch.rand((2, 3, 1024, 2048))

    def test_backbone_forward_shape(self):
        self.setUp()
        model = MobileNetV3Seg(21)

        features = model.backbone(self.x)

        for scale, dim, feature in zip([8, 16], [40, 160], features):
            assert feature.shape == torch.Size((2, dim, 1024 // scale, 2048 // scale))

    def test_forward_shape(self):
        self.setUp()
        model = MobileNetV3Seg(21)

        y = model(self.x)

        assert y.shape == torch.Size((2, 21, 1024, 2048))

    def test_backward(self):
        self.setUp()
        y = torch.randint(0, 21, (2, 1024, 2048))

        model = MobileNetV3Seg(21)

        criterion = torch.nn.CrossEntropyLoss()

        yhat = model(self.x)
        loss = criterion(yhat, y)

        loss.backward()

        assert not torch.isnan(loss)


class TestLRASPP:
    def test_forward_shape(self):
        shallow_x = torch.rand((2, 32, 1024 // 8, 2048 // 8))
        deep_x = torch.rand((2, 32, 1024 // 16, 2048 // 16))

        model = LRASPP([32, 32], 128, 19)
        y = model(shallow_x, deep_x)

        assert y.shape == torch.Size((2, 19, 1024 // 8, 2048 // 8))
