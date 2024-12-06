import torch

from ...controller.weight_manager import WeightManager


class TestSuccess:
    def setUp(self):
        self.format = ".{network_name}.{epoch}.pth"
        self.manager = WeightManager(self.format)
        self.filename = self.manager.get_filename(network_name="ssd", epoch=10)

    def tearDown(self):
        self.filename.unlink()

    def test_get_filename(self):
        self.setUp()
        assert ".ssd.10.pth" == self.filename.name

    def test_save_weight(self):
        self.setUp()

        model = torch.nn.Linear(1, 1)
        self.manager.save_weight(model.state_dict(), self.filename)

        assert self.filename.exists()

        self.tearDown()

    def test_load_weight(self):
        self.setUp()

        model = torch.nn.Linear(1, 1)
        self.manager.save_weight(model.state_dict(), self.filename)

        self.manager.load_weight(self.filename, model)

        self.tearDown()
