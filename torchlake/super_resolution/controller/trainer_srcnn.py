from torchlake.common.controller.trainer import RegressionTrainer
import torchvision.transforms.functional as VF


class SRCNNTrainer(RegressionTrainer):
    def _calc_loss(self, y_hat, row, criterion):
        _, y = row
        y = y.to(self.device)

        y = VF.center_crop(y, y_hat.shape[-2:])

        return criterion(y_hat, y)
