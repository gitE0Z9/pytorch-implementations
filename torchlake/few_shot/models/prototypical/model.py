import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation


class PrototypicalNetwork(nn.Module):
    def __init__(self, input_channel: int):
        super(PrototypicalNetwork, self).__init__()
        self.conv_1 = nn.Sequential(
            Conv2dNormActivation(input_channel, 64, 3),
            nn.MaxPool2d(2),
        )
        self.conv_2 = nn.Sequential(
            Conv2dNormActivation(64, 64, 3),
            nn.MaxPool2d(2),
        )
        self.conv_3 = nn.Sequential(
            Conv2dNormActivation(64, 64, 3),
            nn.MaxPool2d(2),
        )
        self.conv_4 = nn.Sequential(
            Conv2dNormActivation(64, 64, 3),
            nn.MaxPool2d(2),
        )

    def feature_extract(self, x: torch.Tensor) -> torch.Tensor:
        # share embedding
        y = self.conv_1(x)
        y = self.conv_2(y)
        y = self.conv_3(y)
        y = self.conv_4(y)

        # flatten space
        y = torch.flatten(y, start_dim=1)

        return y

    def forward(self, query: torch.Tensor, support_set: torch.Tensor) -> torch.Tensor:
        # batch, embed
        query_vector = self.feature_extract(query)

        batch_size, num_class, shot_size, c, h, w = support_set.shape
        support_vectors = self.feature_extract(support_set.reshape(-1, c, h, w))
        support_vectors = support_vectors.reshape(
            batch_size,
            num_class,
            shot_size,
            query_vector.size(-1),
        )

        # batch, class, embed
        prototypes = support_vectors.mean(2)

        # batch, 1, class
        dist = torch.cdist(query_vector.unsqueeze(1), prototypes)

        return dist.squeeze(1)
