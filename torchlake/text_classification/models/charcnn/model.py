import torch
from torch import nn
from torchlake.common.schemas.nlp import NlpContext

from .network import CharQuantization


class CharCnn(nn.Module):
    def __init__(self, char_size: int, label_size: int, context: NlpContext):
        super(CharCnn, self).__init__()
        # paper page 3
        # l_6 = (l_0 - 96) / 3**3
        # input dim = l_6 * frame_size
        self.fc_channel_size = int((context.max_seq_len - 96) / 27 * 256)

        self.quantization = CharQuantization(char_size, context)
        self.conv = nn.Sequential(
            nn.Conv1d(char_size, 256, 7),
            nn.ReLU(True),
            nn.MaxPool1d(3, 3),
            nn.Conv1d(256, 256, 7),
            nn.ReLU(True),
            nn.MaxPool1d(3, 3),
            nn.Conv1d(256, 256, 3),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 3),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 3),
            nn.ReLU(True),
            nn.Conv1d(256, 256, 3),
            nn.ReLU(True),
            nn.MaxPool1d(3, 3),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_channel_size, 1024),
            nn.Dropout(),
            nn.Linear(1024, 1024),
            nn.Dropout(),
            nn.Linear(1024, label_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.quantization(x).transpose(-1, -2).float()
        y = self.conv(y).transpose(-1, -2).reshape(-1, self.fc_channel_size)

        y = self.fc(y)

        return y
