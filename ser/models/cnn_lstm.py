import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import TorchSERModel
from .bilstm import LogMelFrontend, AttnPool


class CNNLSTMNet(nn.Module):
    '''
    Лог-мел -> блоки Conv2d (BatchNorm, GELU, MaxPool 2x2: время и частота уменьшаются вдвое) ->
    Bi-LSTM -> attention-пулинг -> голова
    '''
    def __init__(self, n_classes, n_mels=80, channels=(32, 64), hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.front = LogMelFrontend(n_mels)
        self.down = 2 ** len(channels)
        blocks, c_in = [], 1

        for c in channels:
            blocks += [nn.Conv2d(c_in, c, 3, padding=1), nn.BatchNorm2d(c), nn.GELU(), nn.MaxPool2d(2)]
            c_in = c

        self.cnn = nn.Sequential(*blocks)
        self.proj = nn.Sequential(nn.Linear(c_in * (n_mels // self.down), hidden), nn.LayerNorm(hidden),
                                  nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.pool = AttnPool(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x, lengths):
        m, flen, _ = self.front(x, lengths)
        h = self.cnn(m.unsqueeze(1))  # (B, C, T/down, F/down)
        h = self.proj(h.permute(0, 2, 1, 3).flatten(2))
        flen = (flen // self.down).clamp(min=1, max=h.size(1))
        mask = torch.arange(h.size(1), device=x.device)[None] < flen[:, None]
        packed = pack_padded_sequence(h, flen.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=h.size(1))
        return self.head(self.pool(out, mask))


class CNNLSTMModel(TorchSERModel):
    '''
    Модель CNN + Bi-LSTM на лог-мел-спектрограмме
    '''
    def build_net(self):
        return CNNLSTMNet(self.n_classes, **self.net_kwargs)
