import torch
import torch.nn as nn
import torchaudio
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .base import TorchSERModel

SR = 16000


def masked_mean_std(x, mask):
    w = mask.unsqueeze(-1).to(x.dtype)
    n = w.sum(1, keepdim=True).clamp(min=1)
    mu = (x * w).sum(1, keepdim=True) / n
    sd = (((x - mu) ** 2 * w).sum(1, keepdim=True) / n).sqrt()
    return mu, sd


class LogMelFrontend(nn.Module):
    def __init__(self, n_mels=80, n_fft=400, hop=160):
        super().__init__()
        self.hop = hop
        self.mel = torchaudio.transforms.MelSpectrogram(SR, n_fft=n_fft, hop_length=hop, n_mels=n_mels)

    def forward(self, x, lengths):
        with torch.autocast(device_type=x.device.type, enabled=False):
            m = (self.mel(x.float()) + 1e-6).log().transpose(1, 2)

        flen = (lengths // self.hop + 1).clamp(max=m.size(1))
        mask = torch.arange(m.size(1), device=x.device)[None] < flen[:, None]
        mu, sd = masked_mean_std(m, mask)

        return (m - mu) / (sd + 1e-5) * mask.unsqueeze(-1), flen, mask


class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.att = nn.Linear(d, 1)

    def forward(self, h, mask):
        s = self.att(h).squeeze(-1).masked_fill(~mask, torch.finfo(h.dtype).min)
        return (h * s.softmax(1).unsqueeze(-1)).sum(1)


class BiLSTMNet(nn.Module):
    def __init__(self, n_classes, n_mels=80, hidden=128, layers=2, dropout=0.3):
        super().__init__()
        self.front = LogMelFrontend(n_mels)
        self.proj = nn.Sequential(nn.Linear(n_mels, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.pool = AttnPool(2 * hidden)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, n_classes))

    def forward(self, x, lengths):
        m, flen, mask = self.front(x, lengths)
        h = self.proj(m)
        packed = pack_padded_sequence(h, flen.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=h.size(1))
        return self.head(self.pool(out, mask))


class BiLSTMModel(TorchSERModel):
    '''
    Модель Bi-LSTM
    '''
    def build_net(self):
        return BiLSTMNet(self.n_classes, **self.net_kwargs)
