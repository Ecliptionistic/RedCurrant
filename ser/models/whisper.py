import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, WhisperFeatureExtractor
from .base import TorchSERModel
from .wav2vec import EncoderNet

SR = 16000
N_FFT, HOP = 400, 160
ENC_STRIDE = 2 * HOP


class WhisperNet(EncoderNet):
    '''
    Энкодер Whisper (многоязычный, обучен с учителем на ASR), без декодера
    '''
    def __init__(self, n_classes, checkpoint="openai/whisper-small", window_sec=10.0, layer_weights=True,
                 head_dim=256, dropout=0.1, freeze_encoder=True):
        cfg = AutoConfig.from_pretrained(checkpoint)
        cfg.encoder_layerdrop = 0.0
        encoder = AutoModel.from_pretrained(checkpoint, config=cfg).get_encoder()
        cfg = encoder.config
        n_pos = int(window_sec * 1000 / 20)

        if n_pos > cfg.max_source_positions:
            raise ValueError(f"window_sec={window_sec} больше окна Whisper (30 с)")

        if n_pos < cfg.max_source_positions:
            old = encoder.embed_positions
            new = nn.Embedding(n_pos, old.embedding_dim)
            new.weight.data.copy_(old.weight.data[:n_pos])
            new.weight.requires_grad_(False)
            encoder.embed_positions = new
            cfg.max_source_positions = n_pos

        super().__init__(encoder, cfg.encoder_layers, cfg.d_model, n_classes,
                         layer_weights, head_dim, dropout, freeze_encoder)
        self.n_pos, self.window = n_pos, int(window_sec * SR)
        fe = WhisperFeatureExtractor(feature_size=cfg.num_mel_bins)
        self.register_buffer("mel_filters", torch.from_numpy(fe.mel_filters).float(), persistent=False)
        self.register_buffer("stft_window", torch.hann_window(N_FFT), persistent=False)

    def _logmel(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            spec = torch.stft(x.float(), N_FFT, HOP, window=self.stft_window, return_complex=True)
            mel = self.mel_filters.T @ (spec[..., :-1].abs() ** 2)
            log = mel.clamp(min=1e-10).log10()
            log = torch.maximum(log, log.amax(dim=(1, 2), keepdim=True) - 8.0)

        return (log + 4.0) / 4.0

    def encode(self, x, lengths):
        x = F.pad(x[:, :self.window], (0, max(0, self.window - x.size(1))))
        feats = self._logmel(x)

        with torch.set_grad_enabled(self.training and self.encoder_trainable):
            out = self.encoder(feats, output_hidden_states=self.layer_w is not None)

        flen = (lengths.clamp(max=self.window) // ENC_STRIDE).clamp(1, self.n_pos)

        return out.hidden_states, out.last_hidden_state, flen


class WhisperModel(TorchSERModel):
    '''
    Модель на энкодере Whisper. max_sec лучше ставить равным net["window_sec"]
    '''
    def build_net(self):
        return WhisperNet(self.n_classes, **self.net_kwargs)
