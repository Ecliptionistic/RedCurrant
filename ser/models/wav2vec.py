import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from .base import TorchSERModel
from .bilstm import masked_mean_std


class EncoderNet(nn.Module):
    '''
    Общая часть для предобученных энкодеров: взвешенная сумма слоёв, пулинг по времени, голова

    Подклассы задают энкодер и реализуют encode(x, lengths) -> (hidden_states, last_hidden_state, frame_lengths)
    '''
    frozen_prefix = ()

    def __init__(self, encoder, n_layers, hidden, n_classes, layer_weights, head_dim, dropout, freeze_encoder):
        super().__init__()
        self.encoder = encoder
        self.layer_w = nn.Parameter(torch.zeros(n_layers + 1)) if layer_weights else None
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, head_dim), nn.GELU(),
                                  nn.Dropout(dropout), nn.Linear(head_dim, n_classes))
        self.set_encoder_trainable(not freeze_encoder)

    def _encoder_params(self):
        return [p for n, p in self.encoder.named_parameters() if not n.startswith(self.frozen_prefix)]

    def set_encoder_trainable(self, flag):
        self.encoder_trainable = flag
        for p in self._encoder_params():
            p.requires_grad = flag

    def param_groups(self, lr, encoder_lr):
        rest = [p for n, p in self.named_parameters() if not n.startswith("encoder.")]
        return [{"params": rest, "lr": lr}, {"params": self._encoder_params(), "lr": encoder_lr or lr}]

    def encode(self, x, lengths):
        raise NotImplementedError

    def forward(self, x, lengths):
        hs, last, flen = self.encode(x, lengths)

        if self.layer_w is not None:
            hs = torch.stack(hs, 0)
            h = (self.layer_w.softmax(0).to(hs.dtype)[:, None, None, None] * hs).sum(0)
        else:
            h = last

        fw = (torch.arange(h.size(1), device=h.device)[None] < flen[:, None]).unsqueeze(-1).to(h.dtype)

        return self.head((h * fw).sum(1) / fw.sum(1).clamp(min=1))


class SSLNet(EncoderNet):
    '''
    Энкодеры на сырой волне: wav2vec 2.0, HuBERT, WavLM, XLS-R, MMS, mHuBERT (любой checkpoint для AutoModel)
    '''
    frozen_prefix = ("feature_extractor",)

    def __init__(self, n_classes, checkpoint="facebook/wav2vec2-base", layer_weights=True,
                 head_dim=256, dropout=0.1, freeze_encoder=True, layerdrop=0.0):
        # layerdrop по умолчанию 0.1: при обучении пропущенные слои пропадают из hidden_states
        # (transformers 5.x), и взвешенная сумма слоёв ломается, поэтому он отключён
        cfg = AutoConfig.from_pretrained(checkpoint)
        cfg.layerdrop = layerdrop
        encoder = AutoModel.from_pretrained(checkpoint, config=cfg)
        encoder.feature_extractor._freeze_parameters()  # freeze_feature_encoder() есть не у всех (HubertModel)
        cfg = encoder.config
        super().__init__(encoder, cfg.num_hidden_layers, cfg.hidden_size, n_classes,
                         layer_weights, head_dim, dropout, freeze_encoder)
        self.use_attn_mask = getattr(cfg, "feat_extract_norm", "layer") == "layer"

    def encode(self, x, lengths):
        mask = torch.arange(x.size(1), device=x.device)[None] < lengths[:, None]
        mu, sd = masked_mean_std(x.unsqueeze(-1), mask)
        x = ((x.unsqueeze(-1) - mu) / (sd + 1e-7)).squeeze(-1) * mask

        with torch.set_grad_enabled(self.training and self.encoder_trainable):
            out = self.encoder(x, attention_mask=mask.long() if self.use_attn_mask else None,
                               output_hidden_states=self.layer_w is not None)

        return out.hidden_states, out.last_hidden_state, self.encoder._get_feat_extract_output_lengths(lengths)


class SSLModel(TorchSERModel):
    '''
    Модель на предобученном энкодере по сырой волне. Конкретная модель задаётся
    через net=dict(checkpoint=...): wav2vec 2.0, HuBERT, WavLM, XLS-R, MMS, mHuBERT
    '''
    def build_net(self):
        return SSLNet(self.n_classes, **self.net_kwargs)


Wav2VecModel = SSLModel
