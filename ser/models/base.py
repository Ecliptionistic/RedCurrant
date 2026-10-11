from abc import ABC, abstractmethod
import numpy as np
import torch.nn as nn
from ..audio import make_loader
from ..trainer import Trainer


class SERModel(ABC):
    '''
    
    '''
    def __init__(self, n_classes):
        self.n_classes = n_classes

    @abstractmethod
    def fit(self, train_df, val_df):
        pass

    @abstractmethod
    def predict_proba(self, df) -> np.ndarray:
        pass


class TorchSERModel(SERModel):
    def __init__(self, n_classes, batch_size=16, max_sec=8.0, eval_max_sec=20.0,
                 num_workers=4, train=None, net=None):
        super().__init__(n_classes)
        self.bs, self.max_sec, self.eval_max_sec, self.nw = batch_size, max_sec, eval_max_sec, num_workers
        self.trainer = Trainer(**(train or {}))
        self.net_kwargs = net or {}

    @abstractmethod
    def build_net(self) -> nn.Module: ...

    def _loader(self, df, train):
        return make_loader(df, self.bs, train, self.max_sec if train else self.eval_max_sec, self.nw)

    def fit(self, train_df, val_df):
        self.net = self.build_net()
        self.trainer.fit(self.net, self._loader(train_df, True), self._loader(val_df, False),
                         train_df["label"].values, self.n_classes)

    def predict_proba(self, df):
        return self.trainer.predict(self.net, self._loader(df, False))[0]
