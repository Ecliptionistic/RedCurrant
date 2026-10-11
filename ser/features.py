import hashlib
from pathlib import Path
import numpy as np
import torch
import torchaudio
from joblib import Parallel, delayed
from .audio import load_audio, SR


class MFCCStats:
    def __init__(self, n_mfcc=40, n_mels=80, n_fft=400, hop=160):
        self.cfg = dict(n_mfcc=n_mfcc, n_mels=n_mels, n_fft=n_fft, hop=hop)
        self.mfcc = torchaudio.transforms.MFCC(
            SR, n_mfcc=n_mfcc, melkwargs=dict(n_fft=n_fft, hop_length=hop, n_mels=n_mels))

    @property
    def key(self):
        return "mfccstats_" + hashlib.md5(repr(sorted(self.cfg.items())).encode()).hexdigest()[:8]

    @torch.no_grad()
    def __call__(self, wav):
        n_fft, hop = self.cfg["n_fft"], self.cfg["hop"]
        if wav.numel() < n_fft:
            wav = torch.nn.functional.pad(wav, (0, n_fft - wav.numel()))
        c = self.mfcc(wav)
        d1 = torchaudio.functional.compute_deltas(c)
        d2 = torchaudio.functional.compute_deltas(d1)
        f = torch.cat([c, d1, d2]).numpy()
        e = np.log(wav.unfold(0, n_fft, hop).pow(2).mean(1).numpy() + 1e-10)
        return np.concatenate([f.mean(1), f.std(1), f.min(1), f.max(1),
                               [e.mean(), e.std()]]).astype(np.float32)


def _extract(ext, path):
    torch.set_num_threads(1)
    return ext(load_audio(path))


def extract_cached(df, ext, cache_dir="cache", n_jobs=8):
    path = Path(cache_dir) / f"{ext.key}.npz"
    store = {}
    if path.exists():
        z = np.load(path)
        store = dict(zip(z["uids"].tolist(), z["X"]))
    todo = [(u, p) for u, p in zip(df["uid"], df["path"]) if u not in store]
    if todo:
        feats = Parallel(n_jobs=n_jobs)(delayed(_extract)(ext, p) for _, p in todo)
        store.update({u: f for (u, _), f in zip(todo, feats)})
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, uids=np.array(list(store)), X=np.stack(list(store.values())))
    return np.stack([store[u] for u in df["uid"]])
