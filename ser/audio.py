import random
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

SR = 16000


def load_audio(path, sr=SR):
    wav, orig = sf.read(path, dtype="float32", always_2d=True)
    wav = torch.from_numpy(wav.mean(1))
    if orig != sr:
        wav = torchaudio.functional.resample(wav, orig, sr)
    return wav


class AudioDataset(Dataset):
    def __init__(self, df, max_sec=8.0, train=False):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.max_len = int(max_sec * SR)
        self.train = train

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        wav = load_audio(self.paths[i])

        if wav.numel() > self.max_len:
            if self.train:
                offset = random.randint(0, wav.numel() - self.max_len)
            else:
                offset = 0
            wav = wav[offset:offset + self.max_len]

        return wav, self.labels[i]


def collate(batch):
    wavs, labels = zip(*batch)
    lengths = torch.tensor([w.numel() for w in wavs])
    x = torch.zeros(len(wavs), int(lengths.max()))

    for i, w in enumerate(wavs):
        x[i, :w.numel()] = w

    return x, lengths, torch.tensor(labels)


def make_loader(df, batch_size, train, max_sec=8.0, num_workers=4):
    return DataLoader(AudioDataset(df, max_sec, train), 
                      batch_size=batch_size, 
                      shuffle=train,
                      num_workers=num_workers, 
                      collate_fn=collate,
                      pin_memory=torch.cuda.is_available(), 
                      persistent_workers=num_workers > 0)
