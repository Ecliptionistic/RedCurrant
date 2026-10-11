import time
from pathlib import Path
import pandas as pd
import torch

from ser.data import LABELS, load_iemocap, loso_folds, cross_fold, split_val
from ser.metrics import compute_metrics
from ser.models import SVMModel, BiLSTMModel, Wav2VecModel
from ser.trainer import seed_everything

IEMOCAP_ROOT = "/home/ecliptionist/Documents/Datasets/IEMOCAP"
SEED = 42
OUT_DIR = Path("results")
N = len(LABELS)

MODELS = {
    "svm": lambda: SVMModel(N, C_grid=[0.1, 1.0, 10.0]),
    "bilstm": lambda: BiLSTMModel(
        N, batch_size=32, max_sec=8,
        train=dict(epochs=40, lr=1e-3, patience=8),
        net=dict(hidden=128, layers=2, dropout=0.3)),
    "wav2vec2": lambda: Wav2VecModel(
        N, batch_size=8, max_sec=8,
        train=dict(epochs=15, lr=1e-3, encoder_lr=2e-5, unfreeze_epoch=3, patience=4),
        net=dict(checkpoint="facebook/wav2vec2-base", layer_weights=True)),
}


def run(name, folds, model_names, seed=SEED):
    out = OUT_DIR / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    records = []
    for mname in model_names:
        for fold, train_df, tests in folds:
            print(f"\n=== {name} | {mname} | {fold} ===")
            seed_everything(seed)
            tr, va = split_val(train_df, seed=seed)
            model = MODELS[mname]()
            model.fit(tr, va)
            for tname, tdf in tests.items():
                m = compute_metrics(tdf["label"].values, model.predict_proba(tdf), LABELS)
                records.append({"model": mname, "fold": fold, "test": tname,
                                "n_train": len(tr), "n_test": len(tdf), **m})
                print(f"  [{tname}] WA={m['wa']:.4f}  UA={m['ua']:.4f}  F1={m['f1']:.4f}")
            del model
            torch.cuda.empty_cache()

    res = pd.DataFrame(records)
    res.to_csv(out / "results.csv", index=False)
    summary = res.groupby(["model", "test"])[["wa", "ua", "f1"]].agg(["mean", "std"]).round(4)
    summary.to_csv(out / "summary.csv")
    print(f"\n{summary}")
    return res


if __name__ == "__main__":
    iemocap = load_iemocap(IEMOCAP_ROOT)
    print(iemocap.groupby(["session", "emotion"]).size().unstack(), f"\nВсего: {len(iemocap)}")

    run("iemocap_loso", loso_folds(iemocap), ["svm", "bilstm", "wav2vec2"])