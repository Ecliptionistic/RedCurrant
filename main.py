import gc
import os
import time
import traceback
from pathlib import Path
import pandas as pd
import torch

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from ser.data import LABELS, load_corpora, select, loso_folds, loco_folds, matrix_folds, custom_folds, split_val
from ser.metrics import compute_metrics, confusion
from ser.models import SVMModel, BiLSTMModel, CNNLSTMModel, SSLModel, WhisperModel
from ser.stats import RunLog
from ser.trainer import seed_everything

DATA_DIR = Path(os.environ.get("SER_DATA", "/home/ecliptionist/Documents/Datasets"))
OUT_DIR = Path("results")
N = len(LABELS)

CORPORA = {
    "iemocap": {"root": DATA_DIR / "IEMOCAP"},
    "cremad": {"root": DATA_DIR / "CREMA-D"},
    "ravdess": {"root": DATA_DIR / "RAVDESS"},
    "emodb": {"root": DATA_DIR / "EmoDB"},
    "savee": {"root": DATA_DIR / "SAVEE"},
    "tess": {"root": DATA_DIR / "TESS"},
    "emovo": {"root": DATA_DIR / "EMOVO"},
    #"dusha": {"root": DATA_DIR / "Dusha", "max_samples": 8000},
    #"dusha_podcast": {"loader": "dusha", "subset": "podcast", "root": DATA_DIR / "Dusha"},
    "nemo": {"root": DATA_DIR / "nEMO"},                      
    "resd": {"root": DATA_DIR / "RESD"},                      
}


def pretrained(cls, batch_size=8, **net):
    '''
    Предобученная модель 
    probe = True - замораживает энкодер, 
    иначе энкодер размораживается на unfreeze_epoch
    '''
    def build(probe=False):
        train = dict(epochs=15, lr=1e-3, encoder_lr=2e-5, unfreeze_epoch=None if probe else 3, patience=4)
        return cls(N, batch_size=batch_size, max_sec=8, train=train,
                   net=dict(layer_weights=True, **net))
    return build


MODELS = {
    "svm": lambda probe=False: SVMModel(N, C_grid=[0.1, 1.0, 10.0]),
    "bilstm": lambda probe=False: BiLSTMModel(
        N, batch_size=32, max_sec=8,
        train=dict(epochs=40, lr=1e-3, patience=8),
        net=dict(hidden=128, layers=2, dropout=0.3)),
    "cnn_lstm": lambda probe=False: CNNLSTMModel(
        N, batch_size=32, max_sec=8,
        train=dict(epochs=40, lr=1e-3, patience=8),
        net=dict(channels=(32, 64), hidden=128, layers=2, dropout=0.3)),
    "wav2vec2": pretrained(SSLModel, checkpoint="facebook/wav2vec2-base"),
    "hubert": pretrained(SSLModel, checkpoint="facebook/hubert-base-ls960"),
    "wavlm": pretrained(SSLModel, checkpoint="microsoft/wavlm-base-plus"),
    "wavlm_large": pretrained(SSLModel, batch_size=4, checkpoint="microsoft/wavlm-large"),
    "xlsr": pretrained(SSLModel, batch_size=4, checkpoint="facebook/wav2vec2-xls-r-300m"),
    "mms": pretrained(SSLModel, batch_size=4, checkpoint="facebook/mms-300m"),
    "mhubert": pretrained(SSLModel, checkpoint="utter-project/mHuBERT-147"),
    "whisper": pretrained(WhisperModel, checkpoint="openai/whisper-small", window_sec=8.0),
    "whisper_large": pretrained(WhisperModel, batch_size=2, checkpoint="openai/whisper-large-v3", window_sec=8.0),
}

RUN_MODELS = [
    "svm",
    "bilstm",
    "cnn_lstm",
    "wav2vec2",
    # "hubert",
    # "wavlm",
    # "wavlm_large",
    # "xlsr",
    # "mms",
    # "mhubert",
    # "whisper",
    # "whisper_large",
]

EXPERIMENTS = [
    # весь IEMOCAP на обучение, проверка на других корпусах (польский, русский, немецкий, итальянский), все модели
    dict(name="iemocap2cross", train=["iemocap"], test=["nemo", "resd", "emodb", "emovo"], models=[
        "svm", "bilstm", "cnn_lstm", "wav2vec2", "hubert", "wavlm", "mhubert", "whisper",
        # "wavlm_large", "xlsr", "mms", "whisper_large",
    ]),
    # dict(name="уь2cross", train=["iemocap"], test=["nemo", "resd", "emodb", "emovo"], models=[
            # "svm", "bilstm", "cnn_lstm", "wav2vec2", "hubert", "wavlm", "mhubert", "whisper",
            # "wavlm_large", "xlsr", "mms", "whisper_large",
        # ]),
    # dict(name="en2other", train=["iemocap", "cremad"], test=["emodb", "emovo"]),
    # dict(name="en_ru", train=["cremad", "dusha"], test=["cremad", "dusha", "emodb", "emovo"]),
]

SEEDS = [42]      
SPLIT_SEED = 42
PROBE = False
CAP = None
FAST = False
NOTE = ""

def corpora_of(exp):
    if exp.get("protocol") == "loso":
        return ["iemocap"]
    return exp.get("corpora") or list(dict.fromkeys(exp["train"] + exp["test"]))


def build_folds(exp, df, cap):
    protocol = exp.get("protocol", "custom")
    if protocol == "custom":
        return custom_folds(df, exp["train"], exp["test"], seed=SPLIT_SEED, cap=cap), "speaker"
    if protocol == "loso":
        return loso_folds(df, exp.get("by", "session")), None
    if protocol == "matrix":
        return matrix_folds(df, seed=SPLIT_SEED, cap_all=cap), "speaker"
    if protocol == "loco":
        return loco_folds(df, seed=SPLIT_SEED, cap=cap), "speaker"
    raise KeyError(protocol)


def timed(fn, *args):
    '''
    Возвращает результат измерения времени в секундах
    '''
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
    t = time.perf_counter()
    res = fn(*args)
    if cuda:
        torch.cuda.synchronize()
    return res, time.perf_counter() - t


def run(exp, df):
    models, probe, cap = exp.get("models", RUN_MODELS), exp.get("probe", PROBE), exp.get("cap", CAP)
    if exp.get("where"):
        df = select(df, **exp["where"])
    folds, val_group = build_folds(exp, df, cap)
    config = {"protocol": "custom", **exp, "models": models, "probe": probe, "cap": cap,
              "seeds": SEEDS, "split_seed": SPLIT_SEED, "fast": FAST, "note": NOTE}

    failed = []

    with RunLog(OUT_DIR, config) as log:
        for mname in models:
            for seed in SEEDS:
                for fold, train_df, tests in folds:
                    print(f"\n=== {exp['name']} | {mname} | seed {seed} | {fold} ===")
                    model = None
                    try:
                        seed_everything(seed)
                        tr, va = split_val(train_df, seed=seed, group=val_group)
                        model = MODELS[mname](probe)
                        if FAST:
                            fast_dev(model)
                        # fit включает загрузку весов и, для SVM, извлечение признаков (при повторном запуске они берутся из кэша)
                        _, fit_sec = timed(model.fit, tr, va)
                        print(f"  обучение: {fit_sec:.1f} с")
                        for tname, tdf in tests.items():
                            probs, predict_sec = timed(model.predict_proba, tdf)
                            m = compute_metrics(tdf["label"].values, probs, LABELS)
                            cm = pd.DataFrame(confusion(tdf["label"].values, probs, LABELS), index=LABELS, columns=LABELS)
                            log.add({"model": mname, "seed": seed, "fold": fold,
                                     "train": "+".join(sorted(train_df["corpus"].unique())), "test": tname,
                                     "n_train": len(tr), "n_test": len(tdf), "fit_sec": fit_sec,
                                     "predict_sec": predict_sec, "ms_per_utt": 1000 * predict_sec / len(tdf), **m}, cm)
                            print(f"  [{tname}] WA={m['wa']:.4f}  UA={m['ua']:.4f}  F1={m['f1']:.4f}  MCC={m['mcc']:.4f}  "
                                  f"predict={predict_sec:.1f} с ({1000 * predict_sec / len(tdf):.1f} мс/запись)")
                    except Exception as e:
                        traceback.print_exc()
                        failed.append(f"{mname} | seed {seed} | {fold}: {type(e).__name__}")
                        print(f"  [!] {mname} пропущена ({type(e).__name__}), эксперимент продолжается")
                    model = None
                    gc.collect()
                    torch.cuda.empty_cache()

    if failed:
        print(f"\n[!] {exp['name']}: не отработали {len(failed)}:\n  " + "\n  ".join(failed))


def fast_dev(model):
    '''
    Быстрая проверка пайплайна: 2 эпохи (для SVM один C)
    '''
    if hasattr(model, "C_grid"):
        model.C_grid = [1.0]
    if hasattr(model, "trainer"):
        model.trainer.epochs, model.trainer.patience = 2, 1
        if model.trainer.unfreeze_epoch:
            model.trainer.unfreeze_epoch = 2


def check():
    errors = []
    for exp in EXPERIMENTS:
        for c in corpora_of(exp):
            if c not in CORPORA:
                errors.append(f"[{exp['name']}] неизвестный корпус {c!r}, есть: {list(CORPORA)}")
            elif not Path(CORPORA[c]["root"]).exists() and CORPORA[c].get("loader", c) != "resd":  # resd скачивается сам
                errors.append(f"[{exp['name']}] нет папки {CORPORA[c]['root']}")
        for m in exp.get("models", RUN_MODELS):
            if m not in MODELS:
                errors.append(f"[{exp['name']}] неизвестная модель {m!r}, есть: {list(MODELS)}")
    if errors:
        raise SystemExit("\n".join(errors))


def main():
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    check()
    print(f"Данные: {DATA_DIR}")
    for e in EXPERIMENTS:
        print(f"  {e['name']}: {e.get('models', RUN_MODELS)}{' (probe)' if e.get('probe', PROBE) else ''}")

    loaded = {}
    for exp in EXPERIMENTS:
        names = corpora_of(exp)
        for c in names:
            if c not in loaded:
                loaded[c] = load_corpora({c: CORPORA[c]})
        run(exp, pd.concat([loaded[c] for c in names], ignore_index=True))


if __name__ == "__main__":
    main()
