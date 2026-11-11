import json
import sys
import time
from pathlib import Path
import pandas as pd

from .data import LABELS

JOURNAL = "journal.csv"

SUMMARY_COLS = ["wa", "ua", "f1", "f1_weighted", "precision", 
                "kappa", "mcc", "logloss", "ece",
                "fit_sec", "predict_sec", "ms_per_utt"]

PRINT_COLS = ["wa", "ua", "f1", "mcc", "ms_per_utt"]

KEYS = ["model", "train", "test"]


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def mean_std(x):
    return f"{x.mean():.3f} ± {x.std():.3f}" if len(x) > 1 else f"{x.mean():.3f}"


class RunLog:
    '''
    Результаты одного эксперимента в папке results/<имя>_<время>/:
      config.json - настройки, log.txt - копия всего вывода, results.csv - все строки,
      summary.csv, per_class.csv, ua_matrix.csv - итоги, confusion/ - матрицы ошибок.
    '''
    def __init__(self, out_dir, config):
        self.root = Path(out_dir)
        self.config = config
        self.run_id = f"{config['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.dir = self.root / self.run_id
        (self.dir / "confusion").mkdir(parents=True)
        (self.dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
        self.records = []

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.file = open(self.dir / "log.txt", "w", encoding="utf-8")
        self.stdout = sys.stdout
        sys.stdout = Tee(self.stdout, self.file)
        print(f"\nЭксперимент {self.run_id}\n{json.dumps(self.config, ensure_ascii=False, default=str)}")
        return self

    def __exit__(self, *exc):
        try:
            if self.records:
                self.summarize()
        finally:
            sys.stdout = self.stdout
            self.file.close()

    def add(self, rec, cm=None):
        c = self.config
        rec = {"run": self.run_id, "date": time.strftime("%Y-%m-%d %H:%M"), "experiment": c["name"],
               "protocol": c["protocol"], "probe": c["probe"], "cap": c["cap"], "note": c["note"], **rec}
        self.records.append(rec)

        if cm is not None:
            cm.to_csv(self.dir / "confusion" / f"{rec['model']}__{rec['fold']}__{rec['test']}__s{rec['seed']}.csv")

        pd.DataFrame(self.records).to_csv(self.dir / "results.csv", index=False)

        if not c["fast"]:
            append_journal(self.root / JOURNAL, rec)

    def summarize(self):
        res, out = pd.DataFrame(self.records), self.dir
        g = res.groupby(KEYS)

        g[SUMMARY_COLS].agg(["mean", "std"]).round(4).to_csv(out / "summary.csv")
        short = g[PRINT_COLS].agg(mean_std)
        short["n"] = g.size()
        print(f"\nИтоги (среднее и разброс по seed'ам и фолдам, n - число повторов):\n{short}")

        per_class = g[[f"{p}_{n}" for p in ("recall", "f1") for n in LABELS]].mean().round(3)
        per_class.to_csv(out / "per_class.csv")
        print(f"\nПолнота по классам:\n{per_class[[f'recall_{n}' for n in LABELS]]}")

        if res["test"].nunique() > 1:
            pivot = res.pivot_table(index=["model", "train"], columns="test", values="ua").round(3)
            pivot.to_csv(out / "ua_matrix.csv")
            print(f"\nUA (обучение x тест):\n{pivot}")

        fit = res.drop_duplicates(["model", "seed", "fold"])["fit_sec"].sum()
        print(f"\nВсего: {(time.perf_counter() - self.t0) / 60:.1f} мин, из них обучение {fit / 60:.1f} мин, "
              f"тест {res['predict_sec'].sum() / 60:.1f} мин. Результаты: {out}")


def append_journal(path, rec):
    row = pd.DataFrame([rec])
    if path.exists():
        row = pd.concat([pd.read_csv(path), row], ignore_index=True)
    row.to_csv(path, index=False)
