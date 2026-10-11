import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = ["neutral", "happy", "sad", "angry"]

IEMOCAP_MAP = {
    "neu": "neutral", 
    "hap": "happy", 
    "exc": "happy", 
    "sad": "sad", 
    "ang": "angry"
    }

_IEMOCAP_LINE = re.compile(r"^\[([\d.]+)\s*-\s*([\d.]+)\]\s+(Ses\S+)\s+(\w+)")


def load_iemocap(root):
    root = Path(root)
    rows = []

    for s in range(1, 6):
        sess = root / f"Session{s}"
        for lab in sorted((sess / "dialog" / "EmoEvaluation").glob("*.txt")):
            wav_dir = sess / "sentences" / "wav" / lab.stem

            for line in lab.read_text(encoding="utf-8", errors="replace").splitlines():
                m = _IEMOCAP_LINE.match(line)

                if not m or m[4] not in IEMOCAP_MAP:
                    continue
                
                uid, path = m[3], wav_dir / f"{m[3]}.wav"

                if not path.exists():
                    continue
                
                gender = uid.split("_")[-1][0]

                rows.append(dict(
                    uid=uid, path=str(path), emotion=IEMOCAP_MAP[m[4]],
                    corpus="iemocap", language="en", speaker=f"iemocap:Ses{s:02d}{gender}",
                    gender=gender, session=s, type="impro" if "impro" in uid else "script"))
    
    df = pd.DataFrame(rows)
    df["label"] = df["emotion"].map(LABELS.index)

    return df


def split_val(df, frac=0.1, seed=42):
    return train_test_split(df, test_size=frac, stratify=df["label"], random_state=seed)


def loso_folds(df, col="session"):
    return [(f"{col}-{g}", df[df[col] != g], {"test": df[df[col] == g]}) for g in sorted(df[col].unique())]


def cross_fold(name, train_df, **tests):
    shared = set(train_df["speaker"]).intersection(*[set(t["speaker"]) for t in tests.values()][:1])

    if shared:
        print(f"Err")

    return name, train_df, tests
