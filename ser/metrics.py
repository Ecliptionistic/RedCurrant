import numpy as np
import sklearn.metrics as metrics

def compute_metrics(y_true, probs, names):
    '''
    Расчёт метрик

    WA: Weighted Accuracy
    UA: Unweighted Accuracy
    F!: F1-мера
    '''
    y, probs = np.asarray(y_true), np.asarray(probs, dtype=np.float64)
    pred, labels = probs.argmax(1), list(range(len(names)))
    return {
        "wa": float(metrics.accuracy_score(y, pred)),
        "ua": float(metrics.balanced_accuracy_score(y, pred)),
        "f1": float(metrics.f1_score(y, pred, labels=labels, average="macro", zero_division=0))
    }
