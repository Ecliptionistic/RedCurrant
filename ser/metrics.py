import numpy as np
import sklearn.metrics as metrics


def confusion(y_true, probs, names):
    '''
    Матрица ошибок
    '''
    pred = np.asarray(probs).argmax(1)

    return metrics.confusion_matrix(np.asarray(y_true), pred, labels=list(range(len(names))))


def calibration_error(y, probs, bins=10):
    '''
    ECE: средний разрыв между уверенностью модели и фактической точностью по корзинам уверенности
    '''
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    ece = 0.0

    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())

    return float(ece)


def compute_metrics(y_true, probs, names):
    '''
    Расчёт метрик

    WA: Weighted Accuracy (обычная точность)
    UA: Unweighted Accuracy (средняя полнота по классам)
    F1: макро F1-мера
    f1_weighted: F1 с весами по размеру классов
    precision: макро точность
    kappa: каппа Коэна
    mcc: коэффициент корреляции Мэттьюса
    logloss: кросс-энтропия по вероятностям
    ece: ошибка калибровки уверенности
    recall_<класс>, f1_<класс>: метрики по каждому классу
    '''
    y, probs = np.asarray(y_true), np.asarray(probs, dtype=np.float64)
    pred, labels = probs.argmax(1), list(range(len(names)))
    p = np.clip(probs, 1e-12, 1.0)
    p = p / p.sum(1, keepdims=True)

    out = {
        "wa": float(metrics.accuracy_score(y, pred)),
        "ua": float(metrics.balanced_accuracy_score(y, pred)),
        "f1": float(metrics.f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(metrics.f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        "precision": float(metrics.precision_score(y, pred, labels=labels, average="macro", zero_division=0)),
        "kappa": float(metrics.cohen_kappa_score(y, pred, labels=labels)),
        "mcc": float(metrics.matthews_corrcoef(y, pred)),
        "logloss": float(metrics.log_loss(y, p, labels=labels)),
        "ece": calibration_error(y, p),
    }

    rec = metrics.recall_score(y, pred, labels=labels, average=None, zero_division=0)
    f1 = metrics.f1_score(y, pred, labels=labels, average=None, zero_division=0)

    for i, n in enumerate(names):
        out[f"recall_{n}"] = float(rec[i])
        out[f"f1_{n}"] = float(f1[i])

    return out
