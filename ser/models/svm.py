import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from .base import SERModel
from ..features import MFCCStats, extract_cached


class SVMModel(SERModel):
    def __init__(self, n_classes, C_grid=(0.1, 1.0, 10.0), gamma="scale",
                 cache_dir="cache", n_jobs=8, feature=None):
        super().__init__(n_classes)
        self.C_grid, self.gamma, self.cache_dir, self.n_jobs = C_grid, gamma, cache_dir, n_jobs
        self.ext = MFCCStats(**(feature or {}))

    def _X(self, df):
        return extract_cached(df, self.ext, self.cache_dir, self.n_jobs)

    def fit(self, train_df, val_df):
        Xtr, Xva = self._X(train_df), self._X(val_df)
        best = (-1.0, None)

        for C in self.C_grid:
            pipe = make_pipeline(StandardScaler(), 
                                 SVC(C=C, 
                                    gamma=self.gamma, 
                                    class_weight="balanced",
                                    probability=True, 
                                    random_state=0))
            pipe.fit(Xtr, train_df["label"].values)
            ua = balanced_accuracy_score(val_df["label"].values, pipe.predict(Xva))

            print(f"C = {C:<6}")
            print(f"Unweighted Accuracy = {ua:.4f}")

            if ua > best[0]:
                best = (ua, pipe)
        self.pipe = best[1]

    def predict_proba(self, df):
        out = np.zeros((len(df), self.n_classes))
        out[:, self.pipe.classes_] = self.pipe.predict_proba(self._X(df))
        return out
