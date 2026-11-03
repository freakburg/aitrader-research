"""ML strategy: predict future direction, retrained every `retrain_every` bars.

Predictions are produced walk-forward inside `proba`, so a bar's prediction only comes from a model fit on
labels that were already known at that bar. `proba` is separate from `signals` so a sweep can compute the
expensive part once and try many thresholds.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .strategy import Strategy


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    r = c.pct_change()
    X = pd.DataFrame({f"ret_{k}": c.pct_change(k) for k in (1, 2, 5, 10, 20)})
    X["vol_10"] = r.rolling(10).std()
    X["vol_20"] = r.rolling(20).std()
    X["sma_ratio"] = c.rolling(10).mean() / c.rolling(50).mean() - 1
    gain = r.clip(lower=0).rolling(14).mean()
    loss = (-r.clip(upper=0)).rolling(14).mean()
    X["rsi"] = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return X


class MLStrategy(Strategy):
    name = "ml"

    def __init__(self, min_train: int = 500, retrain_every: int = 21, threshold: float = 0.55,
                 model: str = "logreg", C: float = 0.1, depth: int = 4, horizon: int = 1,
                 window: str = "expanding"):
        """horizon: predict whether price is higher `horizon` bars ahead. window: "expanding" trains on all
        past data, "rolling" on the last `min_train` bars only. C is for logreg, depth for rf / hgb."""
        self.min_train, self.retrain_every, self.threshold, self.model = min_train, retrain_every, threshold, model
        self.C, self.depth, self.horizon, self.window = C, depth, horizon, window
        self.warmup = min_train
        self.params = {"min_train": min_train, "retrain_every": retrain_every, "threshold": threshold,
                       "model": model, "C": C, "depth": depth, "horizon": horizon, "window": window}

    def _new_model(self):
        if self.model == "rf":
            return RandomForestClassifier(n_estimators=100, max_depth=self.depth, min_samples_leaf=20,
                                          random_state=0, n_jobs=1)
        if self.model == "hgb":
            return HistGradientBoostingClassifier(max_depth=self.depth, learning_rate=0.05, max_iter=100,
                                                  min_samples_leaf=20, random_state=0)
        return make_pipeline(StandardScaler(), LogisticRegression(C=self.C, max_iter=500))

    def proba(self, df: pd.DataFrame) -> pd.Series:
        """P(up) per bar from a model trained only on labels known at that bar; NaN before the first model."""
        X = make_features(df)
        h = self.horizon
        y = (df["close"].shift(-h) > df["close"]).astype(int)
        p = pd.Series(np.nan, index=df.index)
        n, t = len(df), self.min_train
        while t < n:
            end = min(t + self.retrain_every, n)
            # Rows < t-h have labels that use closes <= t-1, all known when predicting from bar t on.
            hi = t - h
            lo = 0 if self.window == "expanding" else max(0, hi - self.min_train)
            Xtr, ytr = X.iloc[lo:hi], y.iloc[lo:hi]
            ok = Xtr.notna().all(axis=1)
            if ok.sum() >= 100 and ytr[ok].nunique() == 2:
                model = self._new_model().fit(Xtr[ok], ytr[ok])
                Xte = X.iloc[t:end]
                good = Xte.notna().all(axis=1)
                if good.any():
                    p.loc[Xte.index[good]] = model.predict_proba(Xte[good])[:, 1]
            t = end
        return p

    def signals(self, df: pd.DataFrame) -> pd.Series:
        return (self.proba(df) > self.threshold).astype(float)
