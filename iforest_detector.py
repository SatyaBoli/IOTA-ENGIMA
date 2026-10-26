import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def make_features(values, window=50):
    """Turn a list of readings into a table of features.
    Only the current and PAST readings are used (no future data)."""
    s = pd.Series(values, dtype="float64")
    past = s.shift(1)
    roll_mean = past.rolling(window, min_periods=2).mean()
    roll_std = past.rolling(window, min_periods=2).std()
    feats = pd.DataFrame({
        "value": s,
        "prev": past,
        "roll_mean": roll_mean,
        "roll_std": roll_std,
        "diff": s - past,
        "dev": s - roll_mean,
        "z": (s - roll_mean) / (roll_std + 1e-9),
    })
    return feats.fillna(0.0)


class IsolationForestDetector:
    def __init__(self, window=50, n_estimators=100, random_state=42):
        self.window = window
        self.model = IsolationForest(
            n_estimators=n_estimators, random_state=random_state
        )

    def fit(self, values):
        self.model.fit(make_features(values, self.window))
        return self

    def score(self, values):
        X = make_features(values, self.window)
        return -self.model.decision_function(X)  # higher = more anomalous

    def predict(self, values):
        X = make_features(values, self.window)
        return self.model.predict(X)  # -1 = anomalous, 1 = normal


class RollingZScoreDetector:
    def __init__(self, window=50, threshold=2.0):
        self.window = window
        self.threshold = threshold

    def score(self, values):
        s = pd.Series(values, dtype="float64")
        past = s.shift(1)
        mean = past.rolling(self.window, min_periods=2).mean()
        std = past.rolling(self.window, min_periods=2).std()
        scores = ((s - mean) / (std + 1e-9)).abs()
        return scores.fillna(0.0).to_numpy()

    def predict(self, values):
        scores = self.score(values)
        return np.where(scores > self.threshold, -1, 1)