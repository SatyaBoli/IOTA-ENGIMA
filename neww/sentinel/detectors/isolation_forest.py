"""Candidate Detector B2: Causal Windowed Isolation Forest.

Leverages tree-based path length anomaly detection over causal rolling lag,
difference, and distributional features.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from ..contracts import Point, RawDetectorOutput
from ..features.causal_features import CausalFeatureExtractor
from .base import BaseDetector


class StreamingIsolationForestDetector(BaseDetector):
    """Causal sliding-window Isolation Forest detector."""

    def __init__(
        self,
        window_size: int = 100,
        warmup_steps: int = 40,
        n_estimators: int = 30,
        contamination: float = 0.02,
        retrain_interval: int = 250,
        random_state: int = 42,
        version: str = "iforest-0.1.0",
    ) -> None:
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.retrain_interval = retrain_interval
        self.random_state = random_state
        self._version = version

        self._feature_extractor = CausalFeatureExtractor(window_size=24, warmup_steps=15)
        self._histories: dict[str, deque[np.ndarray]] = {}
        self._models: dict[str, IsolationForest] = {}
        self._steps_since_train: dict[str, int] = {}

    @property
    def version(self) -> str:
        return self._version

    def _get_history(self, series_id: str) -> deque[np.ndarray]:
        hist = self._histories.get(series_id)
        if hist is None:
            hist = self._histories[series_id] = deque(maxlen=self.window_size)
        return hist

    def _train_model(self, series_id: str) -> None:
        hist = self._get_history(series_id)
        if len(hist) < self.warmup_steps:
            return
        X = np.vstack(list(hist))
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        model.fit(X)
        self._models[series_id] = model
        self._steps_since_train[series_id] = 0

    def score_one(self, point: Point) -> RawDetectorOutput:
        feat = self._feature_extractor.extract(point)
        x_vec = feat.to_array().reshape(1, -1)
        hist = self._get_history(point.series_id)

        model = self._models.get(point.series_id)
        if model is None or len(hist) < self.warmup_steps:
            # Baseline warm-up fallback using raw z-score
            raw_score = abs(feat.z_score)
            return RawDetectorOutput(
                raw_score=float(raw_score),
                observed=float(point.value),
                baseline=float(feat.rolling_mean),
                residual=float(point.value - feat.rolling_mean),
                context={"is_warmup": True, "model": "fallback_z", "n_samples": len(hist)},
            )

        # In scikit-learn, score_samples is negative anomaly score (lower is more anomalous)
        # So raw_score = -score_samples maps higher => more anomalous
        raw_anomaly = -float(model.score_samples(x_vec)[0])
        # Scale into intuitive 0-10+ score range
        # Average score for normal is around 0.4 - 0.5; offset by 0.5 and scale by 10
        norm_score = max((raw_anomaly - 0.45) * 12.0, 0.0)

        diff = point.value - feat.rolling_mean

        return RawDetectorOutput(
            raw_score=float(norm_score),
            observed=float(point.value),
            baseline=float(feat.rolling_mean),
            residual=float(diff),
            context={
                "iforest_raw": raw_anomaly,
                "is_warmup": False,
                "n_samples": len(hist),
                "contributors": ["tree_path_length", "residual_magnitude"],
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        feat = self._feature_extractor.extract(point)
        self._feature_extractor.update(point)

        hist = self._get_history(point.series_id)
        # Guarded update: do not feed anomalous vectors into training buffer
        if not is_anomalous or len(hist) < self.warmup_steps:
            hist.append(feat.to_array())

        steps = self._steps_since_train.get(point.series_id, 0) + 1
        self._steps_since_train[point.series_id] = steps

        if (point.series_id not in self._models and len(hist) >= self.warmup_steps) or (
            steps >= self.retrain_interval and len(hist) >= self.warmup_steps
        ):
            self._train_model(point.series_id)

    def reset_state(self, series_id: Optional[str] = None) -> None:
        self._feature_extractor.reset(series_id)
        if series_id:
            self._histories.pop(series_id, None)
            self._models.pop(series_id, None)
            self._steps_since_train.pop(series_id, None)
        else:
            self._histories.clear()
            self._models.clear()
            self._steps_since_train.clear()
