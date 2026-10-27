"""Causal feature engineering engine for streaming time series.

Guarantees 100% causal transforms: every feature vector at time t is constructed
exclusively from past observations [t - window_size + 1, ..., t].
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..contracts import Point


@dataclass
class CausalFeatures:
    """Structured feature vector for time t."""

    value: float
    lag_1: float
    lag_2: float
    lag_3: float
    diff_1: float
    diff_2: float
    rolling_mean: float
    rolling_std: float
    rolling_median: float
    rolling_mad: float
    z_score: float
    hour_sin: float
    hour_cos: float
    is_warmup: bool

    def to_array(self) -> np.ndarray:
        return np.array(
            [
                self.value,
                self.lag_1,
                self.lag_2,
                self.lag_3,
                self.diff_1,
                self.diff_2,
                self.rolling_mean,
                self.rolling_std,
                self.rolling_median,
                self.rolling_mad,
                self.z_score,
                self.hour_sin,
                self.hour_cos,
            ],
            dtype=np.float32,
        )


class CausalFeatureExtractor:
    """Maintains past-only sliding windows and extracts multi-scale features."""

    def __init__(self, window_size: int = 24, warmup_steps: int = 15) -> None:
        self.window_size = window_size
        self.warmup_steps = warmup_steps
        self._buffers: dict[str, deque[float]] = {}

    def _get_buffer(self, series_id: str) -> deque[float]:
        buf = self._buffers.get(series_id)
        if buf is None:
            buf = self._buffers[series_id] = deque(maxlen=self.window_size)
        return buf

    def extract(self, point: Point) -> CausalFeatures:
        """Extract features using past window AND current point."""
        buf = self._get_buffer(point.series_id)
        n = len(buf)
        v = float(point.value)

        # Parse cyclical timestamp features
        try:
            ts = pd.to_datetime(point.timestamp)
            hour = ts.hour + ts.minute / 60.0
            h_sin = math.sin(2.0 * math.pi * hour / 24.0)
            h_cos = math.cos(2.0 * math.pi * hour / 24.0)
        except Exception:
            h_sin, h_cos = 0.0, 1.0

        if n < 3:
            # Cold start fallback
            return CausalFeatures(
                value=v,
                lag_1=v,
                lag_2=v,
                lag_3=v,
                diff_1=0.0,
                diff_2=0.0,
                rolling_mean=v,
                rolling_std=1.0,
                rolling_median=v,
                rolling_mad=1.0,
                z_score=0.0,
                hour_sin=h_sin,
                hour_cos=h_cos,
                is_warmup=True,
            )

        hist = list(buf)
        lag1 = hist[-1]
        lag2 = hist[-2] if n >= 2 else lag1
        lag3 = hist[-3] if n >= 3 else lag2

        diff1 = v - lag1
        diff2 = v - lag2

        arr = np.array(hist, dtype=np.float64)
        mean = float(np.mean(arr))
        std = float(max(np.std(arr), 1e-4))
        med = float(np.median(arr))
        mad = float(max(np.median(np.abs(arr - med)), 1e-4))
        z = (v - mean) / std

        is_warmup = n < self.warmup_steps

        return CausalFeatures(
            value=v,
            lag_1=lag1,
            lag_2=lag2,
            lag_3=lag3,
            diff_1=diff1,
            diff_2=diff2,
            rolling_mean=mean,
            rolling_std=std,
            rolling_median=med,
            rolling_mad=mad,
            z_score=float(z),
            hour_sin=h_sin,
            hour_cos=h_cos,
            is_warmup=is_warmup,
        )

    def update(self, point: Point) -> None:
        """Push point into history buffer after feature extraction and scoring."""
        if math.isfinite(point.value):
            buf = self._get_buffer(point.series_id)
            buf.append(point.value)

    def reset(self, series_id: Optional[str] = None) -> None:
        if series_id:
            if series_id in self._buffers:
                self._buffers[series_id].clear()
        else:
            self._buffers.clear()
