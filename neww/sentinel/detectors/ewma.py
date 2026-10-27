"""Baseline Detector B1: Streaming Exponentially Weighted Moving Average (EWMA).

Computes causal exponential smoothing of baseline mean and variance:
  mean_t = alpha * x_{t-1} + (1 - alpha) * mean_{t-1}
  var_t  = alpha * (x_{t-1} - mean_{t-1})^2 + (1 - alpha) * var_{t-1}
  z_t    = |x_t - mean_t| / max(sqrt(var_t), scale_floor)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..contracts import Point, RawDetectorOutput
from .base import BaseDetector


@dataclass
class EWMAState:
    mean: float = 0.0
    var: float = 1.0
    n_seen: int = 0


class EWMADetector(BaseDetector):
    """Causal streaming EWMA anomaly detector with online variance tracking."""

    def __init__(
        self,
        alpha: float = 0.15,
        warmup_steps: int = 30,
        scale_floor: float = 1e-4,
        version: str = "ewma-0.1.0",
    ) -> None:
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.scale_floor = scale_floor
        self._version = version
        self._states: dict[str, EWMAState] = {}

    @property
    def version(self) -> str:
        return self._version

    def _get_state(self, series_id: str) -> EWMAState:
        st = self._states.get(series_id)
        if st is None:
            st = self._states[series_id] = EWMAState()
        return st

    def score_one(self, point: Point) -> RawDetectorOutput:
        st = self._get_state(point.series_id)

        if st.n_seen < 3:
            return RawDetectorOutput(
                raw_score=0.0,
                observed=point.value,
                baseline=point.value,
                residual=0.0,
                context={"mean": point.value, "std": 0.0, "is_warmup": True, "n_seen": st.n_seen},
            )

        std = math.sqrt(max(st.var, 0.0))
        std_eff = max(std, self.scale_floor)
        diff = point.value - st.mean
        score = abs(diff) / std_eff

        if not math.isfinite(score):
            score = 1e6

        is_warmup = st.n_seen < self.warmup_steps

        return RawDetectorOutput(
            raw_score=float(score),
            observed=float(point.value),
            baseline=float(st.mean),
            residual=float(diff),
            context={
                "mean": float(st.mean),
                "std": float(std),
                "std_effective": float(std_eff),
                "is_warmup": is_warmup,
                "n_seen": st.n_seen,
                "contributors": ["ewma_residual"],
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        st = self._get_state(point.series_id)
        if not math.isfinite(point.value):
            return

        if st.n_seen == 0:
            st.mean = point.value
            st.var = 1.0
            st.n_seen = 1
            return

        # Warmup or accepted normal point updates the baseline
        if not is_anomalous or st.n_seen < self.warmup_steps:
            diff = point.value - st.mean
            # Update variance then mean
            st.var = (1.0 - self.alpha) * st.var + self.alpha * (diff * diff)
            st.mean = (1.0 - self.alpha) * st.mean + self.alpha * point.value
            st.n_seen += 1

    def reset_state(self, series_id: Optional[str] = None) -> None:
        if series_id:
            if series_id in self._states:
                del self._states[series_id]
        else:
            self._states.clear()
