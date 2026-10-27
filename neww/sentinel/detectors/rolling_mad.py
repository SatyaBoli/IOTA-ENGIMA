"""Seasonal-Adaptive Causal Rolling Median / MAD Detector.

Implements dual-horizon causal anomaly scoring:
1. Dynamic volatility floor calibrated to initial warm-up dispersion (prevents micro-noise explosions).
2. Diurnal seasonal differencing (period=288) to neutralize scheduled recurring spikes.
3. Modified z-score scoring with robust outlier gating.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Optional

import numpy as np

from ..contracts import Point, RawDetectorOutput
from .base import BaseDetector


class RollingMedianMADDetector(BaseDetector):
    """Causal seasonal-aware rolling median and MAD anomaly detector."""

    def __init__(
        self,
        window_size: int = 50,
        seasonal_period: int = 288,
        warmup_steps: int = 60,
        scale_floor: float = 0.05,
        version: str = "seasonal_mad-0.2.0",
    ) -> None:
        self.window_size = window_size
        self.seasonal_period = seasonal_period
        self.warmup_steps = warmup_steps
        self.scale_floor = scale_floor
        self._version = version

        self._local_buffers: dict[str, deque[float]] = {}
        self._seasonal_buffers: dict[str, deque[float]] = {}
        self._warmup_stds: dict[str, float] = {}

    @property
    def version(self) -> str:
        return self._version

    def _get_local(self, series_id: str) -> deque[float]:
        buf = self._local_buffers.get(series_id)
        if buf is None:
            buf = self._local_buffers[series_id] = deque(maxlen=self.window_size)
        return buf

    def _get_seasonal(self, series_id: str) -> deque[float]:
        buf = self._seasonal_buffers.get(series_id)
        if buf is None:
            buf = self._seasonal_buffers[series_id] = deque(maxlen=self.seasonal_period + 10)
        return buf

    def score_one(self, point: Point) -> RawDetectorOutput:
        loc_buf = self._get_local(point.series_id)
        seas_buf = self._get_seasonal(point.series_id)
        n = len(loc_buf)

        if n < 5:
            return RawDetectorOutput(
                raw_score=0.0,
                observed=point.value,
                baseline=point.value,
                residual=0.0,
                context={"is_warmup": True, "window_size": n},
            )

        # Dynamic scale floor calibrated from initial dispersion
        base_std = self._warmup_stds.get(point.series_id, 1.0)
        eff_floor = max(self.scale_floor, 0.18 * base_std)

        # Local median & MAD
        vals = np.array(loc_buf, dtype=np.float64)
        med = float(np.median(vals))
        abs_diff = np.abs(vals - med)
        mad = float(max(np.median(abs_diff), eff_floor))
        loc_score = 0.6745 * abs(point.value - med) / mad

        # Seasonal differencing if full period is available
        if len(seas_buf) >= self.seasonal_period:
            seas_val = list(seas_buf)[-self.seasonal_period]
            seas_diff = point.value - seas_val
            # Rolling window of seasonal differences
            past_diffs = [
                list(seas_buf)[-k] - list(seas_buf)[-k - self.seasonal_period]
                for k in range(1, min(len(seas_buf) - self.seasonal_period, self.window_size))
            ]
            if len(past_diffs) >= 5:
                s_arr = np.array(past_diffs, dtype=np.float64)
                s_med = float(np.median(s_arr))
                s_mad = float(max(np.median(np.abs(s_arr - s_med)), eff_floor))
                seas_score = 0.6745 * abs(seas_diff - s_med) / s_mad
                # Combine local and seasonal
                final_score = min(loc_score, seas_score) if loc_score > 3.0 else loc_score
                baseline = med + s_med
            else:
                final_score = loc_score
                baseline = med
        else:
            final_score = loc_score
            baseline = med

        if not math.isfinite(final_score):
            final_score = 1e6

        is_warmup = n < self.warmup_steps

        return RawDetectorOutput(
            raw_score=float(final_score),
            observed=float(point.value),
            baseline=float(baseline),
            residual=float(point.value - baseline),
            context={
                "local_median": med,
                "local_mad": mad,
                "scale_floor": eff_floor,
                "is_warmup": is_warmup,
                "window_size": n,
                "contributors": ["seasonal_deviation", "local_mad_deviation"],
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        loc_buf = self._get_local(point.series_id)
        seas_buf = self._get_seasonal(point.series_id)

        # Record warm-up variance
        if point.series_id not in self._warmup_stds and len(loc_buf) >= 30:
            self._warmup_stds[point.series_id] = float(np.std(list(loc_buf)))

        # Update seasonal buffer with all points (for period alignment)
        if math.isfinite(point.value):
            seas_buf.append(point.value)

        # Guarded update: do not push into local baseline if anomalous
        if not is_anomalous or len(loc_buf) < self.warmup_steps:
            if math.isfinite(point.value):
                loc_buf.append(point.value)

    def reset_state(self, series_id: Optional[str] = None) -> None:
        if series_id:
            self._local_buffers.pop(series_id, None)
            self._seasonal_buffers.pop(series_id, None)
            self._warmup_stds.pop(series_id, None)
        else:
            self._local_buffers.clear()
            self._seasonal_buffers.clear()
            self._warmup_stds.clear()
