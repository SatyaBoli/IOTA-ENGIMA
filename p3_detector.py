"""Online causal robust-MAD detector implementing the P3 detector protocol."""

from __future__ import annotations

import math
import statistics
from collections import deque

from p3.sentinel.contracts import Point, RawDetectorOutput

MAD_SCALE_FLOOR = 1e-3
ROBUST_Z_FACTOR = 0.6745


class CausalMadDetector:
    def __init__(self, window: int = 50, min_periods: int = 5) -> None:
        if min_periods < 2 or window < min_periods:
            raise ValueError("window must be >= min_periods >= 2.")
        self.window = window
        self.min_periods = min_periods

    def init_state(self, _series_id: str) -> deque[float]:
        return deque(maxlen=self.window)

    def score_one(
        self, point: Point, state: deque[float]
    ) -> RawDetectorOutput:
        if len(state) < self.min_periods:
            return RawDetectorOutput(
                raw_score=0.0,
                observed=point.value,
                baseline=point.value,
                residual=0.0,
                context={
                    "is_warmup": True,
                    "mad": 0.0,
                    "window_size": len(state),
                    "contributors": ["rolling_median", "rolling_mad"],
                },
            )

        baseline = statistics.median(state)
        mad = statistics.median(abs(value - baseline) for value in state)
        residual = point.value - baseline
        score = ROBUST_Z_FACTOR * abs(residual) / max(mad, MAD_SCALE_FLOOR)
        return RawDetectorOutput(
            raw_score=score if math.isfinite(score) else 1e6,
            observed=point.value,
            baseline=baseline,
            residual=residual,
            context={
                "is_warmup": False,
                "mad": mad,
                "window_size": len(state),
                "contributors": ["rolling_median", "rolling_mad"],
            },
        )

    def update_state(
        self, point: Point, is_accepted_normal: bool, state: deque[float]
    ) -> deque[float]:
        if is_accepted_normal:
            state.append(point.value)
        return state
