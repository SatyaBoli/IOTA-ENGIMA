"""ADWIN-style adaptive-window drift detector (Bifet & Gavalda, SIAM SDM 2007).

The window grows while the stream is stationary and is cut when two sub-windows have means that
differ by more than a Hoeffding-type bound, so no time scale has to be guessed. `delta` bounds the
probability of a false drift detection. This is a compact, dependency-free variant that keeps a
bounded raw window and tests every split point with prefix sums (instead of the paper's exponential
histogram), checking every `clock` insertions. `river.drift.ADWIN` is a drop-in alternative if it is
an allowed dependency; the interface used by guarded_update.py is just add() -> bool.

Inputs are clipped to [0, value_range] and scaled to [0, 1], as the bound assumes bounded values.
"""
from __future__ import annotations

import math
from collections import deque
from itertools import accumulate


class ADWIN:
    def __init__(self, delta: float = 0.002, max_window: int = 1000, clock: int = 8,
                 min_sub_window: int = 10, value_range: float = 1.0) -> None:
        # P3-IMPROVEMENT: reject malformed bounds before any streaming state is created.
        try:
            delta_finite = math.isfinite(float(delta)) and 0 < float(delta) < 1
        except (TypeError, ValueError, OverflowError):
            delta_finite = False
        if not delta_finite:
            raise ValueError("delta must be finite and in (0, 1)")
        try:
            range_valid = math.isfinite(float(value_range)) and float(value_range) > 0
        except (TypeError, ValueError, OverflowError):
            range_valid = False
        if not range_valid:
            raise ValueError("value_range must be finite and > 0")
        delta, value_range = float(delta), float(value_range)
        for name, value, minimum in (
            ("max_window", max_window, 2),
            ("clock", clock, 1),
            ("min_sub_window", min_sub_window, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if max_window < 2 * min_sub_window:
            raise ValueError("max_window must be >= 2 * min_sub_window")
        self.delta, self.max_window, self.clock = delta, max_window, clock
        self.min_sub, self.range = min_sub_window, value_range
        self._w: deque[float] = deque()
        self._t = 0

    @property
    def width(self) -> int:
        return len(self._w)

    @property
    def mean(self) -> float:
        return (sum(self._w) / len(self._w)) * self.range if self._w else 0.0

    def add(self, x: float) -> bool:
        """Insert one value. Returns True if a distribution change was detected (old data dropped)."""
        # P3-IMPROVEMENT: malformed/overflowing samples are ignored, never inserted into ADWIN.
        try:
            x = float(x)
        except (TypeError, ValueError, OverflowError):
            return False
        if not math.isfinite(x):
            return False
        self._w.append(min(max(x, 0.0), self.range) / self.range)
        if len(self._w) > self.max_window:
            self._w.popleft()
        self._t += 1
        if self._t % self.clock:
            return False
        changed = False
        while self._cut():
            changed = True
        return changed

    def _cut(self) -> bool:
        n = len(self._w)
        if n < 2 * self.min_sub:
            return False
        data = list(self._w)
        ps = [0.0] + list(accumulate(data))
        mu = ps[-1] / n
        var = max(sum((v - mu) ** 2 for v in data) / n, 0.0)
        dprime = self.delta / n  # union bound over the n possible split points
        ln_term = math.log(2.0 / dprime)
        for i in range(self.min_sub, n - self.min_sub + 1):
            n0, n1 = i, n - i
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt(2.0 / m * var * ln_term) + 2.0 / (3.0 * m) * ln_term
            if abs(ps[i] / n0 - (ps[-1] - ps[i]) / n1) > eps:
                for _ in range(i):  # drop the older sub-window
                    self._w.popleft()
                return True
        return False

    def reset(self) -> None:
        self._w.clear()
        self._t = 0
