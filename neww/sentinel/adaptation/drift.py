"""ADWIN-style adaptive-window drift detector (Bifet & Gavalda, SIAM SDM 2007).

The window grows while the stream is stationary and cuts older data when two sub-windows
have means differing by more than a Hoeffding-type bound. No manual time scale required.
"""
from __future__ import annotations

import math
from collections import deque
from itertools import accumulate


class ADWIN:
    """Adaptive Windowing drift detector."""

    def __init__(
        self,
        delta: float = 0.002,
        max_window: int = 1000,
        clock: int = 8,
        min_sub_window: int = 10,
        value_range: float = 1.0,
    ) -> None:
        if not (0 < delta < 1):
            raise ValueError("delta must be in (0, 1)")
        if value_range <= 0:
            raise ValueError("value_range must be > 0")
        self.delta = delta
        self.max_window = max_window
        self.clock = clock
        self.min_sub = min_sub_window
        self.range = value_range
        self._w: deque[float] = deque()
        self._t = 0

    @property
    def width(self) -> int:
        return len(self._w)

    @property
    def mean(self) -> float:
        return (sum(self._w) / len(self._w)) * self.range if self._w else 0.0

    def add(self, x: float) -> bool:
        """Insert one value. Returns True if a distribution change was detected."""
        if not math.isfinite(x):
            return False
        self._w.append(min(max(x, 0.0), self.range) / self.range)
        if len(self._w) > self.max_window:
            self._w.popleft()
        self._t += 1
        if self._t % self.clock != 0:
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
        dprime = self.delta / n
        ln_term = math.log(2.0 / dprime)
        for i in range(self.min_sub, n - self.min_sub + 1):
            n0, n1 = i, n - i
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            eps = math.sqrt(2.0 / m * var * ln_term) + 2.0 / (3.0 * m) * ln_term
            if abs(ps[i] / n0 - (ps[-1] - ps[i]) / n1) > eps:
                for _ in range(i):
                    self._w.popleft()
                return True
        return False

    def reset(self) -> None:
        self._w.clear()
        self._t = 0
