"""Adaptive score transforms: Modified z-score and Ahmad et al. Anomaly Likelihood.

Handles scale normalization, tail probabilities, and reference distribution protection.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Protocol

from ..config import AdaptationConfig

_SIGMA_FLOOR = 1e-4


class ScoreTransform(Protocol):
    def transform(self, raw_score: float) -> float: ...
    def update(self, score: float, accepted_normal: bool) -> None: ...
    def forget_older_than(self, keep: int) -> None: ...
    def reset(self) -> None: ...


def _clean(x: float, max_score: float) -> float:
    try:
        s = float(x)
    except (TypeError, ValueError):
        return max_score
    return max_score if not math.isfinite(s) else min(max(s, 0.0), max_score)


class RobustZThreshold:
    """Identity transform with bounds checking for robust z/MAD scores."""

    def __init__(self, cfg: AdaptationConfig) -> None:
        self.cfg = cfg

    def transform(self, raw_score: float) -> float:
        return _clean(raw_score, self.cfg.max_score)

    def update(self, score: float, accepted_normal: bool) -> None:
        pass

    def forget_older_than(self, keep: int) -> None:
        pass

    def reset(self) -> None:
        pass


class LikelihoodThreshold:
    """Ahmad, Lavin, Purdy & Agha (2017) Anomaly Likelihood.

    Compares short-term moving average of scores against long-run normal distribution.
    Emits standardized Gaussian z-score and tail probability Q(z).
    """

    def __init__(self, cfg: AdaptationConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._long: deque[float] = deque(maxlen=self.cfg.likelihood_long_window)
        self._short: deque[float] = deque(maxlen=self.cfg.likelihood_short_window)
        self._sum = 0.0
        self._sumsq = 0.0
        self.last_z = 0.0

    @staticmethod
    def tail_probability(z: float) -> float:
        """Q(z) = P(N(0, 1) > z), evaluated via math.erfc."""
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    @property
    def last_likelihood(self) -> float:
        return 1.0 - self.tail_probability(self.last_z)

    def _stats(self) -> tuple[float, float]:
        n = len(self._long)
        if n == 0:
            return 0.0, 1.0
        mu = self._sum / n
        var = max(self._sumsq / n - mu * mu, 0.0)
        return mu, max(math.sqrt(var), _SIGMA_FLOOR)

    def transform(self, raw_score: float) -> float:
        s = _clean(raw_score, self.cfg.max_score)
        self._short.append(s)
        if len(self._long) < self.cfg.likelihood_min_samples:
            self.last_z = 0.0
            return 0.0
        mu, sigma = self._stats()
        short_mean = sum(self._short) / len(self._short)
        self.last_z = max((short_mean - mu) / sigma, 0.0)
        return min(self.last_z, self.cfg.max_score)

    def update(self, score: float, accepted_normal: bool) -> None:
        if not accepted_normal:
            return
        s = _clean(score, self.cfg.max_score)
        if len(self._long) == self._long.maxlen:
            old = self._long[0]
            self._sum -= old
            self._sumsq -= old * old
        self._long.append(s)
        self._sum += s
        self._sumsq += s * s

    def forget_older_than(self, keep: int) -> None:
        keep = max(keep, self.cfg.likelihood_min_samples)
        while len(self._long) > keep:
            old = self._long.popleft()
            self._sum -= old
            self._sumsq -= old * old


def make_threshold(cfg: AdaptationConfig) -> ScoreTransform:
    return LikelihoodThreshold(cfg) if cfg.threshold_mode == "likelihood" else RobustZThreshold(cfg)
