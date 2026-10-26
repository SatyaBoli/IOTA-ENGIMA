"""Score transforms applied between P2's raw score and the event state machine.

robust_z    : identity. P2 already emits 0.6745*|x-median|/MAD, so the 3.5 / 2.0 hysteresis band
              is Iglewicz & Hoaglin (1993)'s outlier rule (3.5) plus their "borderline" zone (2-3.5).
likelihood  : a research-inspired adaptation of Ahmad, Lavin, Purdy & Agha (2017). Compare the
              short-term MEAN of recent raw scores against the long-run distribution of accepted-normal
              raw scores. This compact variant returns a one-sided Gaussian z-score using the same
              3.5 / 2.0 thresholds as robust_z; it is not claimed to reproduce the paper exactly.
              `tail_probability` (via math.erfc) exposes the corresponding Gaussian Q(z) for evidence.

Both are causal: transform() sees only past data plus the current score. The long-run window is only
fed by update(..., accepted_normal=True), so anomalies cannot inflate the reference distribution.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Protocol

from ..config import AdaptationConfig

_SIGMA_FLOOR = 1e-3  # avoids z blowing up on a perfectly flat reference window


class ScoreTransform(Protocol):
    def transform(self, raw_score: float) -> float: ...
    def update(self, score: float, accepted_normal: bool) -> None: ...
    def forget_older_than(self, keep: int) -> None: ...
    def reset(self) -> None: ...


def _finite_float(x: object) -> float | None:
    """Return a finite numeric value, or None for malformed/non-finite input."""
    if isinstance(x, bool):
        return None
    try:
        value = float(x)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _clean(x: float, max_score: float) -> float:
    value = _finite_float(x)
    return max_score if value is None else min(max(value, 0.0), max_score)


class RobustZThreshold:
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
    def __init__(self, cfg: AdaptationConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._long: deque[float] = deque(maxlen=self.cfg.likelihood_long_window)
        self._short: deque[float] = deque(maxlen=self.cfg.likelihood_short_window)
        self.last_z = 0.0

    @staticmethod
    def tail_probability(z: float) -> float:
        """Q(z) = P(N(0,1) > z), computed with math.erfc."""
        return 0.5 * math.erfc(z / math.sqrt(2.0))

    @property
    def last_likelihood(self) -> float:
        return 1.0 - self.tail_probability(self.last_z)

    def _stats(self) -> tuple[float, float]:
        # P3-IMPROVEMENT: recompute stable central moments with fsum. The E[x²]-E[x]²
        # identity can lose precision when long-running score values have a large offset.
        n = len(self._long)
        mu = math.fsum(self._long) / n
        var = math.fsum((x - mu) * (x - mu) for x in self._long) / n
        return mu, max(math.sqrt(max(var, 0.0)), _SIGMA_FLOOR)

    def transform(self, raw_score: float) -> float:
        # P3-IMPROVEMENT: invalid detector scores fail closed but never enter the rolling
        # likelihood window. AdaptationLayer separately records NON_FINITE_SCORE and blocks updates.
        finite_score = _finite_float(raw_score)
        if finite_score is None:
            self.last_z = self.cfg.max_score
            return self.cfg.max_score
        s = min(max(finite_score, 0.0), self.cfg.max_score)
        self._short.append(s)
        if len(self._long) < self.cfg.likelihood_min_samples:
            self.last_z = 0.0
            return 0.0  # not enough reference data; the state machine's warmup covers this too
        mu, sigma = self._stats()
        short_mean = sum(self._short) / len(self._short)
        self.last_z = max((short_mean - mu) / sigma, 0.0)  # one-sided: only high scores matter
        return min(self.last_z, self.cfg.max_score)

    def update(self, score: float, accepted_normal: bool) -> None:
        # P3-IMPROVEMENT: reference samples are the detector's RAW scores, not transformed z-scores.
        # An invalid value is never admitted even if a caller accidentally passes accepted_normal=True.
        if not accepted_normal:
            return
        finite_score = _finite_float(score)
        if finite_score is None:
            return
        s = min(max(finite_score, 0.0), self.cfg.max_score)
        self._long.append(s)

    def forget_older_than(self, keep: int) -> None:
        """After a drift signal, keep only the `keep` most recent reference scores."""
        keep = max(keep, self.cfg.likelihood_min_samples)
        while len(self._long) > keep:
            self._long.popleft()


def make_threshold(cfg: AdaptationConfig) -> ScoreTransform:
    return LikelihoodThreshold(cfg) if cfg.threshold_mode == "likelihood" else RobustZThreshold(cfg)
