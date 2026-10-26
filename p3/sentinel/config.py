"""Injectable configuration. No file paths are hardcoded; callers pass dicts or a path."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from numbers import Real
from typing import Any, Mapping


@dataclass(frozen=True)
class AdaptationConfig:
    """Tunables for thresholds, event handling and baseline adaptation.

    Defaults assume a robust z-score / MAD scaled anomaly score.
    """

    # --- hysteresis -------------------------------------------------------
    enter_threshold: float = 3.5  # score >= enter opens an event
    exit_threshold: float = 2.0  # while in an event, score < exit closes it
    # --- event shaping ----------------------------------------------------
    min_event_length: int = 1  # consecutive exceedances needed to open (1 = lowest delay)
    cooldown_steps: int = 3  # re-entry within this many NORMAL points reuses the event_id
    # --- adaptation -------------------------------------------------------
    warmup_steps: int = 30  # per-series points before any event may open
    regime_shift_patience: int = 50  # consecutive anomalous points before a regime shift is suspected
    # --- score transform (adaptation/threshold.py) -------------------------
    threshold_mode: str = "robust_z"  # "robust_z" (use P2 score as-is) | "likelihood" (Ahmad et al. 2017)
    likelihood_long_window: int = 500  # accepted-normal scores kept for the long-run mean/std
    likelihood_short_window: int = 10  # recent scores averaged for the short-term mean
    likelihood_min_samples: int = 30  # accepted-normal scores needed before the transform is live
    # --- drift (adaptation/drift.py, ADWIN) --------------------------------
    drift_enabled: bool = True
    adwin_delta: float = 0.002  # probability of a false drift detection
    adwin_max_window: int = 1000  # cap on ADWIN's window (bounds per-check cost)
    adwin_clock: int = 8  # test for a cut every `clock` insertions
    adwin_min_sub_window: int = 10  # smallest sub-window compared at a cut
    drift_refresh_steps: int = 50  # after DRIFT_RESET, accept NORMAL points up to enter_threshold for this long
    # --- output hygiene ---------------------------------------------------
    max_score: float = 1e6  # clamp for non-finite / huge scores
    detector_version: str = "sentinel-0.1.0"

    def __post_init__(self) -> None:
        for name in ("enter_threshold", "exit_threshold", "max_score", "adwin_delta"):
            value = getattr(self, name)
            try:
                finite_number = not isinstance(value, bool) and isinstance(value, Real) and math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                finite_number = False
            if not finite_number:
                raise ValueError(f"{name} must be a finite number")

        # P3-IMPROVEMENT: deque(maxlen=...) and range/window logic require integer counts. Validate types
        # before comparisons so malformed YAML fails at configuration load, not at inference.
        integer_minima = {
            "min_event_length": 1, "cooldown_steps": 0, "warmup_steps": 0,
            "regime_shift_patience": 1, "likelihood_long_window": 2,
            "likelihood_short_window": 1, "likelihood_min_samples": 2,
            "adwin_max_window": 2, "adwin_clock": 1,
            "adwin_min_sub_window": 1, "drift_refresh_steps": 0,
        }
        for name, minimum in integer_minima.items():
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

        if self.enter_threshold <= 0:
            raise ValueError("enter_threshold must be > 0")
        if not 0 <= self.exit_threshold <= self.enter_threshold:
            raise ValueError("require 0 <= exit_threshold <= enter_threshold")
        if self.max_score < self.enter_threshold:
            raise ValueError("max_score must be >= enter_threshold")
        if self.threshold_mode not in ("robust_z", "likelihood"):
            raise ValueError("threshold_mode must be 'robust_z' or 'likelihood'")
        if self.likelihood_short_window >= self.likelihood_long_window:
            raise ValueError("require likelihood_short_window < likelihood_long_window")
        if self.likelihood_min_samples > self.likelihood_long_window:
            raise ValueError("require likelihood_min_samples <= likelihood_long_window")
        if not 0 < self.adwin_delta < 1:
            raise ValueError("adwin_delta must be in (0, 1)")
        if self.adwin_max_window < 2 * self.adwin_min_sub_window:
            raise ValueError("adwin_max_window must be >= 2 * adwin_min_sub_window")
        if not isinstance(self.drift_enabled, bool):
            raise ValueError("drift_enabled must be bool")
        if not isinstance(self.detector_version, str) or not self.detector_version.strip():
            raise ValueError("detector_version must be a non-empty string")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AdaptationConfig":
        """Accepts a flat dict or one nested under an 'adaptation' key (as in baseline.yaml).

        Unknown keys raise, so typos in YAML are caught instead of silently ignored.
        """
        section = data.get("adaptation", data)
        allowed = {f.name for f in fields(cls)}
        unknown = set(section) - allowed
        if unknown:
            raise ValueError(f"Unknown adaptation config keys: {sorted(unknown)}")
        return cls(**section)

    @classmethod
    def from_yaml(cls, path: str) -> "AdaptationConfig":
        import yaml  # imported lazily so the core package works without PyYAML

        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})
