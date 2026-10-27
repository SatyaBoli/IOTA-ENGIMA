"""Injectable configuration management for Sentinel detectors, adaptation, and evaluation.

Supports loading from YAML and programmatic dictionary instantiation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class AdaptationConfig:
    """Tunables for thresholds, event state machine, and drift adaptation."""

    # Hysteresis thresholding
    enter_threshold: float = 3.5  # score >= enter opens an anomaly event
    exit_threshold: float = 2.0   # score < exit closes an event
    # Event shaping
    min_event_length: int = 1     # consecutive detections to trigger alert
    cooldown_steps: int = 3       # re-entry within cooldown reuses event_id
    # Adaptation & Warmup
    warmup_steps: int = 30        # initial steps before alerting is allowed
    regime_shift_patience: int = 50  # consecutive alerts before re-baselining
    # Score transformation
    threshold_mode: str = "robust_z"  # "robust_z" | "likelihood"
    likelihood_long_window: int = 500
    likelihood_short_window: int = 10
    likelihood_min_samples: int = 30
    # Drift detection (ADWIN)
    drift_enabled: bool = True
    adwin_delta: float = 0.002
    adwin_max_window: int = 1000
    adwin_clock: int = 8
    adwin_min_sub_window: int = 10
    drift_refresh_steps: int = 50
    # Safeguards
    max_score: float = 1e6
    detector_version: str = "sentinel-0.1.0"

    def __post_init__(self) -> None:
        for name in ("enter_threshold", "exit_threshold", "max_score"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.enter_threshold <= 0:
            raise ValueError("enter_threshold must be positive")
        if not (0 <= self.exit_threshold <= self.enter_threshold):
            raise ValueError("exit_threshold must satisfy 0 <= exit_threshold <= enter_threshold")
        if self.min_event_length < 1:
            raise ValueError("min_event_length must be >= 1")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be >= 0")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if self.regime_shift_patience < 1:
            raise ValueError("regime_shift_patience must be >= 1")
        if self.threshold_mode not in ("robust_z", "likelihood"):
            raise ValueError("threshold_mode must be 'robust_z' or 'likelihood'")
        if not (0 < self.adwin_delta < 1):
            raise ValueError("adwin_delta must be in (0, 1)")


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for specific detectors (B0, B1, B2, B3, Ensemble)."""

    detector_type: str = "rolling_mad"  # "rolling_mad" | "ewma" | "iforest" | "autoencoder" | "ensemble"
    window_size: int = 50
    scale_floor: float = 1e-4
    ewma_alpha: float = 0.15
    iforest_n_estimators: int = 100
    iforest_contamination: float = 0.02
    ae_latent_dim: int = 8
    ae_epochs: int = 25
    ae_batch_size: int = 64
    ae_lr: float = 1e-3
    ae_device: str = "cuda"  # auto-detected to cuda if available


@dataclass(frozen=True)
class SentinelConfig:
    """Master configuration encompassing detector, adaptation, and runtime options."""

    detector: DetectorConfig = field(default_factory=DetectorConfig)
    adaptation: AdaptationConfig = field(default_factory=AdaptationConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SentinelConfig":
        det_data = data.get("detector", {})
        det_allowed = {f.name for f in fields(DetectorConfig)}
        det_config = DetectorConfig(**{k: v for k, v in det_data.items() if k in det_allowed})

        adapt_data = data.get("adaptation", {})
        adapt_allowed = {f.name for f in fields(AdaptationConfig)}
        adapt_config = AdaptationConfig(**{k: v for k, v in adapt_data.items() if k in adapt_allowed})

        return cls(detector=det_config, adaptation=adapt_config)

    @classmethod
    def from_yaml(cls, path: str) -> "SentinelConfig":
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)
