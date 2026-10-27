"""Shared interfaces between data loaders, detectors, adaptation, and evaluation.

Defines standardized data shapes to guarantee strict causality and zero future data leakage.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, TypedDict, TypeVar

S = TypeVar("S")

LABEL_NORMAL = "NORMAL"
LABEL_ANOMALOUS = "ANOMALOUS"


@dataclass(frozen=True)
class Point:
    """One single streaming observation."""

    series_id: str
    timestamp: str  # ISO-8601 string or numeric string
    value: float


@dataclass(frozen=True)
class RawDetectorOutput:
    """Output from any detector scoring a single point.

    raw_score: Larger finite value => more anomalous (non-negative scale).
    observed:  The raw input value at time t.
    baseline:  Expected / reconstructed normal value based strictly on past context.
    residual:  observed - baseline (or relative deviation).
    context:   Metadata dictionary (e.g., mad, rolling_std, window_size, model_type).
    """

    raw_score: float
    observed: float
    baseline: float
    residual: float
    context: dict[str, Any] = field(default_factory=dict)


class Evidence(TypedDict, total=False):
    """Deterministic, machine-verifiable explanation for flagged points."""

    observed: float
    baseline: float
    residual: float
    deviation_sigmas: float
    drift_detected: bool
    context: dict[str, Any]
    contributors: list[str]
    rule_summary: str


class PredictionRecord(TypedDict):
    """Emitted for every timestamp. Conforms to organizer evaluation requirements."""

    series_id: str
    timestamp: str
    anomaly_score: float  # always finite, monotonic (higher = more anomalous)
    label: str  # "NORMAL" | "ANOMALOUS"
    threshold: float  # finite dynamic threshold evaluated at time t
    event_id: Optional[str]  # unique contiguous anomaly window ID or None
    detector_version: str
    reason_codes: list[str]
    evidence: Evidence


class DetectorProtocol(Protocol[S]):
    """Standard protocol implemented by every Sentinel detector plugin."""

    @property
    def version(self) -> str:
        """Version or identifier of the detector."""
        ...

    def fit(self, train_data: list[Point] | Any) -> None:
        """Fit detector on historical training data if applicable."""
        ...

    def score_one(self, point: Point) -> RawDetectorOutput:
        """Score a single incoming point using only prior state.

        Strictly causal: must NOT update state before computing score.
        """
        ...

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        """Update detector internal state with the point, conditioned on guard policy."""
        ...

    def reset_state(self) -> None:
        """Reset internal memory/buffers for a fresh series."""
        ...
