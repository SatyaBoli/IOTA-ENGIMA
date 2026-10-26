"""Shared interfaces between P2 (detectors), P3 (this layer) and P4/P1 (pipeline, evaluator).

Nothing here contains logic; it only pins down the data shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypedDict, TypeVar

S = TypeVar("S")

LABEL_NORMAL = "NORMAL"
LABEL_ANOMALOUS = "ANOMALOUS"


@dataclass(frozen=True)
class Point:
    """One observation of one series."""

    series_id: str
    timestamp: str  # ISO-8601
    value: float


@dataclass(frozen=True)
class RawDetectorOutput:
    """What P2's detector returns for a single point.

    raw_score: larger finite value => more anomalous (0.0+ scale).
    residual:  observed - baseline (or scaled deviation).
    context:   e.g. {"mad": float, "is_warmup": bool, "window_size": int,
                     "contributors": list[str]}
    """

    raw_score: float
    observed: float
    baseline: float
    residual: float
    context: dict[str, Any] = field(default_factory=dict)


class Evidence(TypedDict):
    # Invalid/non-finite detector evidence is represented as null, never NaN/Infinity.
    observed: float | None
    baseline: float | None
    residual: float | None
    context: dict[str, Any]
    contributors: list[str]
    # P3 audit fields: additive to the internal contract; P4 should map only required fields
    # into the organizer's external prediction schema.
    raw_score: float | None
    score_mode: str
    baseline_update_accepted: bool
    drift_window_width: int | None
    tail_probability: float | None
    event_start_timestamp: Optional[str]
    incident_id: Optional[str]
    incident_start_timestamp: Optional[str]


class PredictionRecord(TypedDict):
    """Emitted for every timestamp. Consumed by P4's pipeline and P1's evaluator."""

    series_id: str
    timestamp: str
    anomaly_score: float  # always finite
    label: str  # "NORMAL" | "ANOMALOUS"
    threshold: float  # finite threshold applied at this timestamp
    event_id: Optional[str]  # unique per contiguous anomalous span; set iff label == "ANOMALOUS"
    incident_id: Optional[str]  # groups spans separated by a causal cooldown gap
    detector_version: str
    reason_codes: list[str]
    evidence: Evidence


class DetectorProtocol(Protocol[S]):
    """Adapter interface P2's detectors must satisfy. State is opaque to P3."""

    def score_one(self, point: Point, state: S) -> RawDetectorOutput:
        """Score `point` using only information available before it. Must not mutate state."""
        ...

    def update_state(self, point: Point, is_accepted_normal: bool, state: S) -> S:
        """Fold `point` into the baseline. Called AFTER labeling.

        is_accepted_normal=False means the point must not contaminate the baseline.
        """
        ...
