"""Base classes and utilities for Sentinel anomaly detectors."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..contracts import Point, RawDetectorOutput


class BaseDetector(ABC):
    """Abstract base class for all Sentinel streaming anomaly detectors."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Detector identifier/version string."""
        ...

    @abstractmethod
    def score_one(self, point: Point) -> RawDetectorOutput:
        """Score incoming point using only prior state. Must be strictly causal."""
        ...

    @abstractmethod
    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        """Update internal rolling buffers/parameters conditioned on guard status."""
        ...

    @abstractmethod
    def reset_state(self, series_id: Optional[str] = None) -> None:
        """Reset internal buffers for a specific series or all series."""
        ...
