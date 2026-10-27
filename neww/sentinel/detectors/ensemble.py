"""Hybrid Ensemble Detector: Calibrated fusion of deep and statistical detectors.

Fuses Causal TCN-AE, Rolling MAD, and EWMA into a calibrated multi-perspective detector.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

from ..contracts import Point, RawDetectorOutput
from .base import BaseDetector
from .ewma import EWMADetector
from .rolling_mad import RollingMedianMADDetector
from .tcn_autoencoder import CausalTCNAEDetector

EXPECTED_TCN_CHECKPOINT_SHA256 = (
    "6bd10611237536852794d46c70c02bd341891053273d5184a6e36001f1d31717"
)


class HybridEnsembleDetector(BaseDetector):
    """Calibrated ensemble combining statistical baselines and deep neural reconstruction."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = "models/sentinel_tcn_best.pt",
        weights: tuple[float, float, float] = (0.45, 0.25, 0.30),  # (MAD, EWMA, TCN-AE)
        scale_floor: float = 0.05,
        device: str = "cuda",
        version: str = "ensemble-0.2.0",
    ) -> None:
        self.weights = weights
        self._version = version
        self.scale_floor = scale_floor

        self.mad_detector = RollingMedianMADDetector(window_size=50, warmup_steps=30, scale_floor=scale_floor)
        self.ewma_detector = EWMADetector(alpha=0.15, warmup_steps=30, scale_floor=scale_floor)
        self.tcn_detector = CausalTCNAEDetector(
            window_size=32,
            warmup_steps=25,
            scale_floor=scale_floor,
            device=device,
        )

        if not checkpoint_path:
            raise ValueError("The NEWW ensemble requires its trained TCN checkpoint.")
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"NEWW TCN checkpoint not found: {checkpoint}")
        checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint_sha256 != EXPECTED_TCN_CHECKPOINT_SHA256:
            raise ValueError(
                "NEWW TCN checkpoint checksum mismatch: "
                f"expected {EXPECTED_TCN_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
            )

        import torch

        try:
            state_dict = torch.load(
                checkpoint,
                map_location=self.tcn_detector.device,
                weights_only=True,
            )
            if not isinstance(state_dict, dict):
                raise ValueError("Checkpoint does not contain a model state dictionary.")
            self.tcn_detector.net.load_state_dict(state_dict, strict=True)
        except (
            EOFError,
            OSError,
            pickle.UnpicklingError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(
                f"Could not load the NEWW trained TCN checkpoint {checkpoint}: {exc}"
            ) from exc
        self.tcn_detector.net.eval()
        self.checkpoint_path = str(checkpoint.resolve())
        self.checkpoint_sha256 = checkpoint_sha256
        self.checkpoint_loaded = True

    @property
    def version(self) -> str:
        return self._version

    def score_one(self, point: Point) -> RawDetectorOutput:
        out_mad = self.mad_detector.score_one(point)
        out_ewma = self.ewma_detector.score_one(point)
        out_tcn = self.tcn_detector.score_one(point)

        w_mad, w_ewma, w_tcn = self.weights

        # Calibrated weighted combination
        combined_score = (
            w_mad * out_mad.raw_score +
            w_ewma * out_ewma.raw_score +
            w_tcn * out_tcn.raw_score
        )

        # Consensus baseline
        consensus_baseline = (
            w_mad * out_mad.baseline +
            w_ewma * out_ewma.baseline +
            w_tcn * out_tcn.baseline
        )
        combined_residual = point.value - consensus_baseline

        is_warmup = (
            out_mad.context.get("is_warmup", False) or
            out_tcn.context.get("is_warmup", False)
        )

        contributors = []
        if out_mad.raw_score > 3.0:
            contributors.append("statistical_spike_mad")
        if out_ewma.raw_score > 3.0:
            contributors.append("trend_residual_ewma")
        if out_tcn.raw_score > 3.0:
            contributors.append("temporal_pattern_distortion_tcn")
        if not contributors:
            contributors.append("nominal_concurrence")

        return RawDetectorOutput(
            raw_score=float(combined_score),
            observed=float(point.value),
            baseline=float(consensus_baseline),
            residual=float(combined_residual),
            context={
                "mad_score": float(out_mad.raw_score),
                "ewma_score": float(out_ewma.raw_score),
                "tcn_score": float(out_tcn.raw_score),
                "checkpoint_loaded": self.checkpoint_loaded,
                "is_warmup": is_warmup,
                "contributors": contributors,
            },
        )

    def update_state(self, point: Point, is_anomalous: bool = False) -> None:
        self.mad_detector.update_state(point, is_anomalous=is_anomalous)
        self.ewma_detector.update_state(point, is_anomalous=is_anomalous)
        self.tcn_detector.update_state(point, is_anomalous=is_anomalous)

    def reset_state(self, series_id: Optional[str] = None) -> None:
        self.mad_detector.reset_state(series_id)
        self.ewma_detector.reset_state(series_id)
        self.tcn_detector.reset_state(series_id)
