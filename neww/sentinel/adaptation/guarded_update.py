"""Guarded baseline update: coordinates detector, score transform, event machine, and drift.

Enforces strict causal ordering per point:
  1. score_one (using strictly past context)
  2. transform score (robust z or anomaly likelihood)
  3. event state machine step
  4. emit PredictionRecord
  5. guarded update of detector baseline state (only if accepted as normal)
"""
from __future__ import annotations

from typing import Any, Callable, Generic, Optional, TypeVar

from ..config import AdaptationConfig
from ..contracts import LABEL_NORMAL, DetectorProtocol, Evidence, Point, PredictionRecord
from ..events.state_machine import NON_FINITE_SCORE, EventStateMachine
from .drift import ADWIN
from .threshold import ScoreTransform, make_threshold

S = TypeVar("S")

DRIFT_RESET = "DRIFT_RESET"
REGIME_SHIFT_REBASELINE = "REGIME_SHIFT_REBASELINE"
BASELINE_UPDATE_SKIPPED = "BASELINE_UPDATE_SKIPPED"


class SeriesContext:
    def __init__(self, series_id: str, cfg: AdaptationConfig) -> None:
        self.series_id = series_id
        self.machine = EventStateMachine(series_id, cfg)
        self.scorer: ScoreTransform = make_threshold(cfg)
        self.drift = ADWIN(
            cfg.adwin_delta,
            cfg.adwin_max_window,
            cfg.adwin_clock,
            cfg.adwin_min_sub_window,
            value_range=cfg.enter_threshold,
        )
        self.refresh_left = 0


class AdaptationLayer:
    """Orchestrates causal scoring, adaptation, and state update guarding across multiple series."""

    def __init__(
        self,
        detector: Any,
        config: Optional[AdaptationConfig] = None,
    ) -> None:
        self.detector = detector
        self.cfg = config or AdaptationConfig()
        self._series: dict[str, SeriesContext] = {}

    def get_context(self, series_id: str) -> SeriesContext:
        ctx = self._series.get(series_id)
        if ctx is None:
            ctx = self._series[series_id] = SeriesContext(series_id, self.cfg)
        return ctx

    def process(self, point: Point) -> PredictionRecord:
        cfg = self.cfg
        ctx = self.get_context(point.series_id)

        # 1. Score strictly with past state
        out = self.detector.score_one(point)

        # 2. Transform score
        score = ctx.scorer.transform(out.raw_score)

        # 3. Classify with event state machine
        is_warmup = bool(out.context.get("is_warmup", False))
        dec = ctx.machine.step(score, is_warmup=is_warmup)
        codes = list(dec.reason_codes)

        # 4. Guard policy: accept into baseline only if safe
        limit = cfg.enter_threshold if ctx.refresh_left > 0 else cfg.exit_threshold
        ctx.refresh_left = max(ctx.refresh_left - 1, 0)

        accepted = dec.in_warmup or (
            dec.label == LABEL_NORMAL and dec.anomaly_score < limit and not ctx.machine.in_cooldown
        )
        if dec.regime_shift_suspected:
            accepted = True
            codes.append(REGIME_SHIFT_REBASELINE)

        # 5. Drift detection: watches all post-warmup scores
        drift_fired = False
        if (
            cfg.drift_enabled
            and not dec.in_warmup
            and NON_FINITE_SCORE not in codes
            and ctx.drift.add(dec.anomaly_score)
        ):
            drift_fired = True
            codes.append(DRIFT_RESET)
            ctx.refresh_left = cfg.drift_refresh_steps
            ctx.scorer.forget_older_than(ctx.drift.width)

        if not accepted and dec.label == LABEL_NORMAL:
            codes.append(BASELINE_UPDATE_SKIPPED)

        # 6. Update reference scorer & detector state
        ctx.scorer.update(out.raw_score, accepted)
        self.detector.update_state(point, is_anomalous=(not accepted))

        # 7. Formulate evidence record
        evidence: Evidence = {
            "observed": float(out.observed),
            "baseline": float(out.baseline),
            "residual": float(out.residual),
            "raw_score": float(out.raw_score),
            "deviation_sigmas": float(dec.anomaly_score),
            "drift_detected": drift_fired,
            "context": dict(out.context),
            "contributors": list(out.context.get("contributors", ["residual_magnitude"])),
            "rule_summary": f"Observed={out.observed:.2f}, Baseline={out.baseline:.2f}, Score={dec.anomaly_score:.2f} (Thresh={dec.threshold:.2f})",
        }

        return {
            "series_id": point.series_id,
            "timestamp": point.timestamp,
            "anomaly_score": float(dec.anomaly_score),
            "label": dec.label,
            "threshold": float(dec.threshold),
            "event_id": dec.event_id,
            "detector_version": cfg.detector_version,
            "reason_codes": codes,
            "evidence": evidence,
        }

    def reset(self) -> None:
        self._series.clear()
        self.detector.reset_state()
