"""Guarded baseline update: ties detector, score transform, event machine and drift together.

Per point:  score_one -> transform -> state machine -> decide accepted_normal -> update_state.

A point is folded into the baseline only if it is labeled NORMAL, scored below the EXIT threshold
(so the hysteresis band never leaks in), and is not inside a cooldown gap (it may be an event's tail).
Exceptions, both logged via reason codes:
  * warmup points are accepted (the baseline has to learn from somewhere);
  * REGIME_SHIFT_REBASELINE: after regime_shift_patience anomalous points in a row we assume the
    level really moved and let the baseline absorb data again (the backstop for abrupt change).
ADWIN runs on every post-warmup score (clipped to enter_threshold) and catches slow creep. On
DRIFT_RESET the likelihood reference window is trimmed to ADWIN's surviving window and, for
drift_refresh_steps points, NORMAL points up to enter_threshold may update the baseline.
Labels already emitted are never revised.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Any, Callable, Generic, TypeVar

from ..config import AdaptationConfig
from ..contracts import (LABEL_NORMAL, DetectorProtocol, Evidence, Point, PredictionRecord)
from ..events.state_machine import NON_FINITE_SCORE, EventStateMachine
from .drift import ADWIN
from .threshold import ScoreTransform, make_threshold

S = TypeVar("S")

DRIFT_RESET = "DRIFT_RESET"
REGIME_SHIFT_REBASELINE = "REGIME_SHIFT_REBASELINE"
BASELINE_UPDATE_SKIPPED = "BASELINE_UPDATE_SKIPPED"
INVALID_EVIDENCE = "INVALID_EVIDENCE"
INVALID_RAW_SCORE = "INVALID_RAW_SCORE"


def _finite_float_or_none(value: object) -> tuple[float | None, bool]:
    """Return a finite float and validity flag without leaking NaN/Infinity to records."""
    if isinstance(value, bool):
        return None, False
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, False
    return (number, True) if math.isfinite(number) else (None, False)


def _json_safe(value: object) -> tuple[Any, bool]:
    """Recursively sanitize detector context for strict JSON serialization."""
    if value is None or isinstance(value, (str, bool)):
        return value, True
    if isinstance(value, int):
        # Keep ordinary integer context values exact; reject absurdly large ones that cannot
        # be safely serialized by Python's strict JSON encoder.
        return (value, True) if value.bit_length() <= 4096 else (None, False)
    if isinstance(value, float):
        return (value, True) if math.isfinite(value) else (None, False)
    if isinstance(value, Real):
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None, False
        return (number, True) if math.isfinite(number) else (None, False)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        valid = True
        for key, item in value.items():
            safe_item, item_valid = _json_safe(item)
            result[str(key)] = safe_item
            valid = valid and item_valid
        return result, valid
    if isinstance(value, (list, tuple)):
        result = []
        valid = True
        for item in value:
            safe_item, item_valid = _json_safe(item)
            result.append(safe_item)
            valid = valid and item_valid
        return result, valid
    # Unknown objects are not allowed to make a prediction record unserializable.
    return None, False


class _SeriesCtx(Generic[S]):
    def __init__(self, series_id: str, cfg: AdaptationConfig, state: S) -> None:
        self.state = state
        self.machine = EventStateMachine(series_id, cfg)
        self.scorer: ScoreTransform = make_threshold(cfg)
        self.drift = ADWIN(cfg.adwin_delta, cfg.adwin_max_window, cfg.adwin_clock,
                           cfg.adwin_min_sub_window, value_range=cfg.enter_threshold)
        self.refresh_left = 0


class AdaptationLayer(Generic[S]):
    def __init__(self, detector: DetectorProtocol[S], init_state: Callable[[str], S],
                 config: AdaptationConfig | None = None) -> None:
        self.detector, self.init_state = detector, init_state
        self.cfg = config or AdaptationConfig()
        self._series: dict[str, _SeriesCtx[S]] = {}

    def reset(self, series_id: str | None = None) -> None:
        """Reset one series for a clean replay, or all series when omitted."""
        if series_id is None:
            self._series.clear()
        else:
            self._series.pop(series_id, None)

    def process(self, point: Point) -> PredictionRecord:
        cfg = self.cfg
        ctx = self._series.get(point.series_id)
        if ctx is None:
            ctx = self._series[point.series_id] = _SeriesCtx(point.series_id, cfg, self.init_state(point.series_id))

        out = self.detector.score_one(point, ctx.state)  # must not mutate state
        raw_score, raw_score_finite = _finite_float_or_none(out.raw_score)
        # The P2 contract is non-negative scores with larger meaning more anomalous.
        # A negative finite score is a contract violation, not evidence of normality.
        raw_score_valid = raw_score_finite and raw_score is not None and raw_score >= 0.0
        observed, observed_valid = _finite_float_or_none(out.observed)
        baseline, baseline_valid = _finite_float_or_none(out.baseline)
        residual, residual_valid = _finite_float_or_none(out.residual)
        safe_context, context_valid = _json_safe(out.context)
        if not isinstance(safe_context, dict):
            safe_context, context_valid = {}, False
        detector_warmup = safe_context.get("is_warmup", False)
        if "is_warmup" in safe_context and not isinstance(detector_warmup, bool):
            # Never interpret a malformed truthy string/int as an instruction to suppress alerts.
            context_valid = False
            safe_context["is_warmup"] = False
        contributors_raw = safe_context.get("contributors", [])
        contributors_valid = isinstance(contributors_raw, list) and all(
            isinstance(item, str) for item in contributors_raw
        )
        contributors = contributors_raw if contributors_valid else []
        evidence_valid = (
            observed_valid and baseline_valid and residual_valid and context_valid and contributors_valid
        )

        input_valid = raw_score_valid and evidence_valid
        # Send an invalid sentinel to the transform rather than appending a malformed value to
        # its short window. Both transform modes convert it to a fail-closed maximum score.
        score = ctx.scorer.transform(out.raw_score if input_valid else float("nan"))
        dec = ctx.machine.step(
            score,
            is_warmup=detector_warmup if isinstance(detector_warmup, bool) else False,
            timestamp=point.timestamp,
        )
        codes = list(dec.reason_codes)
        if not raw_score_finite and NON_FINITE_SCORE not in codes:
            codes.append(NON_FINITE_SCORE)
        if raw_score_finite and not raw_score_valid:
            codes.append(INVALID_RAW_SCORE)
        if not evidence_valid:
            codes.append(INVALID_EVIDENCE)

        # during a drift refresh the guard is relaxed from exit_threshold to enter_threshold
        limit = cfg.enter_threshold if ctx.refresh_left > 0 else cfg.exit_threshold
        ctx.refresh_left = max(ctx.refresh_left - 1, 0)
        # P3-IMPROVEMENT: corrupt scores/evidence must never be admitted during warmup or rebaseline.
        accepted = raw_score_valid and evidence_valid and (dec.in_warmup or (
            dec.label == LABEL_NORMAL and dec.anomaly_score < limit and not ctx.machine.in_cooldown
        ))
        if dec.regime_shift_suspected and raw_score_valid and evidence_valid:
            accepted = True
            codes.append(REGIME_SHIFT_REBASELINE)

        # ADWIN watches ALL post-warmup scores (clipped to enter_threshold), not just accepted ones:
        # the guard rejects exactly the points that reveal creep, so an accepted-only stream is blind.
        if (cfg.drift_enabled and not dec.in_warmup and raw_score_valid and evidence_valid
                and NON_FINITE_SCORE not in codes and ctx.drift.add(dec.anomaly_score)):
            codes.append(DRIFT_RESET)
            ctx.refresh_left = cfg.drift_refresh_steps
            ctx.scorer.forget_older_than(ctx.drift.width)  # drop reference data from before the change
        if not accepted and dec.label == LABEL_NORMAL:
            codes.append(BASELINE_UPDATE_SKIPPED)

        # P3-IMPROVEMENT: update the likelihood reference with the detector's RAW score.
        # dec.anomaly_score is transformed and belongs only to classification/evaluation.
        ctx.scorer.update(out.raw_score, accepted)
        ctx.state = self.detector.update_state(point, accepted, ctx.state)

        evidence: Evidence = {
            "observed": observed, "baseline": baseline, "residual": residual,
            "context": safe_context, "contributors": contributors,
            # These fields make the P3 transformation and guarded-update decision auditable.
            "raw_score": raw_score, "score_mode": cfg.threshold_mode,
            "baseline_update_accepted": accepted,
            "drift_window_width": ctx.drift.width if cfg.drift_enabled else None,
            "tail_probability": (
                ctx.scorer.tail_probability(ctx.scorer.last_z)
                if hasattr(ctx.scorer, "last_z") else None
            ),
            "event_start_timestamp": dec.event_start_timestamp,
            "incident_id": dec.incident_id,
            "incident_start_timestamp": dec.incident_start_timestamp,
        }
        return {
            "series_id": point.series_id, "timestamp": point.timestamp,
            "anomaly_score": dec.anomaly_score, "label": dec.label, "threshold": dec.threshold,
            "event_id": dec.event_id, "incident_id": dec.incident_id,
            "detector_version": cfg.detector_version,
            "reason_codes": codes, "evidence": evidence,
        }
