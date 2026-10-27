"""Causal per-series event state machine: hysteresis, confirmation, cooldown merge.

Each decision at time t depends only on scores up to and including t.
Labels are never revised after emission.

Threshold semantics: `threshold` is the one actually applied at t.
  * not in an event  -> enter_threshold
  * in an event      -> exit_threshold (including the step on which the event closes)

Gap semantics: points inside a cooldown gap are labeled NORMAL.
Re-entry within `cooldown_steps` NORMAL points reuses the previous event_id,
so downstream evaluation can merge by event_id.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..config import AdaptationConfig
from ..contracts import LABEL_ANOMALOUS, LABEL_NORMAL

# Reason codes emitted by this state machine
THRESHOLD_EXCEEDED = "THRESHOLD_EXCEEDED"
EVENT_START = "EVENT_START"
HYSTERESIS_HOLD = "HYSTERESIS_HOLD"
EVENT_END = "EVENT_END"
COOLDOWN_MERGE = "COOLDOWN_MERGE"
PENDING_CONFIRMATION = "PENDING_CONFIRMATION"
WARMUP = "WARMUP"
NON_FINITE_SCORE = "NON_FINITE_SCORE"
REGIME_SHIFT_SUSPECTED = "REGIME_SHIFT_SUSPECTED"


@dataclass(frozen=True)
class EventDecision:
    label: str
    event_id: Optional[str]
    anomaly_score: float  # sanitized, finite, in [0, max_score]
    threshold: float
    reason_codes: list[str]
    regime_shift_suspected: bool  # signal for the guarded-update layer
    in_warmup: bool


class EventStateMachine:
    """State machine for ONE series. Never share an instance across series."""

    def __init__(self, series_id: str, config: AdaptationConfig) -> None:
        self.series_id = series_id
        self.cfg = config
        self.reset()

    def reset(self) -> None:
        self._n_seen = 0
        self._active = False
        self._event_id: Optional[str] = None
        self._last_event_id: Optional[str] = None
        self._event_counter = 0
        self._pending = 0  # consecutive exceedances while waiting for confirmation
        self._run_len = 0  # consecutive anomalous-labeled points
        self._gap_len: Optional[int] = None  # NORMAL points since last anomalous point

    @property
    def in_event(self) -> bool:
        return self._active

    @property
    def in_cooldown(self) -> bool:
        """True while recent NORMAL points may still be the tail of the last event."""
        return (not self._active) and self._gap_len is not None and self._gap_len <= self.cfg.cooldown_steps

    def _sanitize(self, raw_score: float) -> tuple[float, bool]:
        try:
            s = float(raw_score)
        except (TypeError, ValueError):
            return self.cfg.max_score, False
        if not math.isfinite(s):
            return self.cfg.max_score, False
        return min(max(s, 0.0), self.cfg.max_score), True

    def _new_event_id(self) -> str:
        self._event_counter += 1
        return f"{self.series_id}_evt_{self._event_counter:04d}"

    def _close(self) -> None:
        self._last_event_id = self._event_id
        self._active = False
        self._event_id = None
        self._run_len = 0
        self._gap_len = 1

    def step(self, raw_score: float, is_warmup: bool = False) -> EventDecision:
        cfg = self.cfg
        score, finite = self._sanitize(raw_score)
        codes: list[str] = [] if finite else [NON_FINITE_SCORE]
        idx = self._n_seen
        self._n_seen += 1

        # Warmup: never open events, close any open one
        if is_warmup or idx < cfg.warmup_steps:
            if self._active:
                self._close()
            self._pending = 0
            codes.append(WARMUP)
            return EventDecision(LABEL_NORMAL, None, score, cfg.enter_threshold, codes, False, True)

        # Inside an active event
        if self._active:
            if score >= cfg.exit_threshold:
                self._run_len += 1
                codes.append(THRESHOLD_EXCEEDED if score >= cfg.enter_threshold else HYSTERESIS_HOLD)
                regime = self._run_len >= cfg.regime_shift_patience
                if regime:
                    codes.append(REGIME_SHIFT_SUSPECTED)
                return EventDecision(
                    LABEL_ANOMALOUS, self._event_id, score, cfg.exit_threshold, codes, regime, False
                )
            self._close()
            codes.append(EVENT_END)
            return EventDecision(LABEL_NORMAL, None, score, cfg.exit_threshold, codes, False, False)

        # Not in an active event
        if score >= cfg.enter_threshold:
            merge = self._gap_len is not None and self._gap_len <= cfg.cooldown_steps
            if merge:
                self._event_id = self._last_event_id
                codes += [THRESHOLD_EXCEEDED, COOLDOWN_MERGE]
            else:
                self._pending += 1
                if self._pending < cfg.min_event_length:
                    return EventDecision(
                        LABEL_NORMAL, None, score, cfg.enter_threshold,
                        codes + [PENDING_CONFIRMATION], False, False,
                    )
                self._event_id = self._new_event_id()
                codes += [THRESHOLD_EXCEEDED, EVENT_START]
            self._active = True
            self._pending = 0
            self._gap_len = None
            self._run_len = 1
            regime = self._run_len >= cfg.regime_shift_patience
            if regime:
                codes.append(REGIME_SHIFT_SUSPECTED)
            return EventDecision(
                LABEL_ANOMALOUS, self._event_id, score, cfg.enter_threshold, codes, regime, False
            )

        self._pending = 0
        if self._gap_len is not None:
            self._gap_len += 1
        return EventDecision(LABEL_NORMAL, None, score, cfg.enter_threshold, codes, False, False)


class EventRegistry:
    """Lazily creates one isolated state machine per series_id."""

    def __init__(self, config: AdaptationConfig) -> None:
        self.cfg = config
        self._machines: dict[str, EventStateMachine] = {}

    def get(self, series_id: str) -> EventStateMachine:
        m = self._machines.get(series_id)
        if m is None:
            m = self._machines[series_id] = EventStateMachine(series_id, self.cfg)
        return m

    def reset_all(self) -> None:
        self._machines.clear()
