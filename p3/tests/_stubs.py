"""Minimal rolling median/MAD detector that satisfies DetectorProtocol (for tests only)."""
from __future__ import annotations
import statistics
from collections import deque
from sentinel.contracts import Point, RawDetectorOutput


class MadDetector:
    def __init__(self, window: int = 100, min_n: int = 20) -> None:
        self.window, self.min_n = window, min_n

    def init_state(self, _sid: str) -> deque:
        return deque(maxlen=self.window)

    def score_one(self, point: Point, state: deque) -> RawDetectorOutput:
        if len(state) < self.min_n:
            return RawDetectorOutput(0.0, point.value, point.value, 0.0, {"is_warmup": True})
        med = statistics.median(state)
        mad = max(statistics.median(abs(v - med) for v in state), 1e-6)
        r = point.value - med
        return RawDetectorOutput(0.6745 * abs(r) / mad, point.value, med, r, {"mad": mad})

    def update_state(self, point: Point, ok: bool, state: deque) -> deque:
        if ok:
            state.append(point.value)
        return state


def event_f1(records, truth_ranges):
    """Event-level F1 WITHOUT point adjustment: a predicted event is correct iff it overlaps a true
    range; a true range is found iff some predicted event overlaps it."""
    ev: dict[str, list[int]] = {}
    for i, r in enumerate(records):
        if r["event_id"]:
            ev.setdefault(r["event_id"], []).append(i)
    if not ev:
        return 0.0
    hit = lambda idx, rg: any(rg[0] <= i < rg[1] for i in idx)
    prec = sum(any(hit(idx, rg) for rg in truth_ranges) for idx in ev.values()) / len(ev)
    rec = sum(any(hit(idx, rg) for idx in ev.values()) for rg in truth_ranges) / len(truth_ranges)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
