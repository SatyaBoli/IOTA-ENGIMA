"""Event-based anomaly evaluation harness adhering to Utkrishta 2026 specs.

Implements rigorous range-based event matching without point adjustment bias
(Kim et al., AAAI 2022; Tatbul et al., NeurIPS 2018).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class EventWindow:
    """Ground truth anomaly interval [start, end]."""

    series_id: str
    start: str
    end: str


@dataclass(frozen=True)
class PredictedEvent:
    """Predicted contiguous anomaly event."""

    series_id: str
    event_id: str
    start: str
    end: str
    first_trigger: str
    points_count: int


@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics."""

    event_f1: float
    precision: float
    recall: float
    true_positives: int
    false_positives: int
    false_negatives: int
    total_true_events: int
    total_predicted_events: int
    mean_detection_delay_seconds: float
    median_detection_delay_seconds: float
    per_series_metrics: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_f1": round(self.event_f1, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_true_events": self.total_true_events,
            "total_predicted_events": self.total_predicted_events,
            "mean_detection_delay_seconds": round(self.mean_detection_delay_seconds, 2),
            "median_detection_delay_seconds": round(self.median_detection_delay_seconds, 2),
        }


def extract_predicted_events(records: Sequence[Mapping[str, Any]]) -> list[PredictedEvent]:
    """Group contiguous or cooldown-merged anomalous records into predicted events."""
    events_by_id: dict[tuple[str, str], dict[str, Any]] = {}
    fallback_events: list[PredictedEvent] = []
    current_unlabeled: dict[str, Any] | None = None

    for r in records:
        lbl = r.get("label")
        eid = r.get("event_id")
        ts = r["timestamp"]
        sid = str(r["series_id"])

        if lbl == "ANOMALOUS":
            if eid:
                key = (sid, str(eid))
                if key not in events_by_id:
                    events_by_id[key] = {
                        "series_id": sid,
                        "event_id": str(eid),
                        "start": ts,
                        "end": ts,
                        "first_trigger": ts,
                        "count": 1,
                    }
                else:
                    events_by_id[key]["end"] = ts
                    events_by_id[key]["count"] += 1
            else:
                if current_unlabeled and current_unlabeled["series_id"] == sid:
                    current_unlabeled["end"] = ts
                    current_unlabeled["count"] += 1
                else:
                    if current_unlabeled:
                        fallback_events.append(
                            PredictedEvent(
                                series_id=current_unlabeled["series_id"],
                                event_id="unlabeled",
                                start=current_unlabeled["start"],
                                end=current_unlabeled["end"],
                                first_trigger=current_unlabeled["first_trigger"],
                                points_count=current_unlabeled["count"],
                            )
                        )
                    current_unlabeled = {
                        "series_id": sid,
                        "start": ts,
                        "end": ts,
                        "first_trigger": ts,
                        "count": 1,
                    }
        else:
            if current_unlabeled:
                fallback_events.append(
                    PredictedEvent(
                        series_id=current_unlabeled["series_id"],
                        event_id="unlabeled",
                        start=current_unlabeled["start"],
                        end=current_unlabeled["end"],
                        first_trigger=current_unlabeled["first_trigger"],
                        points_count=current_unlabeled["count"],
                    )
                )
                current_unlabeled = None

    if current_unlabeled:
        fallback_events.append(
            PredictedEvent(
                series_id=current_unlabeled["series_id"],
                event_id="unlabeled",
                start=current_unlabeled["start"],
                end=current_unlabeled["end"],
                first_trigger=current_unlabeled["first_trigger"],
                points_count=current_unlabeled["count"],
            )
        )

    all_events = [
        PredictedEvent(
            series_id=v["series_id"],
            event_id=v["event_id"],
            start=v["start"],
            end=v["end"],
            first_trigger=v["first_trigger"],
            points_count=v["count"],
        )
        for v in events_by_id.values()
    ]
    all_events.extend(fallback_events)
    return all_events


def evaluate_event_f1(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth_windows: Mapping[str, list[tuple[str, str]] | list[list[str]]],
) -> EvaluationMetrics:
    """Compute rigorous event-based precision, recall, F1, and latency.

    ground_truth_windows is a mapping from series_id or series_name to list of [start_iso, end_iso].
    """
    predicted_events = extract_predicted_events(predictions)

    # Convert timestamps to pandas Timestamps for accurate interval arithmetic
    gt_windows_by_series: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for sid, windows in ground_truth_windows.items():
        parsed = []
        for w in windows:
            s_ts = pd.to_datetime(w[0])
            e_ts = pd.to_datetime(w[1])
            parsed.append((s_ts, e_ts))
        gt_windows_by_series[str(sid)] = parsed

    # Track matches per series
    total_tp = 0
    total_fp = 0
    total_fn = 0
    delays_seconds: list[float] = []
    per_series_stats: dict[str, dict[str, float]] = {}

    all_series_ids = set(str(k) for k in gt_windows_by_series.keys()).union({str(p.series_id) for p in predicted_events})

    for sid in sorted(all_series_ids, key=str):
        sid_str = str(sid)
        true_windows = gt_windows_by_series.get(sid_str, [])
        pred_for_series = [p for p in predicted_events if str(p.series_id) == sid_str]

        matched_true = set()
        matched_pred = set()
        series_delays: list[float] = []

        for p_idx, pe in enumerate(pred_for_series):
            p_start = pd.to_datetime(pe.start)
            p_end = pd.to_datetime(pe.end)
            p_trig = pd.to_datetime(pe.first_trigger)

            hit_any = False
            for t_idx, (t_start, t_end) in enumerate(true_windows):
                # Overlap check between [p_start, p_end] and [t_start, t_end]
                if not (p_end < t_start or p_start > t_end):
                    hit_any = True
                    matched_true.add(t_idx)
                    matched_pred.add(p_idx)
                    # Compute delay relative to window start
                    delay_sec = max((p_trig - t_start).total_seconds(), 0.0)
                    series_delays.append(delay_sec)
                    delays_seconds.append(delay_sec)

        tp = len(matched_true)
        fn = len(true_windows) - tp
        fp = len(pred_for_series) - len(matched_pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_series_stats[sid] = {
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "total_windows": len(true_windows),
        }

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = (
        2 * overall_prec * overall_rec / (overall_prec + overall_rec)
        if (overall_prec + overall_rec) > 0
        else 0.0
    )

    mean_delay = float(pd.Series(delays_seconds).mean()) if delays_seconds else 0.0
    median_delay = float(pd.Series(delays_seconds).median()) if delays_seconds else 0.0

    return EvaluationMetrics(
        event_f1=overall_f1,
        precision=overall_prec,
        recall=overall_rec,
        true_positives=total_tp,
        false_positives=total_fp,
        false_negatives=total_fn,
        total_true_events=total_tp + total_fn,
        total_predicted_events=total_tp + total_fp,
        mean_detection_delay_seconds=mean_delay,
        median_detection_delay_seconds=median_delay,
        per_series_metrics=per_series_stats,
    )
