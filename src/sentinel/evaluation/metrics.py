"""Validation, loading, and provisional event-based anomaly metrics.

The repository contains NAB's pointwise/window-weighted scorer, but no
Sentinel competition event-matching specification. This module deliberately
keeps its event-F1 semantics separate and labels its output as provisional.
"""

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Tuple, Union


@dataclass(frozen=True)
class Prediction:
    """One saved anomaly score for a named time series and timestamp."""

    series_id: str
    timestamp: Union[datetime, str]
    score: float


@dataclass(frozen=True)
class _Event:
    start: datetime
    end: datetime


@dataclass
class EvaluationReport:
    """Stable metrics plus the runtime for one evaluation invocation."""

    threshold: float
    evaluation_runtime_seconds: float
    series: dict
    aggregate: dict
    protocol_status: str = "provisional"

    def to_dict(self):
        return {
            "protocol_status": self.protocol_status,
            "threshold": self.threshold,
            "evaluation_runtime_seconds": self.evaluation_runtime_seconds,
            "aggregate": self.aggregate,
            "per_series": self.series,
        }


def _parse_timestamp(value, context):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        if text.endswith(("Z", "z")):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(
                "{} must be a valid ISO-8601 timestamp: {!r}".format(
                    context, value
                )
            ) from exc
    else:
        raise ValueError(
            "{} must be a non-empty ISO-8601 timestamp".format(context)
        )

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _parse_windows(raw_windows, series_id):
    if not isinstance(raw_windows, (list, tuple)):
        raise ValueError(
            "Labels for {!r} must be a list of [start, end] windows".format(
                series_id
            )
        )

    windows = []
    for index, raw_window in enumerate(raw_windows):
        context = "Label window {} for {!r}".format(index, series_id)
        if not isinstance(raw_window, (list, tuple)) or len(raw_window) != 2:
            raise ValueError("{} must contain exactly [start, end]".format(context))
        start = _parse_timestamp(raw_window[0], context + " start")
        end = _parse_timestamp(raw_window[1], context + " end")
        if start > end:
            raise ValueError("{} starts after it ends".format(context))
        windows.append(_Event(start, end))

    windows.sort(key=lambda event: (event.start, event.end))
    merged = []
    for event in windows:
        if merged and event.start <= merged[-1].end:
            previous = merged[-1]
            merged[-1] = _Event(previous.start, max(previous.end, event.end))
        else:
            merged.append(event)
    return merged


def _parse_score(value, context):
    if isinstance(value, bool):
        raise ValueError("{} must be numeric; got {!r}".format(context, value))
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("{} must be numeric; got {!r}".format(context, value)) from exc
    if not math.isfinite(score):
        raise ValueError("{} must be finite; got {!r}".format(context, value))
    return score


def read_labels(path):
    """Read NAB-compatible JSON labels into normalized inclusive windows.

    The JSON must map each series ID to a list of timestamp pairs. All entries
    are validated, including empty-label series.
    """
    label_path = Path(path)
    try:
        with label_path.open("r", encoding="utf-8") as label_file:
            raw_labels = json.load(label_file)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid labels JSON in {}: {}".format(label_path, exc)) from exc

    if not isinstance(raw_labels, dict):
        raise ValueError("Labels JSON must be an object mapping series IDs to windows")

    labels = {}
    for series_id, raw_windows in raw_labels.items():
        if not isinstance(series_id, str) or not series_id.strip():
            raise ValueError("Label series IDs must be non-empty strings")
        normalized_id = _normalize_series_id(series_id, "Label series_id")
        if normalized_id in labels:
            raise ValueError(
                "Multiple label keys normalize to series ID {!r}".format(
                    normalized_id
                )
            )
        labels[normalized_id] = [
            (event.start, event.end)
            for event in _parse_windows(raw_windows, normalized_id)
        ]
    return labels


def _normalize_series_id(value, context):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("{} must be a non-empty string".format(context))
    return value.strip().replace("\\", "/")


def _infer_series_id(csv_path, input_path, labels):
    if input_path.is_file():
        relative = csv_path.name
    else:
        relative = csv_path.relative_to(input_path).as_posix()
    relative = relative.replace("\\", "/")
    relative_parent = relative.rsplit("/", 1)[0] if "/" in relative else ""
    filename = relative.rsplit("/", 1)[-1]

    candidates = []
    for label_id in labels:
        normalized = label_id.replace("\\", "/")
        label_parent = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
        label_filename = normalized.rsplit("/", 1)[-1]
        parent_matches = (
            not label_parent
            or relative_parent == label_parent
            or relative_parent.endswith("/" + label_parent)
        )
        filename_matches = (
            filename == label_filename
            or filename.endswith("_" + label_filename)
        )
        if parent_matches and filename_matches:
            candidates.append(label_id)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Cannot infer a unique series ID for {}; add a series_id column "
            "or specify --series-id".format(csv_path)
        )
    raise ValueError(
        "Cannot match {} to a labeled series; add a series_id column or "
        "specify --series-id".format(csv_path)
    )


def _read_prediction_file(csv_path, input_path, labels, series_id_override):
    try:
        prediction_file = csv_path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        raise OSError("Cannot read predictions from {}: {}".format(csv_path, exc)) from exc

    with prediction_file:
        reader = csv.DictReader(prediction_file)
        if not reader.fieldnames:
            raise ValueError("{} is empty or has no CSV header".format(csv_path))
        reader.fieldnames = [
            name.strip() if name else name for name in reader.fieldnames
        ]
        fieldnames = {name.strip() for name in reader.fieldnames if name}
        if "" in fieldnames or len(fieldnames) != len(reader.fieldnames):
            raise ValueError("{} has duplicate or empty CSV column names".format(csv_path))
        missing = {"timestamp"} - fieldnames
        if missing:
            raise ValueError(
                "{} is missing required column(s): {}".format(
                    csv_path, ", ".join(sorted(missing))
                )
            )
        score_column = (
            "score" if "score" in fieldnames
            else "anomaly_score" if "anomaly_score" in fieldnames
            else None
        )
        if score_column is None:
            raise ValueError(
                "{} must contain a 'score' or 'anomaly_score' column".format(
                    csv_path
                )
            )
        if series_id_override and input_path.is_dir():
            raise ValueError("--series-id can only be used with a single CSV file")

        file_predictions = []
        inferred_series_id = None
        for row_number, row in enumerate(reader, start=2):
            context = "{} row {}".format(csv_path, row_number)
            if None in row:
                raise ValueError("{} has more values than its header".format(context))

            row_id = row.get("series_id")
            if "series_id" in fieldnames:
                current_series_id = _normalize_series_id(
                    row_id, context + " series_id"
                )
            elif series_id_override:
                current_series_id = _normalize_series_id(
                    series_id_override, "--series-id"
                )
            else:
                if inferred_series_id is None:
                    inferred_series_id = _infer_series_id(
                        csv_path, input_path, labels
                    )
                current_series_id = inferred_series_id

            if current_series_id not in labels:
                raise ValueError(
                    "{} references series {!r}, which has no label entry".format(
                        context, current_series_id
                    )
                )

            timestamp = _parse_timestamp(row.get("timestamp"), context + " timestamp")
            score = _parse_score(row.get(score_column), context + " score")
            file_predictions.append(
                Prediction(current_series_id, timestamp, score)
            )

        return file_predictions


def read_predictions(path, labels, series_id=None):
    """Read prediction CSV file(s), supporting Sentinel and saved NAB results.

    Required columns are ``timestamp`` and ``score`` (or NAB's
    ``anomaly_score``). ``series_id`` is required for combined files; a single
    series can instead be inferred from a NAB-style relative path or supplied
    with ``series_id``. Extra columns such as NAB's ``label`` are ignored.
    """
    input_path = Path(path)
    if not isinstance(labels, Mapping):
        raise ValueError("labels must map series IDs to anomaly windows")
    normalized_labels = {}
    for label_id, windows in labels.items():
        normalized_id = _normalize_series_id(label_id, "Label series_id")
        if normalized_id in normalized_labels:
            raise ValueError(
                "Multiple label keys normalize to series ID {!r}".format(
                    normalized_id
                )
            )
        normalized_labels[normalized_id] = windows
    if not input_path.exists():
        raise FileNotFoundError("Prediction path does not exist: {}".format(input_path))
    if input_path.is_file():
        csv_paths = [input_path]
    elif input_path.is_dir():
        csv_paths = sorted(
            item for item in input_path.rglob("*")
            if item.is_file()
            and item.suffix.lower() == ".csv"
            and not item.name.endswith("_scores.csv")
        )
        if not csv_paths:
            raise ValueError("No prediction CSV files found in {}".format(input_path))
    else:
        raise ValueError("Prediction path is not a file or directory: {}".format(input_path))

    if series_id is not None and len(csv_paths) != 1:
        raise ValueError("--series-id can only be used with a single CSV file")

    predictions = []
    for csv_path in csv_paths:
        file_predictions = _read_prediction_file(
            csv_path, input_path, normalized_labels, series_id
        )
        predictions.extend(file_predictions)
    return predictions


def _coerce_prediction(item, index):
    if isinstance(item, Prediction):
        series_id = _normalize_series_id(
            item.series_id, "Prediction {} series_id".format(index)
        )
        timestamp = _parse_timestamp(
            item.timestamp, "Prediction {} timestamp".format(index)
        )
        score = _parse_score(item.score, "Prediction {} score".format(index))
        return Prediction(series_id, timestamp, score)

    if isinstance(item, Mapping):
        try:
            series_id = _normalize_series_id(
                item["series_id"], "Prediction {} series_id".format(index)
            )
            timestamp = _parse_timestamp(
                item["timestamp"], "Prediction {} timestamp".format(index)
            )
            score_key = "score" if "score" in item else "anomaly_score"
            score = _parse_score(
                item[score_key], "Prediction {} score".format(index)
            )
            return Prediction(series_id, timestamp, score)
        except KeyError as exc:
            raise ValueError(
                "Prediction {} is missing required field {!r}".format(index, exc.args[0])
            ) from exc
    raise ValueError(
        "Prediction {} must be a Prediction or mapping".format(index)
    )


def _safe_ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _f1(precision, recall):
    if precision is None and recall is None:
        return None
    if not precision or not recall:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _metric_counts(events, alerts, threshold):
    alerts = sorted(alerts, key=lambda prediction: prediction.timestamp)
    detected_events = []
    missed_events = []
    event_alert_timestamps = set()
    for event in events:
        in_event = [
            alert for alert in alerts
            if event.start <= alert.timestamp <= event.end
            and alert.score >= threshold
        ]
        if in_event:
            first_alert = in_event[0]
            delay = (first_alert.timestamp - event.start).total_seconds()
            detected_events.append({
                "start": event.start.isoformat().replace("+00:00", "Z"),
                "end": event.end.isoformat().replace("+00:00", "Z"),
                "detection_timestamp": (
                    first_alert.timestamp.isoformat().replace("+00:00", "Z")
                ),
                "detection_delay_seconds": delay,
            })
            event_alert_timestamps.update(alert.timestamp for alert in in_event)
        else:
            missed_events.append({
                "start": event.start.isoformat().replace("+00:00", "Z"),
                "end": event.end.isoformat().replace("+00:00", "Z"),
            })

    alarm_timestamps = {
        alert.timestamp for alert in alerts if alert.score >= threshold
    }
    false_alarm_timestamps = alarm_timestamps - event_alert_timestamps
    tp = len(detected_events)
    fp = len(false_alarm_timestamps)
    fn = len(missed_events)
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    delays = [
        item["detection_delay_seconds"] for item in detected_events
    ]
    return {
        "anomaly_events": len(events),
        "detected_events": tp,
        "missed_events": fn,
        "false_alarms": fp,
        "precision": precision,
        "recall": recall,
        "f1": _f1(precision, recall),
        "mean_detection_delay_seconds": (
            sum(delays) / len(delays) if delays else None
        ),
        "detected_event_details": detected_events,
        "missed_event_details": missed_events,
        "_false_alarm_timestamps": false_alarm_timestamps,
    }


def evaluate_predictions(
    predictions: Iterable[Prediction],
    labels: Mapping[
        str, Sequence[Tuple[Union[datetime, str], Union[datetime, str]]]
    ],
    threshold: float,
):
    """Calculate provisional event precision/recall/F1 and detection delay.

    Event windows are inclusive and overlapping windows are merged. An event
    is detected by its earliest threshold-crossing alert inside that event.
    Further alerts inside that event are duplicates and do not add true
    positives or false alarms. Every unique threshold-crossing alert outside
    all events is a false alarm. Scores at or above ``threshold`` are alerts.
    """
    started = time.perf_counter()
    threshold = _parse_score(threshold, "threshold")
    if not isinstance(labels, Mapping):
        raise ValueError("labels must map series IDs to anomaly windows")

    normalized_labels = {}
    for series_id, windows in labels.items():
        normalized_id = _normalize_series_id(series_id, "Label series_id")
        if normalized_id in normalized_labels:
            raise ValueError(
                "Multiple label keys normalize to series ID {!r}".format(
                    normalized_id
                )
            )
        normalized_labels[normalized_id] = _parse_windows(windows, normalized_id)

    scores_by_timestamp = {}
    for index, item in enumerate(predictions):
        prediction = _coerce_prediction(item, index)
        if prediction.series_id not in normalized_labels:
            raise ValueError(
                "Prediction {} references series {!r}, which has no label entry"
                .format(index, prediction.series_id)
            )
        key = (prediction.series_id, prediction.timestamp)
        scores_by_timestamp[key] = max(
            prediction.score, scores_by_timestamp.get(key, prediction.score)
        )

    normalized_predictions = [
        Prediction(series_id, timestamp, score)
        for (series_id, timestamp), score in sorted(scores_by_timestamp.items())
    ]

    alerts_by_series = {}
    for prediction in normalized_predictions:
        alerts_by_series.setdefault(prediction.series_id, []).append(prediction)

    per_series = {}
    totals = {"anomaly_events": 0, "detected_events": 0, "missed_events": 0}
    delays = []
    for series_id in sorted(normalized_labels):
        counts = _metric_counts(
            normalized_labels[series_id],
            alerts_by_series.get(series_id, []),
            threshold,
        )
        false_alarm_timestamps = counts.pop("_false_alarm_timestamps")
        counts["false_alarm_timestamps"] = [
            timestamp.isoformat().replace("+00:00", "Z")
            for timestamp in sorted(false_alarm_timestamps)
        ]
        per_series[series_id] = counts
        for key in totals:
            totals[key] += counts[key]
        delays.extend(
            event["detection_delay_seconds"]
            for event in counts["detected_event_details"]
        )

    false_alarm_count = sum(
        item["false_alarms"] for item in per_series.values()
    )
    aggregate_precision = _safe_ratio(
        totals["detected_events"],
        totals["detected_events"] + false_alarm_count,
    )
    aggregate_recall = _safe_ratio(
        totals["detected_events"],
        totals["detected_events"] + totals["missed_events"],
    )
    aggregate = {
        "series_count": len(per_series),
        "anomaly_events": totals["anomaly_events"],
        "detected_events": totals["detected_events"],
        "missed_events": totals["missed_events"],
        "false_alarms": false_alarm_count,
        "precision": aggregate_precision,
        "recall": aggregate_recall,
        "f1": _f1(aggregate_precision, aggregate_recall),
        "mean_detection_delay_seconds": (
            sum(delays) / len(delays) if delays else None
        ),
        "macro_f1": (
            sum(metrics["f1"] for metrics in per_series.values()
                if metrics["f1"] is not None)
            / sum(1 for metrics in per_series.values()
                  if metrics["f1"] is not None)
            if any(metrics["f1"] is not None for metrics in per_series.values())
            else None
        ),
    }
    elapsed = time.perf_counter() - started
    return EvaluationReport(
        threshold=threshold,
        evaluation_runtime_seconds=elapsed,
        series=per_series,
        aggregate=aggregate,
    )
