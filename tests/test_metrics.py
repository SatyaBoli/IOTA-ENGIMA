import csv
import json
import math

import pytest

from sentinel.evaluation import (
    Prediction,
    evaluate_predictions,
    read_labels,
    read_predictions,
)
from sentinel.evaluation.cli import main


def prediction(series, timestamp, score=1.0):
    return Prediction(series, timestamp, score)


def test_correct_detection_at_window_start_has_zero_delay():
    report = evaluate_predictions(
        [prediction("s", "2024-01-01T00:00:00Z")],
        {"s": [["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"]]},
        threshold=0.5,
    ).to_dict()

    assert report["aggregate"]["f1"] == 1.0
    assert report["aggregate"]["precision"] == 1.0
    assert report["aggregate"]["recall"] == 1.0
    assert report["aggregate"]["mean_detection_delay_seconds"] == 0


def test_late_detection_reports_elapsed_seconds():
    report = evaluate_predictions(
        [prediction("s", "2024-01-01T00:03:00Z")],
        {"s": [["2024-01-01T00:00:00Z", "2024-01-01T00:10:00Z"]]},
        threshold=0.5,
    ).to_dict()

    assert report["aggregate"]["mean_detection_delay_seconds"] == 180
    assert report["per_series"]["s"]["detected_events"] == 1


def test_missed_event_without_predictions_is_false_negative():
    metrics = evaluate_predictions(
        [], {"s": [["2024-01-01", "2024-01-02"]]}, 0.5
    ).to_dict()["aggregate"]

    assert metrics["missed_events"] == 1
    assert metrics["detected_events"] == 0
    assert metrics["recall"] == 0
    assert metrics["f1"] == 0


def test_alert_outside_window_is_false_alarm():
    metrics = evaluate_predictions(
        [prediction("s", "2024-01-01T00:00:00Z")],
        {"s": [["2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"]]},
        0.5,
    ).to_dict()["aggregate"]

    assert metrics["false_alarms"] == 1
    assert metrics["missed_events"] == 1
    assert metrics["precision"] == 0
    assert metrics["recall"] == 0


def test_repeated_alerts_inside_one_event_are_duplicates_not_false_alarms():
    metrics = evaluate_predictions(
        [
            prediction("s", "2024-01-01T00:01:00Z"),
            prediction("s", "2024-01-01T00:02:00Z"),
            prediction("s", "2024-01-01T00:03:00Z"),
        ],
        {"s": [["2024-01-01T00:00:00Z", "2024-01-01T00:05:00Z"]]},
        0.5,
    ).to_dict()["aggregate"]

    assert metrics["detected_events"] == 1
    assert metrics["false_alarms"] == 0
    assert metrics["mean_detection_delay_seconds"] == 60


def test_overlapping_windows_are_merged_before_matching():
    metrics = evaluate_predictions(
        [prediction("s", "2024-01-01T00:02:00Z")],
        {
            "s": [
                ["2024-01-01T00:00:00Z", "2024-01-01T00:03:00Z"],
                ["2024-01-01T00:02:00Z", "2024-01-01T00:05:00Z"],
            ]
        },
        0.5,
    ).to_dict()["aggregate"]

    assert metrics["anomaly_events"] == 1
    assert metrics["detected_events"] == 1
    assert metrics["missed_events"] == 0


@pytest.mark.parametrize(
    "timestamp, expected_detected, expected_false_alarms",
    [
        ("2023-12-31T23:59:59Z", 0, 1),
        ("2024-01-01T00:00:00Z", 1, 0),
        ("2024-01-01T00:00:10Z", 1, 0),
        ("2024-01-01T00:00:11Z", 0, 1),
    ],
)
def test_window_boundaries_are_inclusive_and_adjacent_points_are_outside(
    timestamp, expected_detected, expected_false_alarms
):
    metrics = evaluate_predictions(
        [prediction("s", timestamp)],
        {"s": [["2024-01-01T00:00:00Z", "2024-01-01T00:00:10Z"]]},
        0.5,
    ).to_dict()["aggregate"]

    assert metrics["detected_events"] == expected_detected
    assert metrics["false_alarms"] == expected_false_alarms


def test_empty_series_without_events_reports_undefined_rates():
    metrics = evaluate_predictions([], {"normal": []}, 0.5).to_dict()

    assert metrics["aggregate"]["false_alarms"] == 0
    assert metrics["aggregate"]["precision"] is None
    assert metrics["aggregate"]["recall"] is None
    assert metrics["aggregate"]["f1"] is None
    assert metrics["per_series"]["normal"]["f1"] is None


def test_alerts_on_series_without_anomalies_are_false_alarms():
    metrics = evaluate_predictions(
        [prediction("normal", "2024-01-01T00:00:00Z")],
        {"normal": []},
        0.5,
    ).to_dict()["per_series"]["normal"]

    assert metrics["false_alarms"] == 1
    assert metrics["precision"] == 0
    assert metrics["recall"] is None
    assert metrics["f1"] == 0


def test_multiple_series_are_aggregated_without_cross_series_state():
    report = evaluate_predictions(
        [
            prediction("first", "2024-01-01T00:00:00Z"),
            prediction("second", "2024-01-01T00:01:00Z"),
            prediction("second", "2024-01-01T00:03:00Z"),
        ],
        {
            "first": [["2024-01-01T00:00:00Z", "2024-01-01T00:00:30Z"]],
            "second": [["2024-01-01T00:02:00Z", "2024-01-01T00:02:30Z"]],
        },
        0.5,
    ).to_dict()

    assert report["per_series"]["first"]["detected_events"] == 1
    assert report["per_series"]["second"]["missed_events"] == 1
    assert report["per_series"]["second"]["false_alarms"] == 2
    assert report["aggregate"]["anomaly_events"] == 2
    assert report["aggregate"]["detected_events"] == 1
    assert report["aggregate"]["false_alarms"] == 2
    assert report["aggregate"]["missed_events"] == 1
    assert report["aggregate"]["f1"] == pytest.approx(0.4)


def test_absent_series_predictions_count_as_missed_events():
    metrics = evaluate_predictions(
        [prediction("present", "2024-01-01T00:00:00Z")],
        {
            "present": [["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"]],
            "missing": [["2024-01-02T00:00:00Z", "2024-01-02T00:00:01Z"]],
        },
        0.5,
    ).to_dict()["aggregate"]

    assert metrics["anomaly_events"] == 2
    assert metrics["detected_events"] == 1
    assert metrics["missed_events"] == 1


@pytest.mark.parametrize(
    "bad_timestamp", ["", "not-a-time", None, "2024-02-30T12:00:00Z"]
)
def test_invalid_prediction_timestamps_are_rejected(bad_timestamp):
    with pytest.raises(ValueError, match="timestamp"):
        evaluate_predictions(
            [prediction("s", bad_timestamp)],
            {"s": []},
            0.5,
        )


@pytest.mark.parametrize(
    "bad_score", [float("nan"), float("inf"), -float("inf"), "bad", True]
)
def test_invalid_prediction_scores_are_rejected(bad_score):
    with pytest.raises(ValueError, match="score"):
        evaluate_predictions(
            [prediction("s", "2024-01-01T00:00:00Z", bad_score)],
            {"s": []},
            0.5,
        )


def test_duplicate_series_timestamps_use_max_score_once():
    metrics = evaluate_predictions(
        [
            prediction("s", "2024-01-01T00:00:00Z", 0.2),
            prediction("s", "2024-01-01T00:00:00+00:00", 0.8),
        ],
        {"s": [["2024-01-01T00:00:00Z", "2024-01-01T00:00:01Z"]]},
        0.5,
    ).to_dict()["aggregate"]

    assert metrics["detected_events"] == 1
    assert metrics["false_alarms"] == 0


def test_invalid_overlapping_label_order_and_window_shape_are_validated():
    with pytest.raises(ValueError, match="starts after"):
        evaluate_predictions(
            [],
            {"s": [["2024-01-02", "2024-01-01"]]},
            0.5,
        )
    with pytest.raises(ValueError, match="exactly"):
        evaluate_predictions([], {"s": [["2024-01-01"]]}, 0.5)


def test_prediction_for_unlabeled_series_is_rejected():
    with pytest.raises(ValueError, match="no label entry"):
        evaluate_predictions(
            [prediction("unlabeled", "2024-01-01T00:00:00Z")],
            {"labeled": []},
            0.5,
        )


def test_non_finite_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold"):
        evaluate_predictions([], {"s": []}, float("nan"))


def test_identical_inputs_have_identical_metrics():
    inputs = [
        prediction("s", "2024-01-01T00:00:01Z"),
        prediction("s", "2024-01-01T00:00:02Z"),
    ]
    labels = {"s": [["2024-01-01T00:00:00Z", "2024-01-01T00:00:03Z"]]}
    first = evaluate_predictions(inputs, labels, 0.5).to_dict()
    second = evaluate_predictions(inputs, labels, 0.5).to_dict()

    first.pop("evaluation_runtime_seconds")
    second.pop("evaluation_runtime_seconds")
    assert first == second


def test_read_labels_and_saved_predictions_end_to_end(tmp_path):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        json.dumps({"series/a.csv": [["2024-01-01T00:00:00Z",
                                      "2024-01-01T00:00:10Z"]]}),
        encoding="utf-8",
    )
    prediction_path = tmp_path / "predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["series_id", "timestamp", "score", "label"]
        )
        writer.writeheader()
        writer.writerow({
            "series_id": "series/a.csv",
            "timestamp": "2024-01-01T00:00:05Z",
            "score": "0.8",
            "label": "0",
        })

    labels = read_labels(labels_path)
    predictions = read_predictions(prediction_path, labels)
    report = evaluate_predictions(predictions, labels, 0.5).to_dict()

    assert report["protocol_status"] == "provisional"
    assert report["aggregate"]["f1"] == 1.0
    assert math.isfinite(report["evaluation_runtime_seconds"])


def test_multi_series_csv_allows_same_timestamp_in_independent_series(tmp_path):
    labels = {"first": [], "second": []}
    prediction_path = tmp_path / "multi.csv"
    prediction_path.write_text(
        "series_id,timestamp,score\n"
        "first,2024-01-01T00:00:00Z,0.9\n"
        "second,2024-01-01T00:00:00Z,0.9\n",
        encoding="utf-8",
    )

    predictions = read_predictions(prediction_path, labels)
    report = evaluate_predictions(predictions, labels, 0.5).to_dict()

    assert report["aggregate"]["false_alarms"] == 2
    assert report["per_series"]["first"]["false_alarms"] == 1
    assert report["per_series"]["second"]["false_alarms"] == 1


def test_saved_nab_results_infer_series_and_ignore_embedded_label(tmp_path):
    labels = {
        "realKnownCause/metric.csv": [
            ["2024-01-01T00:00:00Z", "2024-01-01T00:00:10Z"]
        ]
    }
    result_path = tmp_path / "realKnownCause" / "detector_metric.csv"
    result_path.parent.mkdir()
    result_path.write_text(
        "timestamp,anomaly_score,label\n"
        "2024-01-01T00:00:05Z,0.9,0\n",
        encoding="utf-8",
    )

    predictions = read_predictions(tmp_path, labels)
    report = evaluate_predictions(predictions, labels, 0.5).to_dict()

    assert report["aggregate"]["detected_events"] == 1
    assert report["aggregate"]["false_alarms"] == 0


def test_saved_nab_score_summary_csv_is_skipped(tmp_path):
    labels = {"group/series.csv": []}
    series_file = tmp_path / "group" / "detector_series.csv"
    series_file.parent.mkdir()
    series_file.write_text(
        "timestamp,anomaly_score\n2024-01-01T00:00:00Z,0.9\n",
        encoding="utf-8",
    )
    (tmp_path / "detector_standard_scores.csv").write_text(
        "Detector,Profile,Score\n",
        encoding="utf-8",
    )

    predictions = read_predictions(tmp_path, labels)
    report = evaluate_predictions(predictions, labels, 0.5).to_dict()

    assert len(predictions) == 1
    assert report["aggregate"]["false_alarms"] == 1


def test_csv_missing_columns_and_invalid_scores_are_rejected(tmp_path):
    labels = {"s": []}
    bad_header = tmp_path / "bad_header.csv"
    bad_header.write_text("timestamp,value\n2024-01-01,1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="score"):
        read_predictions(bad_header, labels, series_id="s")

    bad_score = tmp_path / "bad_score.csv"
    bad_score.write_text(
        "timestamp,score\n2024-01-01,NaN\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="finite"):
        read_predictions(bad_score, labels, series_id="s")


def test_cli_writes_reproducible_json_report(tmp_path, capsys):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(
        '{"s":[["2024-01-01T00:00:00Z","2024-01-01T00:00:10Z"]]}',
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text(
        "series_id,timestamp,score\ns,2024-01-01T00:00:05Z,0.8\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"

    assert main([
        str(predictions_path), str(labels_path), "--threshold", "0.5",
        "--output", str(report_path),
    ]) == 0
    written = json.loads(report_path.read_text(encoding="utf-8"))
    printed = json.loads(capsys.readouterr().out)
    assert written == printed
    assert written["aggregate"]["f1"] == 1.0
