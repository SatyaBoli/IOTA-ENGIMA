import json

import pandas as pd
import pytest

import scripts.audit_nab as audit_nab


@pytest.fixture
def corpus(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    labels_file = tmp_path / "combined_windows.json"

    def run(labels):
        labels_file.write_text(json.dumps(labels), encoding="utf-8")
        return audit_nab.audit_dataset(data_dir, labels_file).set_index("file")

    return data_dir, run


def write_csv(data_dir, relative_path, contents):
    path = data_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_empty_and_header_only_csvs_are_reported(corpus):
    data_dir, run = corpus
    write_csv(data_dir, "empty.csv", "")
    write_csv(data_dir, "header_only.csv", "timestamp,value\n")

    report = run({})

    assert report.loc["empty.csv", "rows"] == 0
    assert report.loc["empty.csv", "empty_series_flag"]
    assert report.loc["empty.csv", "read_error_flag"]
    assert report.loc["empty.csv", "missing_label_entry_flag"]
    assert report.loc["header_only.csv", "rows"] == 0
    assert report.loc["header_only.csv", "empty_series_flag"]
    assert not report.loc["header_only.csv", "read_error_flag"]


def test_missing_and_invalid_timestamps_and_mixed_timezones(corpus):
    data_dir, run = corpus
    write_csv(
        data_dir,
        "mixed.csv",
        "timestamp,value\n"
        "2020-01-01T00:00:00Z,1\n"
        "2020-01-01 02:00:00+02:00,2\n"
        "not-a-time,3\n"
        ",4\n",
    )

    row = run({"mixed.csv": []}).loc["mixed.csv"]

    assert row["missing_timestamps"] == 1
    assert row["invalid_timestamps"] == 1
    assert row["duplicate_timestamps"] == 1
    assert row["timestamp_processing_errors"] == 0
    assert row["start_time"] == pd.Timestamp("2020-01-01T00:00:00Z")


def test_duplicate_out_of_order_and_suspicious_gap_flags(corpus):
    data_dir, run = corpus
    write_csv(
        data_dir,
        "times.csv",
        "timestamp,value\n"
        "2020-01-01 00:02:00,1\n"
        "2020-01-01 00:01:00,2\n"
        "2020-01-01 00:01:00,3\n"
        "2020-01-01 00:10:00,4\n",
    )

    row = run({"times.csv": []}).loc["times.csv"]

    assert row["duplicate_timestamps"] == 1
    assert row["duplicate_timestamps_flag"]
    assert row["out_of_order"]
    assert row["out_of_order_records"] == 1
    assert row["out_of_order_flag"]
    assert row["potential_time_gaps"] == 1
    assert row["suspicious_time_gaps_flag"]


def test_value_categories_and_quality_flags_are_separate(corpus):
    data_dir, run = corpus
    write_csv(
        data_dir,
        "values.csv",
        "timestamp,value\n"
        "2020-01-01 00:00:00,\n"
        "2020-01-01 00:01:00,not-numeric\n"
        "2020-01-01 00:02:00,inf\n"
        "2020-01-01 00:03:00,-inf\n"
        "2020-01-01 00:04:00,NaN\n"
        "2020-01-01 00:05:00,5\n",
    )

    row = run({"values.csv": []}).loc["values.csv"]

    assert row["missing_values"] == 1
    assert row["invalid_values"] == 1
    assert row["non_finite_values"] == 3
    assert row["missing_values_flag"]
    assert row["invalid_values_flag"]
    assert row["non_finite_values_flag"]


def test_valid_invalid_and_boundary_windows_are_counted_consistently(corpus):
    data_dir, run = corpus
    write_csv(
        data_dir,
        "series.csv",
        "timestamp,value\n"
        "2020-01-01 00:00:00,1\n"
        "2020-01-01 00:01:00,2\n"
        "2020-01-01 00:02:00,3\n",
    )
    labels = {
        "series.csv": [
            ["2020-01-01 00:00:00", "2020-01-01 00:01:00"],
            ["malformed"],
            ["not-a-time", "2020-01-01 00:02:00"],
            ["2020-01-01 00:02:00", "2020-01-01 00:00:00"],
        ]
    }

    row = run(labels).loc["series.csv"]

    assert row["anomaly_windows"] == 4
    assert row["valid_anomaly_windows"] == 1
    assert row["invalid_anomaly_windows"] == 3
    assert (
        row["valid_anomaly_windows"] + row["invalid_anomaly_windows"]
        == row["anomaly_windows"]
    )
    assert row["points_in_anomaly_windows"] == 2
    assert row["malformed_windows_flag"]


def test_non_list_window_entry_counts_as_one_invalid_window(corpus):
    data_dir, run = corpus
    write_csv(data_dir, "series.csv", "timestamp,value\n2020-01-01,1\n")

    row = run({"series.csv": "not a window list"}).loc["series.csv"]

    assert row["anomaly_windows"] == 1
    assert row["valid_anomaly_windows"] == 0
    assert row["invalid_anomaly_windows"] == 1
    assert row["valid_anomaly_windows"] + row["invalid_anomaly_windows"] == (
        row["anomaly_windows"]
    )


def test_missing_label_entry_and_label_file_are_distinguished(corpus):
    data_dir, run = corpus
    write_csv(data_dir, "unlabeled.csv", "timestamp,value\n2020-01-01,1\n")

    report = run({"gone.csv": []})

    assert report.loc["unlabeled.csv", "missing_label_entry_flag"]
    assert report.loc["gone.csv", "label_file_missing_flag"]
    assert not report.loc["gone.csv", "data_file_exists"]


def test_normalized_label_path_collision_is_flagged_once(corpus):
    data_dir, run = corpus
    write_csv(data_dir, "nested/series.csv", "timestamp,value\n2020-01-01,1\n")

    report = run({
        "nested/series.csv": [],
        "nested\\series.csv": [],
    })

    assert len(report) == 1
    assert report.loc["nested/series.csv", "normalized_path_collision_flag"]
    assert report.loc["nested/series.csv", "audit_status"] == "label_error"


def test_unexpected_timestamp_processing_errors_do_not_abort(corpus, monkeypatch):
    data_dir, run = corpus
    write_csv(
        data_dir,
        "bad_timestamp_parser.csv",
        "timestamp,value\n2020-01-01,1\n",
    )

    def fail_parse(*args, **kwargs):
        raise RuntimeError("unexpected timestamp parser failure")

    monkeypatch.setattr(audit_nab.pd, "to_datetime", fail_parse)
    row = run({"bad_timestamp_parser.csv": []}).loc["bad_timestamp_parser.csv"]

    assert row["timestamp_processing_errors"] == 1
    assert row["timestamp_processing_error_flag"]
    assert "unexpected timestamp parser failure" in row["timestamp_errors"]
    assert row["audit_status"] == "data_quality_issues"


def test_non_object_label_document_is_rejected(corpus):
    data_dir, run = corpus
    labels_file = data_dir.parent / "not-an-object.json"
    labels_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        audit_nab.audit_dataset(data_dir, labels_file)
