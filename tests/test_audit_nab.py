import json

import pandas as pd
import pytest

import scripts.audit_nab as audit_nab


@pytest.fixture
def corpus(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    labels_file = tmp_path / "combined_windows.json"

    def audit(labels):
        labels_file.write_text(json.dumps(labels), encoding="utf-8")
        return audit_nab.audit_dataset(data_dir, labels_file).set_index("file")

    return data_dir, audit


def write_csv(data_dir, name, contents):
    path = data_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_malformed_timestamps_are_counted(corpus):
    data_dir, audit = corpus
    write_csv(
        data_dir,
        "malformed_timestamps.csv",
        "timestamp,value\n2020-01-01,1\nnot-a-timestamp,2\n,3\n",
    )

    row = audit({"malformed_timestamps.csv": []}).loc[
        "malformed_timestamps.csv"
    ]

    assert row["invalid_timestamps"] == 1
    assert row["missing_timestamps"] == 1
    assert row["invalid_timestamps_flag"]
    assert row["missing_timestamps_flag"]


def test_duplicate_timestamps_are_reported(corpus):
    data_dir, audit = corpus
    write_csv(
        data_dir,
        "duplicates.csv",
        "timestamp,value\n2020-01-01,1\n2020-01-01,2\n",
    )

    row = audit({"duplicates.csv": []}).loc["duplicates.csv"]

    assert row["duplicate_timestamps"] == 1
    assert row["duplicate_timestamps_flag"]


def test_empty_and_header_only_csvs_are_distinguished(corpus):
    data_dir, audit = corpus
    (data_dir / "empty.csv").write_bytes(b"")
    write_csv(data_dir, "header_only.csv", "timestamp,value\n")

    report = audit({"empty.csv": [], "header_only.csv": []})

    assert report.loc["empty.csv", "empty_series_flag"]
    assert report.loc["empty.csv", "read_error_flag"]
    assert report.loc["header_only.csv", "empty_series_flag"]
    assert not report.loc["header_only.csv", "read_error_flag"]


def test_invalid_numeric_values_are_reported_separately(corpus):
    data_dir, audit = corpus
    write_csv(
        data_dir,
        "values.csv",
        "timestamp,value\n"
        "2020-01-01 00:00:00,\n"
        "2020-01-01 00:01:00,not-numeric\n"
        "2020-01-01 00:02:00,inf\n"
        "2020-01-01 00:03:00,NaN\n",
    )

    row = audit({"values.csv": []}).loc["values.csv"]

    assert row["missing_values"] == 1
    assert row["invalid_values"] == 1
    assert row["non_finite_values"] == 2
    assert row["invalid_values_flag"]


def test_malformed_anomaly_windows_count_invalid_entries(corpus):
    data_dir, audit = corpus
    write_csv(data_dir, "series.csv", "timestamp,value\n2020-01-01,1\n")

    row = audit({
        "series.csv": [
            ["2020-01-01", "2020-01-01"],
            ["not-a-timestamp", "2020-01-02"],
            ["2020-01-03", "2020-01-02"],
            ["missing-end"],
        ]
    }).loc["series.csv"]

    assert row["anomaly_windows"] == 4
    assert row["valid_anomaly_windows"] == 1
    assert row["invalid_anomaly_windows"] == 3
    assert (
        row["valid_anomaly_windows"] + row["invalid_anomaly_windows"]
        == row["anomaly_windows"]
    )
    assert row["malformed_windows_flag"]


def test_missing_label_entry_and_orphaned_label_file_are_reported(corpus):
    data_dir, audit = corpus
    write_csv(data_dir, "unlabeled.csv", "timestamp,value\n2020-01-01,1\n")

    report = audit({"orphan.csv": []})

    assert report.loc["unlabeled.csv", "missing_label_entry_flag"]
    assert report.loc["orphan.csv", "label_file_missing_flag"]
    assert not report.loc["orphan.csv", "data_file_exists"]


def test_timezone_offsets_are_normalized_for_windows(corpus):
    data_dir, audit = corpus
    write_csv(
        data_dir,
        "timezone.csv",
        "timestamp,value\n"
        "2020-01-01T00:00:00Z,1\n"
        "2020-01-01T01:00:00+01:00,2\n",
    )

    row = audit({
        "timezone.csv": [[
            "2020-01-01T01:00:00+01:00",
            "2020-01-01T00:00:00Z",
        ]]
    }).loc["timezone.csv"]

    assert row["invalid_anomaly_windows"] == 0
    assert row["valid_anomaly_windows"] == 1
    assert row["duplicate_timestamps"] == 1
    assert row["points_in_anomaly_windows"] == 2


def test_normalized_path_collisions_are_reported(corpus):
    data_dir, audit = corpus
    write_csv(data_dir, "nested/series.csv", "timestamp,value\n2020-01-01,1\n")

    report = audit({
        "nested/series.csv": [],
        "nested\\series.csv": [],
    })

    assert report.loc["nested/series.csv", "normalized_path_collision_flag"]
    assert report.loc["nested/series.csv", "audit_status"] == "label_error"


def test_anomaly_window_boundaries_are_inclusive(corpus):
    data_dir, audit = corpus
    write_csv(
        data_dir,
        "boundaries.csv",
        "timestamp,value\n"
        "2020-01-01 00:00:00,1\n"
        "2020-01-01 00:01:00,2\n"
        "2020-01-01 00:02:00,3\n",
    )

    row = audit({
        "boundaries.csv": [[
            "2020-01-01 00:00:00",
            "2020-01-01 00:01:00",
        ]]
    }).loc["boundaries.csv"]

    assert row["points_in_anomaly_windows"] == 2


def test_one_unreadable_file_does_not_stop_remaining_audit(corpus):
    data_dir, audit = corpus
    (data_dir / "unreadable.csv").write_bytes(
        b"timestamp,value\n\xff,not-a-value\n"
    )
    write_csv(data_dir, "valid.csv", "timestamp,value\n2020-01-01,1\n")

    report = audit({"unreadable.csv": [], "valid.csv": []})

    assert report.loc["unreadable.csv", "read_error_flag"]
    assert report.loc["valid.csv", "audit_status"] == "ok"
    assert len(report) == 2


def test_unexpected_file_processing_error_does_not_stop_audit(
    corpus, monkeypatch
):
    data_dir, audit = corpus
    write_csv(
        data_dir,
        "problem.csv",
        "timestamp,value\n2020-01-01,1\n",
    )
    write_csv(data_dir, "valid.csv", "timestamp,value\n2020-01-01,1\n")

    audit_file_contents = audit_nab._audit_file_contents

    def fail_for_one_file(path, relative_path, windows_by_file, collisions):
        if relative_path == "problem.csv":
            raise RuntimeError("unexpected per-file audit failure")
        return audit_file_contents(
            path, relative_path, windows_by_file, collisions
        )

    monkeypatch.setattr(
        audit_nab, "_audit_file_contents", fail_for_one_file
    )
    report = audit({"problem.csv": [], "valid.csv": []})

    assert report.loc["problem.csv", "audit_error_flag"]
    assert "unexpected per-file audit failure" in report.loc[
        "problem.csv", "audit_error"
    ]
    assert report.loc["valid.csv", "audit_status"] == "ok"
    assert len(report) == 2
