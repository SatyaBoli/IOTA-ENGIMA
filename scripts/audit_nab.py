"""Create a CSV inventory and data-quality report for the NAB corpus."""

from pathlib import Path
import json
import posixpath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
LABELS_FILE = ROOT / "labels" / "combined_windows.json"
OUTPUT_DIR = ROOT / "outputs"

FLAG_COLUMNS = [
    "duplicate_timestamps_flag",
    "missing_timestamps_flag",
    "invalid_timestamps_flag",
    "missing_values_flag",
    "invalid_values_flag",
    "non_finite_values_flag",
    "out_of_order_flag",
    "empty_series_flag",
    "suspicious_time_gaps_flag",
    "malformed_windows_flag",
    "windows_outside_data_flag",
    "missing_label_entry_flag",
    "label_file_missing_flag",
    "normalized_path_collision_flag",
    "read_error_flag",
    "missing_columns_flag",
    "timestamp_processing_error_flag",
    "audit_error_flag",
]

REPORT_COLUMNS = [
    "file",
    "data_file_exists",
    "audit_status",
    "rows",
    "start_time",
    "end_time",
    "cadence_median_seconds",
    "anomaly_windows",
    "valid_anomaly_windows",
    "invalid_anomaly_windows",
    "windows_outside_data",
    "points_in_anomaly_windows",
    "missing_timestamps",
    "invalid_timestamps",
    "timestamp_processing_errors",
    "timestamp_errors",
    "duplicate_timestamps",
    "out_of_order",
    "out_of_order_records",
    "missing_values",
    "invalid_values",
    "non_finite_values",
    "potential_time_gaps",
    "has_label_entry",
    "read_error",
    "audit_error",
    "label_errors",
] + FLAG_COLUMNS


def _empty_record(relative_path, status, has_label_entry=False,
                  data_file_exists=False):
    record = {
        "file": relative_path,
        "data_file_exists": data_file_exists,
        "audit_status": status,
        "rows": 0,
        "start_time": pd.NaT,
        "end_time": pd.NaT,
        "cadence_median_seconds": np.nan,
        "anomaly_windows": 0,
        "valid_anomaly_windows": 0,
        "invalid_anomaly_windows": 0,
        "windows_outside_data": 0,
        "points_in_anomaly_windows": 0,
        "missing_timestamps": 0,
        "invalid_timestamps": 0,
        "timestamp_processing_errors": 0,
        "timestamp_errors": "",
        "duplicate_timestamps": 0,
        "out_of_order": False,
        "out_of_order_records": 0,
        "missing_values": 0,
        "invalid_values": 0,
        "non_finite_values": 0,
        "potential_time_gaps": 0,
        "has_label_entry": has_label_entry,
        "read_error": "",
        "audit_error": "",
        "label_errors": "",
    }
    record.update((column, False) for column in FLAG_COLUMNS)
    return record


def _is_missing_cell(value):
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "",
            "na",
            "n/a",
            "null",
            "none",
            "nat",
            "#n/a",
        }
    return False


def _parse_timestamp_value(value):
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception as exc:
        return pd.NaT, str(exc)
    if pd.isna(parsed):
        return pd.NaT, ""
    return parsed, ""


def _parse_timestamp_column(raw_timestamps):
    """Parse mixed timestamp formats/time zones to UTC without losing the file."""
    processing_errors = []
    try:
        parsed = pd.to_datetime(
            raw_timestamps, errors="coerce", utc=True
        )
        parsed = pd.Series(parsed, index=raw_timestamps.index)
    except Exception:
        parsed = pd.Series(
            pd.NaT,
            index=raw_timestamps.index,
            dtype="datetime64[ns, UTC]",
        )

    missing = raw_timestamps.map(_is_missing_cell)
    for index, raw_value in raw_timestamps.items():
        if missing.loc[index] or not pd.isna(parsed.loc[index]):
            continue
        try:
            timestamp, error = _parse_timestamp_value(raw_value)
        except Exception as exc:
            timestamp, error = pd.NaT, str(exc)
        parsed.loc[index] = timestamp
        if error:
            processing_errors.append(
                "row {}: {}".format(index + 2, error)
            )

    return parsed, missing, processing_errors


def _parse_windows(relative_path, windows_by_file):
    if relative_path not in windows_by_file:
        return [], 0, 0, [], False

    raw_windows = windows_by_file[relative_path]
    if not isinstance(raw_windows, list):
        return [], 1, 1, [
            "Label entry must be a list of [start, end] windows"
        ], True

    windows = []
    errors = []
    invalid_count = 0
    for index, window in enumerate(raw_windows):
        if not isinstance(window, (list, tuple)) or len(window) != 2:
            errors.append("Window {} must contain start and end".format(index))
            invalid_count += 1
            continue
        if not all(isinstance(bound, str) for bound in window):
            errors.append(
                "Window {} bounds must be timestamp strings".format(index)
            )
            invalid_count += 1
            continue

        try:
            start, start_error = _parse_timestamp_value(window[0])
            end, end_error = _parse_timestamp_value(window[1])
        except Exception as exc:
            errors.append(
                "Window {} timestamp processing failed: {}".format(index, exc)
            )
            invalid_count += 1
            continue
        if start_error or end_error:
            errors.append(
                "Window {} timestamp processing failed: {}".format(
                    index, start_error or end_error
                )
            )
            invalid_count += 1
            continue
        if pd.isna(start) or pd.isna(end):
            errors.append("Window {} has an invalid timestamp".format(index))
            invalid_count += 1
            continue
        if start > end:
            errors.append("Window {} starts after it ends".format(index))
            invalid_count += 1
            continue

        windows.append((start, end))

    return windows, len(raw_windows), invalid_count, errors, False


def _set_status(record):
    if record["read_error_flag"]:
        record["audit_status"] = "read_error"
    elif record["audit_error_flag"]:
        record["audit_status"] = "audit_error"
    elif record["missing_columns_flag"]:
        record["audit_status"] = "missing_columns"
    elif record["empty_series_flag"]:
        record["audit_status"] = "empty_series"
    elif record["label_file_missing_flag"]:
        record["audit_status"] = "label_file_missing"
    elif record["normalized_path_collision_flag"] or record[
        "malformed_windows_flag"
    ]:
        record["audit_status"] = "label_error"
    elif record["missing_label_entry_flag"]:
        record["audit_status"] = "missing_label_entry"
    elif any(record[column] for column in FLAG_COLUMNS):
        record["audit_status"] = "data_quality_issues"
    else:
        record["audit_status"] = "ok"


def _audit_file_contents(path, relative_path, windows_by_file, path_collisions):
    has_label_entry = relative_path in windows_by_file
    record = _empty_record(
        relative_path,
        "ok",
        has_label_entry=has_label_entry,
        data_file_exists=True,
    )
    record["missing_label_entry_flag"] = not has_label_entry
    if relative_path in path_collisions:
        record["normalized_path_collision_flag"] = True
        record["label_errors"] = "Multiple label paths normalize to this path"

    try:
        df = pd.read_csv(path, keep_default_na=False)
    except pd.errors.EmptyDataError:
        record["empty_series_flag"] = True
        record["read_error"] = "CSV is empty and has no header"
        record["read_error_flag"] = True
        _set_status(record)
        return record
    except Exception as exc:
        record["read_error"] = str(exc)
        record["read_error_flag"] = True
        _set_status(record)
        return record

    record["rows"] = len(df)
    record["empty_series_flag"] = df.empty
    missing_columns = {"timestamp", "value"} - set(df.columns)
    if missing_columns:
        record["missing_columns_flag"] = True
        record["read_error"] = "Missing required columns: {}".format(
            ", ".join(sorted(missing_columns))
        )

    timestamps = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    missing_timestamps = pd.Series(False, index=df.index)
    if "timestamp" in df.columns:
        try:
            timestamps, missing_timestamps, timestamp_errors = (
                _parse_timestamp_column(df["timestamp"])
            )
            record["timestamp_processing_errors"] = len(timestamp_errors)
            record["timestamp_errors"] = "; ".join(timestamp_errors[:5])
            record["timestamp_processing_error_flag"] = bool(timestamp_errors)
            record["missing_timestamps"] = int(missing_timestamps.sum())
            record["invalid_timestamps"] = int(
                (~missing_timestamps & timestamps.isna()).sum()
            )
            valid_timestamps = timestamps.dropna()
            if not valid_timestamps.empty:
                record["start_time"] = valid_timestamps.min()
                record["end_time"] = valid_timestamps.max()
            record["duplicate_timestamps"] = int(
                (timestamps.notna() & timestamps.duplicated()).sum()
            )
            record["out_of_order_records"] = int(
                valid_timestamps.diff().lt(pd.Timedelta(0)).sum()
            )
            record["out_of_order"] = record["out_of_order_records"] > 0

            unique_sorted = valid_timestamps.drop_duplicates().sort_values()
            deltas = unique_sorted.diff().dt.total_seconds().dropna()
            positive_deltas = deltas[deltas > 0]
            if not positive_deltas.empty:
                cadence = float(positive_deltas.median())
                record["cadence_median_seconds"] = cadence
                record["potential_time_gaps"] = int(
                    (positive_deltas > 1.5 * cadence).sum()
                )
        except Exception as exc:
            record["timestamp_processing_errors"] += 1
            record["timestamp_processing_error_flag"] = True
            error_text = "timestamp analysis failed: {}".format(exc)
            record["timestamp_errors"] = "; ".join(
                part
                for part in (record["timestamp_errors"], error_text)
                if part
            )[:1000]

    if "value" in df.columns:
        raw_values = df["value"]
        missing_values = raw_values.map(_is_missing_cell)
        numeric_values = pd.to_numeric(raw_values, errors="coerce")
        nan_values = raw_values.map(
            lambda value: isinstance(value, str)
            and value.strip().lower() in {"nan", "+nan", "-nan"}
        )
        record["missing_values"] = int(missing_values.sum())
        record["invalid_values"] = int(
            (~missing_values & ~nan_values & numeric_values.isna()).sum()
        )
        record["non_finite_values"] = int(
            (
                (numeric_values.notna()
                 & ~np.isfinite(numeric_values.astype(float)))
                | nan_values
            ).sum()
        )

    windows, window_total, invalid_windows, label_errors, malformed_entry = (
        _parse_windows(relative_path, windows_by_file)
    )
    record["anomaly_windows"] = window_total
    record["invalid_anomaly_windows"] = invalid_windows
    record["valid_anomaly_windows"] = window_total - invalid_windows
    if malformed_entry:
        record["label_errors"] = "; ".join(
            part
            for part in (record["label_errors"], *label_errors)
            if part
        )
    elif label_errors:
        record["label_errors"] = "; ".join(
            part
            for part in (record["label_errors"], *label_errors)
            if part
        )

    if windows:
        valid_timestamps = timestamps.dropna()
        if valid_timestamps.empty:
            record["windows_outside_data"] = len(windows)
        else:
            first_timestamp = valid_timestamps.min()
            last_timestamp = valid_timestamps.max()
            points_in_windows = pd.Series(False, index=df.index)
            for index, (start, end) in enumerate(windows):
                try:
                    if start < first_timestamp or end > last_timestamp:
                        record["windows_outside_data"] += 1
                    points_in_windows |= timestamps.ge(start) & timestamps.le(
                        end
                    )
                except Exception as exc:
                    record["invalid_anomaly_windows"] += 1
                    record["valid_anomaly_windows"] -= 1
                    label_errors.append(
                        "Window {} could not be compared to data: {}".format(
                            index, exc
                        )
                    )
            record["points_in_anomaly_windows"] = int(
                points_in_windows.sum()
            )
            if label_errors:
                record["label_errors"] = "; ".join(
                    part
                    for part in (record["label_errors"], *label_errors)
                    if part
                )

    record["has_label_entry"] = has_label_entry
    record["missing_label_entry_flag"] = not has_label_entry
    record["duplicate_timestamps_flag"] = record["duplicate_timestamps"] > 0
    record["missing_timestamps_flag"] = record["missing_timestamps"] > 0
    record["invalid_timestamps_flag"] = record["invalid_timestamps"] > 0
    record["missing_values_flag"] = record["missing_values"] > 0
    record["invalid_values_flag"] = record["invalid_values"] > 0
    record["non_finite_values_flag"] = record["non_finite_values"] > 0
    record["out_of_order_flag"] = record["out_of_order"]
    record["suspicious_time_gaps_flag"] = record["potential_time_gaps"] > 0
    record["malformed_windows_flag"] = record["invalid_anomaly_windows"] > 0
    record["windows_outside_data_flag"] = record["windows_outside_data"] > 0
    _set_status(record)
    return record


def _audit_file(path, relative_path, windows_by_file, path_collisions):
    try:
        return _audit_file_contents(
            path, relative_path, windows_by_file, path_collisions
        )
    except Exception as exc:
        record = _empty_record(
            relative_path,
            "audit_error",
            has_label_entry=relative_path in windows_by_file,
            data_file_exists=True,
        )
        record["audit_error"] = "{}: {}".format(type(exc).__name__, exc)
        record["audit_error_flag"] = True
        record["missing_label_entry_flag"] = (
            relative_path not in windows_by_file
        )
        return record


def audit_dataset(data_dir, labels_file):
    with labels_file.open("r", encoding="utf-8") as labels_stream:
        windows_by_file = json.load(labels_stream)
    if not isinstance(windows_by_file, dict):
        raise ValueError("The combined windows file must contain a JSON object")

    normalized_labels = {}
    raw_paths_by_normalized = {}
    invalid_label_paths = []
    for label_path, windows in windows_by_file.items():
        if not isinstance(label_path, str):
            invalid_label_paths.append(label_path)
            continue
        normalized_path = posixpath.normpath(label_path.replace("\\", "/"))
        normalized_labels.setdefault(normalized_path, windows)
        raw_paths_by_normalized.setdefault(normalized_path, []).append(label_path)

    path_collisions = {
        normalized_path
        for normalized_path, raw_paths in raw_paths_by_normalized.items()
        if len(raw_paths) > 1
    }
    csv_files = sorted(data_dir.rglob("*.csv"))
    records = []
    data_paths = set()
    for path in csv_files:
        relative_path = path.relative_to(data_dir).as_posix()
        data_paths.add(relative_path)
        records.append(
            _audit_file(
                path,
                relative_path,
                normalized_labels,
                path_collisions,
            )
        )

    for label_path in sorted(normalized_labels):
        if label_path not in data_paths:
            record = _empty_record(
                label_path, "label_file_missing", has_label_entry=True
            )
            windows, total, invalid, errors, _ = _parse_windows(
                label_path, normalized_labels
            )
            record["anomaly_windows"] = total
            record["valid_anomaly_windows"] = len(windows)
            record["invalid_anomaly_windows"] = invalid
            record["label_errors"] = "; ".join(errors)
            record["label_file_missing_flag"] = True
            record["malformed_windows_flag"] = invalid > 0
            if label_path in path_collisions:
                record["normalized_path_collision_flag"] = True
                record["label_errors"] = "; ".join(
                    part
                    for part in (
                        record["label_errors"],
                        "Multiple label paths normalize to this path",
                    )
                    if part
                )
            _set_status(record)
            records.append(record)

    for label_path in invalid_label_paths:
        record = _empty_record(
            str(label_path), "invalid_label_path", has_label_entry=True
        )
        record["label_errors"] = "Label path must be a string"
        record["label_file_missing_flag"] = True
        _set_status(record)
        records.append(record)

    return pd.DataFrame(records, columns=REPORT_COLUMNS)


def main():
    report = audit_dataset(DATA_DIR, LABELS_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "nab_dataset_inventory.csv"
    report.to_csv(output_path, index=False)

    data_rows = report[report["data_file_exists"]]
    print("Data files audited: {}".format(len(data_rows)))
    print("Report rows: {}".format(len(report)))
    print("Report saved to: {}".format(output_path))
    for flag in FLAG_COLUMNS:
        count = int(data_rows[flag].sum())
        print("{}: {} files".format(flag, count))
    for column in (
        "duplicate_timestamps",
        "missing_timestamps",
        "invalid_timestamps",
        "timestamp_processing_errors",
        "out_of_order_records",
        "missing_values",
        "invalid_values",
        "non_finite_values",
        "potential_time_gaps",
        "invalid_anomaly_windows",
        "windows_outside_data",
    ):
        print("{}: {} records/intervals".format(
            column, int(data_rows[column].sum())
        ))
    print(data_rows.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
