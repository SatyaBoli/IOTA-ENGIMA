"""Causal, robust Isolation Forest scoring for a single time series.

Run with:
    python iforest.py path\\to\\readings.csv --value-column value
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLUMNS = ("robust_z", "robust_deviation", "robust_difference")
SCALE_FLOOR = 1e-3
MAD_NORMALIZATION = 1.4826


def build_causal_features(
    observed: pd.Series,
    window: int,
    min_periods: int,
    scale_floor: float = SCALE_FLOOR,
) -> pd.DataFrame:
    """Build robust residual features using past observations only."""
    past = observed.shift(1)
    baseline_median = past.rolling(
        window=window, min_periods=min_periods
    ).median()
    rolling_mad = past.rolling(window=window, min_periods=min_periods).apply(
        lambda values: np.median(np.abs(values - np.median(values))),
        raw=True,
    )
    local_scale = (MAD_NORMALIZATION * rolling_mad).clip(lower=scale_floor)

    return pd.DataFrame(
        {
            "baseline_median": baseline_median,
            "robust_z": (observed - baseline_median) / local_scale,
            "median_residual": observed - baseline_median,
            "first_difference": observed.diff(),
        },
        index=observed.index,
    )


def score_series(
    frame: pd.DataFrame,
    value_column: str,
    timestamp_column: str | None,
    series_id: str,
    window: int = 50,
    min_periods: int = 5,
    train_fraction: float = 0.7,
    contamination: float | str = "auto",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train on the chronological prefix and return scored holdout rows."""
    if value_column not in frame.columns:
        raise ValueError(f"CSV is missing value column {value_column!r}.")
    if timestamp_column is not None and timestamp_column not in frame.columns:
        raise ValueError(f"CSV is missing timestamp column {timestamp_column!r}.")
    if window < min_periods or min_periods < 2:
        raise ValueError("window must be at least min_periods, and min_periods >= 2.")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if isinstance(contamination, str):
        if contamination != "auto":
            raise ValueError("contamination must be 'auto' or a number in (0, 0.5].")
    elif not 0.0 < contamination <= 0.5:
        raise ValueError("numeric contamination must be in (0, 0.5].")
    if frame.empty:
        raise ValueError("CSV contains no readings.")

    observed = pd.to_numeric(frame[value_column], errors="coerce")
    if observed.isna().any() or not np.isfinite(observed.to_numpy()).all():
        raise ValueError(f"Column {value_column!r} must contain only finite numbers.")

    split = int(len(frame) * train_fraction)
    if split <= min_periods or split >= len(frame):
        raise ValueError(
            "The CSV is too short for the requested train split and baseline window."
        )

    # Gate 1 — causal baseline: shift before calculating rolling median and MAD.
    features = build_causal_features(observed, window, min_periods)

    # Gate 2 — train-only scaling: derive feature scales from the training prefix.
    training_values = observed.iloc[:split]
    training_center = float(training_values.median())
    training_mad = float((training_values - training_center).abs().median())
    training_scale = max(MAD_NORMALIZATION * training_mad, SCALE_FLOOR)

    model_features = pd.DataFrame(
        {
            "robust_z": features["robust_z"],
            "robust_deviation": features["median_residual"] / training_scale,
            "robust_difference": features["first_difference"] / training_scale,
        },
        index=frame.index,
    )

    # Gate 3 — chronological holdout: fit only on valid features before the split.
    train_features = model_features.iloc[:split].dropna()
    test_features = model_features.iloc[split:]
    if len(train_features) < 2:
        raise ValueError("Not enough valid past-only rows to train the detector.")
    if test_features.isna().any().any():
        raise ValueError("Holdout features are incomplete; increase the input history.")

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(train_features.loc[:, FEATURE_COLUMNS])
    scores = -model.decision_function(test_features.loc[:, FEATURE_COLUMNS])
    predictions = model.predict(test_features.loc[:, FEATURE_COLUMNS])

    # Gate 4 — evaluator contract: emit one consistent row per holdout observation.
    result = pd.DataFrame(
        {
            "series_id": series_id,
            "timestamp": (
                frame[timestamp_column].iloc[split:].to_numpy()
                if timestamp_column
                else frame.index[split:].to_numpy()
            ),
            "observed": observed.iloc[split:].to_numpy(),
            "baseline_median": features["baseline_median"].iloc[split:].to_numpy(),
            "robust_z": features["robust_z"].iloc[split:].to_numpy(),
            "score": scores,
            "label": np.where(predictions == -1, "ANOMALOUS", "NORMAL"),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a time-series CSV with a causal robust Isolation Forest."
    )
    parser.add_argument("input_csv", type=Path, help="Input CSV, ordered oldest to newest.")
    parser.add_argument(
        "--value-column",
        default="auto",
        help="Numeric observation column, or 'auto' to detect value/reading.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="auto",
        help="Timestamp column, 'auto' to detect one, or 'none' to use row numbers.",
    )
    parser.add_argument(
        "--series-id",
        default=None,
        help="Series identifier in output (default: input filename stem).",
    )
    parser.add_argument("--window", type=int, default=50, help="Past-only window size.")
    parser.add_argument(
        "--min-periods",
        type=int,
        default=5,
        help="Minimum past observations required for a baseline.",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.7,
        help="Chronological training fraction (default: 0.7).",
    )
    parser.add_argument(
        "--contamination",
        default="auto",
        help="Isolation Forest contamination: 'auto' or a float in (0, 0.5].",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results.csv"),
        help="Holdout results path (default: results.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contamination: float | str = args.contamination
    if contamination != "auto":
        try:
            contamination = float(contamination)
        except ValueError as exc:
            raise SystemExit("--contamination must be 'auto' or a number.") from exc

    frame = pd.read_csv(args.input_csv)
    normalized_columns = {
        str(name).strip().lower(): name for name in frame.columns
    }
    if args.value_column.lower() == "auto":
        value_column = next(
            (
                normalized_columns[name]
                for name in ("value", "reading")
                if name in normalized_columns
            ),
            args.value_column,
        )
    else:
        value_column = normalized_columns.get(
            args.value_column.strip().lower(), args.value_column
        )

    if args.timestamp_column.lower() == "none":
        timestamp_column = None
    elif args.timestamp_column.lower() == "auto":
        timestamp_column = next(
            (
                normalized_columns[name]
                for name in ("timestamp", "time", "datetime", "date")
                if name in normalized_columns
            ),
            None,
        )
    else:
        timestamp_column = normalized_columns.get(
            args.timestamp_column.strip().lower(), args.timestamp_column
        )
    result = score_series(
        frame=frame,
        value_column=value_column,
        timestamp_column=timestamp_column,
        series_id=args.series_id or args.input_csv.stem,
        window=args.window,
        min_periods=args.min_periods,
        train_fraction=args.train_fraction,
        contamination=contamination,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output_csv, index=False)
    anomaly_count = int((result["label"] == "ANOMALOUS").sum())
    print(
        f"Saved {len(result)} holdout rows ({anomaly_count} anomalous) "
        f"to {args.output_csv}"
    )


if __name__ == "__main__":
    main()
