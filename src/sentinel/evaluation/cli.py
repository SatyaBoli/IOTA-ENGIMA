"""Command-line entry point for saved-prediction evaluation."""

import argparse
import json
import math
import sys
from pathlib import Path

from .metrics import evaluate_predictions, read_labels, read_predictions


def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate saved timestamped anomaly scores against labeled "
            "anomaly windows. This evaluator uses provisional event-F1 rules."
        )
    )
    parser.add_argument(
        "predictions",
        type=Path,
        help="Prediction CSV file or directory of CSV files",
    )
    parser.add_argument(
        "labels",
        type=Path,
        help="JSON object mapping series IDs to inclusive [start, end] windows",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Finite score threshold; scores greater than or equal to it alert",
    )
    parser.add_argument(
        "--series-id",
        help="Series ID for a single CSV that has no series_id column",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the JSON report (also printed to stdout)",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not math.isfinite(args.threshold):
        parser.error("--threshold must be finite")

    try:
        label_windows = read_labels(args.labels)
        predictions = read_predictions(
            args.predictions, label_windows, series_id=args.series_id
        )
        report = evaluate_predictions(
            predictions, label_windows, threshold=args.threshold
        ).to_dict()
        rendered = json.dumps(
            report, indent=2, sort_keys=True, allow_nan=False
        ) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        sys.stdout.write(rendered)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 0
