"""Organizer submission adapter and schema validator."""
from __future__ import annotations

import json
import math
import os
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ..contracts import LABEL_ANOMALOUS, LABEL_NORMAL, PredictionRecord
from ..explanations.explainer import generate_explanation


class SubmissionAdapter:
    """Formats internal PredictionRecords into standardized organizer submission files."""

    @staticmethod
    def to_dataframe(records: Sequence[PredictionRecord]) -> pd.DataFrame:
        rows = []
        for r in records:
            expl = generate_explanation(r)
            rows.append({
                "series_id": str(r["series_id"]),
                "timestamp": str(r["timestamp"]),
                "anomaly_score": float(r["anomaly_score"]),
                "label": str(r["label"]),
                "threshold": float(r["threshold"]),
                "event_id": str(r["event_id"]) if r.get("event_id") else "",
                "detector_version": str(r["detector_version"]),
                "reason_codes": ";".join(r.get("reason_codes", [])),
                "explanation": expl["narrative"],
            })
        return pd.DataFrame(rows)

    @classmethod
    def write_csv(cls, records: Sequence[PredictionRecord], output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df = cls.to_dataframe(records)
        df.to_csv(output_path, index=False)
        print(f"[SubmissionAdapter] Wrote {len(df)} records to CSV: {output_path}")

    @classmethod
    def write_json(cls, records: Sequence[PredictionRecord], output_path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        serializable = []
        for r in records:
            expl = generate_explanation(r)
            item = dict(r)
            item["explanation"] = expl
            serializable.append(item)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        print(f"[SubmissionAdapter] Wrote {len(serializable)} records to JSON: {output_path}")

    @staticmethod
    def validate_submission_file(csv_path: str) -> dict[str, Any]:
        """Validate submission file against organizer evaluation contract."""
        df = pd.read_csv(csv_path)

        required_cols = {"series_id", "timestamp", "anomaly_score", "label", "threshold"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Submission missing required columns: {missing}")

        # Check finite scores
        nan_scores = int(df["anomaly_score"].isna().sum())
        inf_scores = int(np.isinf(df["anomaly_score"]).sum()) if hasattr(df["anomaly_score"], "values") else 0
        if nan_scores > 0 or inf_scores > 0:
            raise ValueError(f"Found non-finite anomaly scores: {nan_scores} NaNs, {inf_scores} Infs")

        # Check valid labels
        unique_labels = set(df["label"].unique())
        invalid_labels = unique_labels - {LABEL_NORMAL, LABEL_ANOMALOUS}
        if invalid_labels:
            raise ValueError(f"Found invalid labels: {invalid_labels}")

        anomalies_count = int((df["label"] == LABEL_ANOMALOUS).sum())
        total_rows = len(df)

        return {
            "valid": True,
            "total_rows": total_rows,
            "anomalous_rows": anomalies_count,
            "anomaly_rate": round(anomalies_count / max(total_rows, 1) * 100, 2),
            "series_count": int(df["series_id"].nunique()),
        }
