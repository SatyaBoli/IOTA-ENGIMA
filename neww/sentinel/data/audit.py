"""Automated dataset schema audit and anomaly profile inspection."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from .loader import NABDataLoader


def audit_dataset(loader: NABDataLoader) -> dict[str, Any]:
    """Perform comprehensive data quality and structure audit across all benchmark series."""
    df = loader.load_df()
    gt_windows = loader.get_ground_truth_windows_by_series_id()

    series_list = loader.list_series()
    total_rows = len(df)
    unique_series = len(series_list)

    total_gt_windows = sum(len(w) for w in gt_windows.values())
    series_with_anomalies = sum(1 for s in series_list if s.has_anomaly)

    # Check data cleanliness
    missing_vals = int(df["value"].isna().sum())
    infinite_vals = int(np.isinf(df["value"]).sum())

    categories_count = df["category"].value_counts().to_dict()

    series_stats = []
    for s in series_list:
        sub = df[df["series_id"] == s.series_id]
        vals = sub["value"].values
        windows = gt_windows.get(s.series_id, [])
        series_stats.append({
            "series_id": s.series_id,
            "category": s.category,
            "series_name": s.series_name,
            "rows": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "anomaly_windows_count": len(windows),
        })

    report = {
        "dataset_name": "NAB_AI_Training_Dataset",
        "total_rows": total_rows,
        "total_series": unique_series,
        "series_with_anomalies": series_with_anomalies,
        "total_ground_truth_windows": total_gt_windows,
        "missing_values": missing_vals,
        "infinite_values": infinite_vals,
        "categories_breakdown": categories_count,
        "series_summary": series_stats[:10],
    }
    return report


if __name__ == "__main__":
    loader = NABDataLoader()
    rep = audit_dataset(loader)
    print(json.dumps(rep, indent=2))
