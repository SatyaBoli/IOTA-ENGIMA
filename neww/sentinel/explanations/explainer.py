"""Deterministic, human-interpretable evidence generator for anomaly alerts.

Translates mathematical anomaly outputs and detector consensus into defensible explanations.
"""
from __future__ import annotations

from typing import Any, Mapping


def generate_explanation(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build a transparent evidence package for a flagged prediction."""
    evidence = record.get("evidence", {})
    obs = float(evidence.get("observed", 0.0))
    base = float(evidence.get("baseline", 0.0))
    res = float(evidence.get("residual", obs - base))
    score = float(record.get("anomaly_score", 0.0))
    thresh = float(record.get("threshold", 3.5))
    codes = list(record.get("reason_codes", []))
    contributors = list(evidence.get("contributors", []))
    drift = bool(evidence.get("drift_detected", False))

    pct_diff = ((obs - base) / abs(base) * 100.0) if abs(base) > 1e-4 else 0.0
    direction = "spike" if res > 0 else "drop"

    summary = (
        f"Value deviated by {abs(res):.2f} ({pct_diff:+.1f}%) from expected baseline {base:.2f}. "
        f"Anomaly score reached {score:.2f} (applied threshold: {thresh:.2f})."
    )

    if drift:
        summary += " ADWIN detected distribution drift; baseline guard relaxed for re-adaptation."

    return {
        "series_id": record.get("series_id"),
        "timestamp": record.get("timestamp"),
        "direction": direction,
        "observed_value": obs,
        "expected_baseline": base,
        "residual": res,
        "percent_deviation": round(pct_diff, 2),
        "score": score,
        "threshold": thresh,
        "top_contributors": contributors,
        "reason_codes": codes,
        "narrative": summary,
    }
