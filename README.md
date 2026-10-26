# IOTA-ENGIMA Sentinel

Sentinel is a local dashboard for exploring time-series anomaly detection. It offers three detectors:

- **Sentinel Adaptive (P3):** causal robust-MAD scoring with warmup, hysteresis, event and incident tracking, guarded baseline updates, and drift signals.
- **Rolling Z-Score:** compares each reading with the previous 50 readings and a configurable standard-deviation threshold.
- **Isolation Forest:** trains on the chronological first 70% of the CSV and reports scores for the final 30% holdout.

Detector scores have different meanings and should not be compared directly. The included sensitivity profiles are not calibrated ground-truth settings. Validate and tune thresholds using permitted labeled chronological data.

## Run locally

Requires Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:8001/> and upload a CSV containing a numeric `value` or `reading` column. A `timestamp` and `sensor` column are optional. CSV data is processed locally.

## Validate

```powershell
python -m pytest -q testiforest.py p3\tests
```

The suite covers detector causality, output contracts, P3 event/adaptation behavior, and app integration. The P3 research and remaining validation gates are documented in [`p3/RESEARCH_AND_VALIDATION.md`](p3/RESEARCH_AND_VALIDATION.md).
