import csv
import io
import json
import math
import os
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from iforest import score_series
from iforest_detector import RollingZScoreDetector
from p3.sentinel.adaptation import AdaptationLayer
from p3.sentinel.config import AdaptationConfig
from p3.sentinel.contracts import Point
from p3_detector import CausalMadDetector


MAX_REQUEST_BYTES = 10 * 1024 * 1024
INDEX_FILE = Path(__file__).with_name("index.html")
ADAPTATION_CONFIG = Path(__file__).with_name("p3") / "configs" / "baseline.yaml"
P3_SENSITIVITY = {
    "moderate": (2.5, 1.5),
    "high": (2.0, 1.5),
    "conservative": (3.5, 2.0),
}


def analyze_csv(
    contents,
    model_name="sentinel_adaptive",
    threshold=2.0,
    p3_sensitivity="moderate",
):
    reader = csv.DictReader(io.StringIO(contents))
    if not reader.fieldnames:
        raise ValueError("The CSV must include a header row.")

    columns = {name.strip().lower(): name for name in reader.fieldnames if name}
    value_column = columns.get("value") or columns.get("reading")
    if value_column is None:
        raise ValueError("The CSV needs a numeric 'value' column.")

    timestamp_column = next(
        (columns[name] for name in ("timestamp", "time", "datetime", "date") if name in columns),
        None,
    )
    sensor_column = columns.get("sensor")
    records = []
    values = []

    for row_number, row in enumerate(reader, start=2):
        raw_value = row.get(value_column)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Row {row_number} has a non-numeric value.") from exc
        if not math.isfinite(value):
            raise ValueError(f"Row {row_number} has a non-finite value.")
        values.append(value)
        records.append(
            {
                "timestamp": (row.get(timestamp_column) or "").strip()
                if timestamp_column
                else f"Row {row_number - 1}",
                "sensor": (row.get(sensor_column) or "").strip()
                if sensor_column
                else "—",
                "value": value,
            }
        )

    if not values:
        raise ValueError("The CSV contains no readings.")

    started = time.perf_counter()
    p3_predictions = None
    if model_name == "sentinel_adaptive":
        if p3_sensitivity not in P3_SENSITIVITY:
            raise ValueError("Choose a valid P3 sensitivity profile.")
        enter_threshold, exit_threshold = P3_SENSITIVITY[p3_sensitivity]
        detector = CausalMadDetector()
        config = replace(
            AdaptationConfig.from_yaml(str(ADAPTATION_CONFIG)),
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
        )
        layer = AdaptationLayer(detector, detector.init_state, config)
        p3_predictions = [
            layer.process(
                Point(
                    series_id=record["sensor"] if record["sensor"] != "—" else "series-1",
                    timestamp=record["timestamp"],
                    value=record["value"],
                )
            )
            for record in records
        ]
        for record, prediction in zip(records, p3_predictions):
            record["score"] = prediction["anomaly_score"]
            record["label"] = prediction["label"]
            record["series_id"] = prediction["series_id"]
            record["threshold"] = prediction["threshold"]
            record["event_id"] = prediction["event_id"]
            record["incident_id"] = prediction["incident_id"]
            record["reason_codes"] = prediction["reason_codes"]
            record["detector_version"] = prediction["detector_version"]
            record["evidence"] = prediction["evidence"]
            record["baseline_median"] = prediction["evidence"]["baseline"]
            record["residual"] = prediction["evidence"]["residual"]
            record["raw_score"] = prediction["evidence"]["raw_score"]
            record["baseline_update_accepted"] = prediction["evidence"][
                "baseline_update_accepted"
            ]
            record["event_start_timestamp"] = prediction["evidence"][
                "event_start_timestamp"
            ]
    elif model_name == "isolation_forest":
        frame = pd.DataFrame(records)
        scored = score_series(
            frame=frame,
            value_column="value",
            timestamp_column="timestamp",
            series_id=records[0]["sensor"] if records[0]["sensor"] != "—" else "series-1",
            contamination="auto",
        )
        results = []
        for row in scored.to_dict(orient="records"):
            results.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "sensor": records[0]["sensor"],
                    "value": float(row["observed"]),
                    "score": float(row["score"]),
                    "label": row["label"],
                    "baseline_median": float(row["baseline_median"]),
                    "robust_z": float(row["robust_z"]),
                    "series_id": row["series_id"],
                    "scoring_scope": "chronological_holdout",
                }
            )
        return {
            "results": results,
            "predictions": None,
            "model": model_name,
            "threshold": None,
            "input_readings": len(records),
            "scored_readings": len(results),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
    elif model_name == "rolling_zscore":
        if not math.isfinite(threshold) or not 1.0 <= threshold <= 6.0:
            raise ValueError("Rolling Z-Score threshold must be between 1.0 and 6.0.")
        detector = RollingZScoreDetector(threshold=threshold)
    else:
        raise ValueError("Choose Sentinel Adaptive, Isolation Forest, or Rolling Z-Score.")

    if model_name != "sentinel_adaptive":
        scores = detector.score(values)
        labels = detector.predict(values)
        for record, score, prediction in zip(records, scores, labels):
            record["score"] = float(score)
            record["label"] = "ANOMALOUS" if prediction == -1 else "NORMAL"

    return {
        "results": records,
        "predictions": p3_predictions,
        "model": model_name,
        "threshold": threshold if model_name == "rolling_zscore" else None,
        "sensitivity": p3_sensitivity if model_name == "sentinel_adaptive" else None,
        "enter_threshold": (
            P3_SENSITIVITY[p3_sensitivity][0]
            if model_name == "sentinel_adaptive"
            else None
        ),
        "exit_threshold": (
            P3_SENSITIVITY[p3_sensitivity][1]
            if model_name == "sentinel_adaptive"
            else None
        ),
        "input_readings": len(records),
        "scored_readings": len(records),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
    }


class SentinelHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        if path not in ("/", "/index.html"):
            self.send_error(404, "Not found")
            return

        try:
            page = INDEX_FILE.read_bytes()
        except OSError:
            self.send_error(500, "Could not read index.html")
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def do_POST(self):
        if urlparse(self.path).path != "/api/detect":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._send_json(400, {"error": "A valid Content-Length header is required."})
            return
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "CSV must be between 1 byte and 10 MB."})
            return

        try:
            contents = self.rfile.read(length).decode("utf-8-sig")
            model_name = self.headers.get("X-Detector-Model", "sentinel_adaptive")
            try:
                threshold = float(self.headers.get("X-Z-Threshold", "2.0"))
            except ValueError as exc:
                raise ValueError("Rolling Z-Score threshold must be a number.") from exc
            sensitivity = self.headers.get("X-P3-Sensitivity", "moderate")
            result = analyze_csv(contents, model_name, threshold, sensitivity)
        except (UnicodeDecodeError, csv.Error, ValueError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        self._send_json(200, result)

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print("%s - %s" % (self.address_string(), format % args))


if __name__ == "__main__":
    port = int(os.environ.get("SENTINEL_PORT", "8001"))
    address = ("127.0.0.1", port)
    print(f"Sentinel is running at http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, SentinelHandler).serve_forever()
