# Sentinel evaluation protocol

## Verification status

This checkout contains the Numenta Anomaly Benchmark (NAB), not a Sentinel
competition statement, official evaluator, or competition submission
specification. Searches of the repository found NAB's `nab/sweeper.py`,
`nab/scorer.py`, NAB labels, configuration, and detector-result CSVs, but no
Sentinel event-F1 rules or official event evaluator. The available NAB scorer
uses threshold sweeps and a sigmoid-weighted score for each labeled window;
its semantics are **not** equivalent to event-based F1. The new evaluator does
not modify or claim compliance with NAB scoring.

Consequently, the event-based rules below are an explicit **provisional local
protocol**, not verified competition behavior. Obtain organizer guidance and
replace or configure these semantics before using the results as an official
competition score.

## Provisional prediction input

Input is one CSV or a directory of CSV files. A CSV has:

| Column | Required | Meaning |
| --- | --- | --- |
| `timestamp` | Yes | ISO-8601 timestamp; naive timestamps are interpreted as UTC and timezone-bearing timestamps are normalized to UTC. |
| `score` or `anomaly_score` | Yes | Finite numeric anomaly score; larger values mean more anomalous. |
| `series_id` | Required for combined/multi-series CSVs | Must match a series key in the labels JSON. A one-series file may use `--series-id`; NAB result paths can be inferred when the match is unambiguous. |

Rows with the same normalized series ID and timestamp are coalesced to their
maximum score (the provisional larger-is-more-anomalous direction); they form
at most one alert. Additional columns (including NAB's `label` column) are
ignored and never used as ground truth. Labels are
read separately from a JSON object mapping series IDs to lists of inclusive
`[start, end]` timestamp pairs, as in NAB's `labels/combined_windows.json`.
Every prediction series must have an explicit label entry, even if its window
list is empty. Label series without prediction rows are evaluated as having no
alerts, and their events count as missed. No data values or hidden labels are
loaded by this evaluator.

## Provisional event rules

1. Windows include both endpoints. Overlapping windows are merged before
   scoring so one alert cannot count multiple overlapping ground-truth events.
2. A prediction is an alert exactly when `score >= --threshold`. The threshold
   is supplied explicitly; this command does not optimize it on the evaluation
   labels.
3. An event is detected by the earliest alert within its window. Each event
   contributes at most one TP. Further alerts in that same event are
   duplicates and are ignored, not counted as false alarms.
4. Each unique alert timestamp outside all event windows is one false alarm.
   In particular, alerts before an event or after its end are false alarms.
5. Detection delay is elapsed seconds from event start to its earliest
   in-window alert. Reported mean delay averages detected events only.
6. Precision and recall are `null` when their denominator is zero. F1 is
   `null` if both precision and recall are undefined, and otherwise uses the
   harmonic mean (zero when either defined component is zero). Aggregate
   precision/recall/F1 are micro-averaged from pooled event and false-alarm
   counts; `macro_f1` averages defined per-series F1 values.
7. Runtime is wall-clock time spent validating and calculating metrics in
   `evaluate_predictions`; CSV/JSON file I/O and report writing are excluded.
   Runtime is informational and can differ between identical runs; metric
   fields and their ordering are deterministic.

These choices (including endpoint inclusivity, overlap merging, duplicate
handling, matching, score direction, and delay units) are not confirmed by
competition documentation.

## Run

From the repository root on Windows PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m sentinel.evaluation .\predictions.csv .\labels.json --threshold 0.5 --output .\evaluation-report.json
```

Or evaluate saved NAB results, using matching relative paths and
`anomaly_score` columns. NAB aggregate `*_scores.csv` summary files are skipped:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src).Path
python -m sentinel.evaluation .\results\numenta .\labels\combined_windows.json --threshold 0.5 --output .\evaluation-report.json
```

For a single CSV without a `series_id` column and without an inferable NAB-style
path, pass `--series-id <label-key>`. For a multi-series CSV, include
`series_id,timestamp,score` columns. The command prints JSON and optionally
writes the same report to `--output`.

Run tests with:

```powershell
python -m pytest
```

## Questions requiring organizer clarification

- What is the official evaluator and exact definition of event precision,
  recall, and F1?
- Are event windows and detection timestamps inclusive at either or both
  boundaries? How are overlapping or adjacent events treated?
- Can one alert match more than one event? Are additional alerts in an event
  ignored, penalized, or collapsed into alert episodes?
- Are alerts matched to the first event, nearest event, or by a global
  one-to-one assignment?
- What is the official detection-delay origin, unit, aggregation, and
  treatment of detections before event onset?
- What prediction schema, score direction, threshold-selection procedure, and
  handling of absent timestamps/series are required?
- What are the prescribed data splits, and which labels may be used for
  threshold selection versus final evaluation?
- What official report/submission fields and runtime-measurement boundaries
  are required?

Until these are answered, reported event metrics are useful only for local
development and must not be described as official scores.
