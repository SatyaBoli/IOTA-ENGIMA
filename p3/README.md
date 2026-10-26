# Sentinel P3 — Research-Informed Adaptation Layer (v2)

P3 implements adaptive score transformation, drift signalling, hysteresis/event state, guarded detector updates, and auditable evidence. It is designed to sit between a P2 detector implementing `DetectorProtocol` and the P1/P4 evaluator/output adapter. It is **not** the complete competition submission.

## Quick start

```bash
python -m pytest -q
python -m compileall -q sentinel tests
```

The core is dependency-light. PyYAML is only needed for `AdaptationConfig.from_yaml`; tests skip YAML checks if unavailable.

## Important contracts

- P2 raw scores must be finite, non-negative, and higher means more anomalous. Invalid scores fail closed and do not update adaptive state.
- The detector's `score_one` must be causal and must not mutate its state. `update_state` is called after the current point has been scored and classified.
- `event_id` identifies one contiguous predicted anomaly span. `incident_id` can group spans separated by a cooldown gap. A NORMAL gap remains NORMAL; labels are never revised retrospectively.
- Likelihood mode is research-inspired and not an exact reproduction of Ahmad et al. Its thresholds require chronological validation.
- Drift is detected by a custom bounded-window **ADWIN-style** implementation. Do not claim equivalence to ADWIN2 guarantees without a reference comparison.
- Internal records include additive evidence fields. P4 must map them to the exact organizer schema and must not assume these internal fields are accepted as submission output.

## Documentation

- `P3_IMPROVEMENTS.md` — code-level changes, compatibility notes, verification, and remaining gates.
- `RESEARCH_AND_VALIDATION.md` — research basis, source-derived requirements vs engineering decisions, and recommended experiments.
- `configs/baseline.yaml` — default configuration.

## Research and performance caveat

The test suite is synthetic/regression validation only. No official organizer data/evaluator is included here; no official event-F1 or leaderboard improvement is claimed.
