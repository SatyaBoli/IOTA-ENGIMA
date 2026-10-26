# P3 Research and Validation Report — Sentinel / Utkrishta Challenge 2

## Executive assessment

The P3 layer is a useful modular subsystem, not a complete anomaly detector or competition submission by itself. It takes P2's causal score and controls thresholds, event state, drift signals, baseline updates, and evidence. The research-backed improvement priority is correctness and auditability before adding more algorithms.

The most consequential integration bug addressed in this revision is score-space mismatch in likelihood mode: the accepted-normal reference must be updated with raw detector scores, not the transformed z-score. The most consequential semantics change is splitting contiguous `event_id` from merged `incident_id` so cooldown grouping does not falsely imply that a discontinuous event range is contiguous.

## Evidence basis and boundaries

**From the supplied Sentinel playbook:** P3 owns threshold calibration, drift strategy, event grouping, contamination prevention, and machine-verifiable evidence. The recommended causal order is score using past state, classify, emit, then update state under a documented guard. The playbook calls for causal tests, drift/contamination tests, interleaved-series isolation, finite output, event-F1, delay, and runtime reporting.

**From the supplied P3 research mapping:** modified z-score, anomaly likelihood, ADWIN, rigorous evaluation, and range-based evaluation were selected as design influences. The original mapping explicitly noted that the TSAD-Evaluator cross-check was not completed and the citations were not re-checked against original sources.

**From the deep research synthesis / external research anchors:** mature projects support modular detector/post-processing interfaces and replay/auditability; streaming anomaly benchmarks emphasize timeliness; range-aware evaluation is materially different from pointwise F1; and the original ADWIN method should not be conflated with a custom bounded-window approximation.

**Engineering recommendations in this code:** separate event and incident IDs; preserve the first exceedance timestamp for confirmed events; use strict JSON-safe evidence; fail closed on invalid P2 outputs; and use numerically stable reference statistics. These are implementation decisions motivated by the sources, not claims that any paper specifies this exact code.

## Implementation changes

See `P3_IMPROVEMENTS.md` for the complete itemized changelog. The updated package keeps the dependency-light custom drift implementation but labels it ADWIN-style. It does not silently introduce River or a new detector, since those choices should be validated against the competition environment and allowed dependencies.

## Verification

The package regression suite reports **55 passing tests** in the local environment. The suite includes stable likelihood input, sustained likelihood score increases, invalid values and strict JSON serialization, guarded baseline behavior, slow creep, abrupt regime shift, random-score sanity, per-series isolation, replay reset, causal prefix invariance, cooldown semantics, and confirmation-time evidence.

This is software verification on tests and synthetic streams—not a benchmark result. No official organizer data, official evaluator implementation, or permitted validation split was included with the package. Therefore this report does not claim improved official event-F1, NAB score, leaderboard rank, or production readiness.

## Required next experiments

1. Run the organizer's exact evaluator on allowed chronological validation data.
2. Compare custom ADWIN-style drift with `river.drift.ADWIN` on identical synthetic streams: stationary noise, abrupt mean shift, slow ramp, and transient spike. Report false alarms and detection delay.
3. Ablate fixed vs adaptive threshold, guard on/off, cooldown grouping on/off, and drift on/off. Keep components only when the trade-off is justified.
4. Validate the meaning of `event_id` and `incident_id` with P1/P4. The output adapter must emit exactly the official schema; the additive internal field should not leak into a strict competition file unless allowed.
5. Tune all parameters only on allowed chronological validation data and freeze the final configuration before hidden-test inference.

## Research links

- Numenta Anomaly Benchmark: https://github.com/numenta/NAB
- River ADWIN documentation: https://riverml.xyz/dev/api/drift/ADWIN/
- Salesforce Merlion: https://github.com/salesforce/Merlion
- Etsy Skyline: https://github.com/earthgecko/skyline
- LinkedIn ThirdEye: https://www.linkedin.com/blog/engineering/analytics/introducing-thirdeye-linkedins-business-wide-monitoring-platform
- Tatbul et al., range-based precision/recall: https://people.csail.mit.edu/tatbul/publications/NeurIPS18.pdf
- TSB-AD benchmark: https://thedatumorg.github.io/TSB-AD/
