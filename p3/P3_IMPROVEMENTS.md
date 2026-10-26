# Sentinel P3 Adaptation Layer — Research-Informed Change Log (v2)

**Scope:** P3 only: threshold adaptation, drift signalling, event semantics, guarded updates, and deterministic evidence. P2's detector and P1/P4's official evaluator/output adapter remain external contracts.

## Implemented in this version

1. **Likelihood reference consistency.** The long-run accepted-normal reference window learns from P2's raw anomaly scores, never from the transformed likelihood z-score. Invalid scores are excluded from both short and long windows.
2. **Numerically stable reference statistics.** Likelihood mean/variance are recomputed from the bounded reference window using `math.fsum` and centered squared deviations. This avoids cancellation in `E[x²] - E[x]²` on long-running or offset score streams. Window bounds keep the computation predictable.
3. **Fail-closed score and evidence handling.** Non-finite/unparseable scores are tagged `NON_FINITE_SCORE`; negative values violate P2's non-negative score contract and are tagged `INVALID_RAW_SCORE`. Invalid evidence is sanitized to strict-JSON-safe values and prevents baseline, likelihood-reference, and drift updates.
4. **Warm-up contract validation.** Only a real boolean `context['is_warmup']` is accepted. Truthy strings or integers cannot silently suppress alerts.
5. **Causal drift input.** The bounded custom detector receives only valid post-warmup scores. The implementation is explicitly documented as **ADWIN-style**, not claimed to be a byte-for-byte or guarantee-equivalent implementation of ADWIN2. Drift remains a signal, not proof that a regime is normal.
6. **Guarded adaptation remains explicit.** Normal, low-score observations can update the detector state under the configured policy; anomalies, invalid scores/evidence, and cooldown-tail observations are rejected. A prolonged anomalous run can trigger the configured regime-rebaseline backstop, which is visible through reason codes.
7. **Event and incident IDs now have distinct meanings.** `event_id` is unique per contiguous anomalous span. A cooldown re-entry gets a new `event_id` but retains the same `incident_id`; after the cooldown expires, a later span gets a new incident. This prevents disjoint spans from being misrepresented as one contiguous event.
8. **Detection-delay evidence improved.** For `min_event_length > 1`, the event start timestamp points to the first exceedance in the confirmed consecutive run, not the later confirmation point. A cooldown-reentered event records its own start separately from the incident's original start.
9. **Auditable prediction evidence.** Records retain raw score, score mode, threshold, update acceptance, drift-window width, likelihood tail statistic when available, event start, incident ID, and incident start. Tail probability is a Gaussian-tail statistic, not a calibrated probability that the observation is anomalous.
10. **Replay reset and strict config validation.** Per-series reset removes only that series state; full reset clears all states. Invalid window sizes, threshold relationships, drift parameters, and unknown config keys fail early.
11. **Regression coverage expanded.** Added/updated tests for unique contiguous event IDs, incident grouping through cooldown, new incidents after cooldown expiry, and correct first-exceedance timestamps after confirmation. Existing tests cover likelihood integration, sustained score increases, random-score sanity, invalid evidence/strict JSON, replay reset, contamination, slow creep, abrupt regime shifts, and per-series isolation.

## Compatibility note

`incident_id` is an additive internal record field, and event IDs across cooldown merges intentionally changed. P4 should map only the fields accepted by the organizer's required schema. P1 must verify whether the official evaluator expects contiguous ranges or intentionally merged incidents before submission. Do not silently pass the internal record directly to the competition output writer.

## Verification performed for v2

- `python -m pytest -q`: **55 passed**.
- `python -m compileall -q sentinel tests`: required as the final package check.
- Synthetic and regression tests only; the official organizer dataset/evaluator were not supplied with this package. No leaderboard or official event-F1 improvement is claimed.

## Remaining research/validation gates

- Compare the custom bounded ADWIN-style detector with `river.drift.ADWIN` on the same permitted synthetic and chronological validation streams; report false-drift rate and detection delay before making equivalence claims.
- Run the organizer's exact event-F1 evaluator, including hand-worked edge cases, against the new event/incident semantics.
- Tune enter/exit thresholds, cooldown, regime patience, and drift parameters only on allowed chronological validation data.
- Run ablations: fixed vs adaptive, guard vs no guard, cooldown merge on/off, and custom drift vs reference drift.
- Keep public NAB scoring separate from the organizer's event-F1; they are not interchangeable.

## Research anchors

- Iglewicz & Hoaglin, *How to Detect and Handle Outliers* (1993): modified z-score and the 3.5 outlier rule. The 2.0 exit threshold is a hysteresis engineering choice, not a claim that the original source prescribes this event-state policy.
- Ahmad, Lavin, Purdy & Agha (2017), *Unsupervised real-time anomaly detection for streaming data*: motivation for online anomaly likelihood. This code's rolling accepted-normal reference and z-score are research-inspired, not an exact reproduction.
- Bifet & Gavaldà (2007), *Learning from time-changing data with adaptive windowing*: motivation for adaptive-window drift detection. The custom implementation here is ADWIN-style; compare with the maintained reference implementation before claiming original ADWIN2 guarantees.
- Tatbul et al. (NeurIPS 2018), *Precision and Recall for Time Series*: motivation for range-aware evaluation. The event state machine is not itself a range-based evaluator.
- Kim et al. (AAAI 2022), *Towards a Rigorous Evaluation of Time-Series Anomaly Detection*: motivation for avoiding point-adjusted evaluation and retaining random-score sanity tests.
- Numenta Anomaly Benchmark: streaming benchmark and timeliness-oriented scoring; do not substitute its score for the organizer's event-F1.
- Engineering inspiration in the team's Sentinel playbook: ThirdEye's separation of detection and alert handling, Skyline's alert history, and Merlion's modular model/post-processing interfaces.

## References

- NAB: https://github.com/numenta/NAB
- River ADWIN API: https://riverml.xyz/dev/api/drift/ADWIN/
- Salesforce Merlion: https://github.com/salesforce/Merlion
- Etsy Skyline: https://github.com/earthgecko/skyline
- LinkedIn ThirdEye: https://www.linkedin.com/blog/engineering/analytics/introducing-thirdeye-linkedins-business-wide-monitoring-platform
- Tatbul et al.: https://people.csail.mit.edu/tatbul/publications/NeurIPS18.pdf
- TSB-AD benchmark: https://thedatumorg.github.io/TSB-AD/
