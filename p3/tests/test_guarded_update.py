import random
import pytest
from sentinel.config import AdaptationConfig
from sentinel.contracts import Point
from sentinel.adaptation import AdaptationLayer
from tests._stubs import MadDetector, event_f1


def feed(layer, values, sid="s"):
    return [layer.process(Point(sid, f"t{i}", float(v))) for i, v in enumerate(values)]


def noise(rng, n, mu=0.0):
    return [rng.gauss(mu, 1.0) for _ in range(n)]


def make(guard=True, window=100, **kw):
    det = MadDetector(window=window)
    layer = AdaptationLayer(det, det.init_state, AdaptationConfig(**kw))
    if not guard:  # ablation: accept everything into the baseline
        orig = det.update_state
        det.update_state = lambda p, ok, s: orig(p, True, s)
    return layer, det


def test_contamination_guard_keeps_baseline_clean():
    """A long anomaly (90% of the 100-point window) must not drag the median/MAD baseline with it."""
    rng = random.Random(0)
    vals = noise(rng, 150) + [12.0 + rng.gauss(0, 0.3) for _ in range(90)] + noise(rng, 60)
    layer, _ = make(regime_shift_patience=500)
    recs = feed(layer, vals)
    assert all(r["label"] == "ANOMALOUS" for r in recs[152:238])
    assert abs(sorted(layer._series["s"].state)[len(layer._series["s"].state) // 2]) < 0.5
    assert [r["label"] for r in recs[-30:]].count("ANOMALOUS") == 0  # recovers right after

    # ablation: without the guard the same data poisons the baseline and detection is lost
    bad, _ = make(guard=False, regime_shift_patience=500)
    brecs = feed(bad, vals)
    assert sum(r["label"] == "ANOMALOUS" for r in brecs[150:240]) < sum(r["label"] == "ANOMALOUS" for r in recs[150:240])


def test_regime_shift_rebaselines_instead_of_alarming_forever():
    rng = random.Random(1)
    vals = noise(rng, 150) + noise(rng, 400, mu=15.0)
    recs = feed(make(regime_shift_patience=30)[0], vals)
    assert any("REGIME_SHIFT_REBASELINE" in r["reason_codes"] for r in recs)
    assert [r["label"] for r in recs[-50:]].count("ANOMALOUS") == 0


def test_slow_creep_triggers_drift_reset():
    rng = random.Random(2)
    # slow ramp + slow-window detector: the baseline lags, so accepted-normal scores creep upward
    vals = [rng.gauss(min(0.004 * i, 6.0), 1.0) for i in range(2000)]
    recs = feed(make(window=400)[0], vals)
    assert any("DRIFT_RESET" in r["reason_codes"] for r in recs)
    # the refresh lets the lagging baseline catch up: fewer false alarms than with drift handling off
    off = feed(make(window=400, drift_enabled=False)[0], vals)
    count = lambda rs: sum(r["label"] == "ANOMALOUS" for r in rs)
    assert count(recs) < count(off)


def test_records_satisfy_contract():
    rng = random.Random(3)
    for r in feed(make()[0], noise(rng, 80) + [30.0]):
        assert set(r) == {"series_id", "timestamp", "anomaly_score", "label", "threshold", "event_id",
                          "incident_id", "detector_version", "reason_codes", "evidence"}
        assert (r["label"] == "ANOMALOUS") == (r["event_id"] is not None)
    assert r["label"] == "ANOMALOUS" and r["evidence"]["residual"] > 20


def test_series_are_isolated():
    layer, _ = make()
    rng = random.Random(4)
    feed(layer, noise(rng, 100), "a")
    r = feed(layer, [50.0], "b")[0]
    assert r["label"] == "NORMAL" and "WARMUP" in r["reason_codes"]


@pytest.mark.parametrize("mode", ["robust_z", "likelihood"])
def test_random_scores_do_not_look_good(mode):
    """Kim et al. (AAAI 2022): a score with no information must NOT score well on event F1."""
    rng = random.Random(5)
    n = 2000
    truth = [(400, 430), (900, 920), (1500, 1540)]
    vals = noise(rng, n)  # signal carries no anomaly; anomaly ranges are labels only
    layer, _ = make(threshold_mode=mode)
    # P2 stand-in that ignores the data: uniform random scores
    class Rand:
        def score_one(self, p, s): return __import__("sentinel.contracts").contracts.RawDetectorOutput(rng.uniform(0, 5), p.value, 0, 0, {})
        def update_state(self, p, ok, s): return s
    layer = AdaptationLayer(Rand(), lambda _: None, AdaptationConfig(threshold_mode=mode))
    recs = feed(layer, vals)
    assert event_f1(recs, truth) < 0.3


def test_likelihood_integration_updates_reference_with_raw_scores():
    """P3 regression: transformed z-scores must never be recycled as raw reference scores."""
    from sentinel.contracts import RawDetectorOutput

    class ConstantScore:
        def score_one(self, point, state):
            return RawDetectorOutput(0.5, point.value, 0.5, point.value - 0.5, {})
        def update_state(self, point, accepted, state):
            return state

    cfg = AdaptationConfig(threshold_mode="likelihood", drift_enabled=False)
    layer = AdaptationLayer(ConstantScore(), lambda _: None, cfg)
    recs = feed(layer, [0.5] * 80)

    assert all(r["label"] == "NORMAL" for r in recs)
    long_reference = list(layer._series["s"].scorer._long)
    assert len(long_reference) == 80
    assert all(value == pytest.approx(0.5) for value in long_reference)
    assert max(r["anomaly_score"] for r in recs[cfg.warmup_steps:]) == pytest.approx(0.0)
    assert recs[-1]["evidence"]["raw_score"] == pytest.approx(0.5)
    assert recs[-1]["evidence"]["score_mode"] == "likelihood"


def test_likelihood_sustained_raw_score_increase_is_detected():
    from sentinel.contracts import RawDetectorOutput

    class ValueScore:
        def score_one(self, point, state):
            return RawDetectorOutput(point.value, point.value, 0.5, point.value - 0.5, {})
        def update_state(self, point, accepted, state):
            return state

    cfg = AdaptationConfig(threshold_mode="likelihood", drift_enabled=False)
    layer = AdaptationLayer(ValueScore(), lambda _: None, cfg)
    recs = feed(layer, [0.5] * 60 + [8.0] * 15)
    assert any(r["label"] == "ANOMALOUS" for r in recs[60:])
    # Anomalous points must not enter the long-run accepted-normal reference distribution.
    assert all(value == pytest.approx(0.5) for value in layer._series["s"].scorer._long)


def test_invalid_score_and_evidence_are_fail_closed_and_json_safe():
    import json
    from sentinel.contracts import RawDetectorOutput

    class InvalidOnce:
        def score_one(self, point, state):
            if point.timestamp == "t31":
                return RawDetectorOutput(float("nan"), point.value, 0.5, float("nan"), {"feature": float("inf")})
            return RawDetectorOutput(0.5, point.value, 0.5, point.value - 0.5, {})
        def update_state(self, point, accepted, state):
            return state

    cfg = AdaptationConfig(threshold_mode="likelihood", drift_enabled=True)
    layer = AdaptationLayer(InvalidOnce(), lambda _: None, cfg)
    recs = feed(layer, [0.5] * 35)
    bad = recs[31]
    assert "NON_FINITE_SCORE" in bad["reason_codes"]
    assert "INVALID_EVIDENCE" in bad["reason_codes"]
    assert bad["evidence"]["raw_score"] is None
    assert bad["evidence"]["residual"] is None
    assert bad["evidence"]["context"]["feature"] is None
    assert bad["evidence"]["baseline_update_accepted"] is False
    assert bad["evidence"]["drift_window_width"] == 1  # invalid score did not enter ADWIN
    # strict JSON mode rejects NaN/Infinity; this record must remain standards-safe.
    json.dumps(bad, allow_nan=False)
    # Invalid samples neither enter the long-run nor the short-run likelihood state.
    ctx = layer._series["s"]
    assert all(value == pytest.approx(0.5) for value in ctx.scorer._long)
    # The invalid point is rejected, and the following cooldown points are intentionally frozen.
    assert len(ctx.scorer._long) == 31
    assert all(value == pytest.approx(0.5) for value in ctx.scorer._short)


def test_adaptation_layer_reset_makes_replay_deterministic():
    layer, _ = make(drift_enabled=False)
    first = feed(layer, [0.0] * 35 + [8.0])
    layer.reset("s")
    replay = feed(layer, [0.0] * 35 + [8.0])
    assert [(r["label"], r["event_id"], r["anomaly_score"]) for r in first] == [
        (r["label"], r["event_id"], r["anomaly_score"]) for r in replay
    ]
    layer.reset()
    assert layer._series == {}


def test_negative_raw_scores_fail_closed_instead_of_looking_normal():
    from sentinel.contracts import RawDetectorOutput

    class NegativeScore:
        def score_one(self, point, state):
            raw = -1.0 if point.timestamp == "t30" else 0.5
            return RawDetectorOutput(raw, point.value, 0.5, point.value - 0.5, {})
        def update_state(self, point, accepted, state):
            return state

    cfg = AdaptationConfig(warmup_steps=30, drift_enabled=False)
    layer = AdaptationLayer(NegativeScore(), lambda _: None, cfg)
    recs = feed(layer, [0.5] * 35)
    bad = recs[30]
    assert bad["label"] == "ANOMALOUS"
    assert "INVALID_RAW_SCORE" in bad["reason_codes"]
    assert bad["evidence"]["baseline_update_accepted"] is False


def test_malformed_detector_warmup_flag_cannot_suppress_alerts():
    from sentinel.contracts import RawDetectorOutput

    class BadWarmupFlag:
        def score_one(self, point, state):
            return RawDetectorOutput(10.0, point.value, 0.0, point.value, {"is_warmup": "false"})
        def update_state(self, point, accepted, state):
            return state

    cfg = AdaptationConfig(warmup_steps=0, drift_enabled=False)
    rec = AdaptationLayer(BadWarmupFlag(), lambda _: None, cfg).process(
        __import__("sentinel.contracts").contracts.Point("s", "t0", 10.0)
    )
    assert rec["label"] == "ANOMALOUS"
    assert "INVALID_EVIDENCE" in rec["reason_codes"]



def test_event_evidence_preserves_original_start_timestamp_across_cooldown_merge():
    from sentinel.contracts import RawDetectorOutput

    class ValueScore:
        def score_one(self, point, state):
            return RawDetectorOutput(point.value, point.value, 0.0, point.value, {})
        def update_state(self, point, accepted, state):
            return state

    cfg = AdaptationConfig(warmup_steps=0, cooldown_steps=3, drift_enabled=False)
    layer = AdaptationLayer(ValueScore(), lambda _: None, cfg)
    values = [4.0, 1.0, 1.0, 1.0, 4.0]
    records = [layer.process(Point("s", f"t{i}", value)) for i, value in enumerate(values)]
    assert records[0]["event_id"] == "s_evt_0001"
    assert records[4]["event_id"] == "s_evt_0002"
    assert records[0]["incident_id"] == records[4]["incident_id"] == "s_inc_0001"
    assert records[0]["evidence"]["event_start_timestamp"] == "t0"
    assert records[4]["evidence"]["event_start_timestamp"] == "t4"
    assert records[4]["evidence"]["incident_start_timestamp"] == "t0"
