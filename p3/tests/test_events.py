import math
import os

import pytest

from sentinel.config import AdaptationConfig
from sentinel.events import EventRegistry, EventStateMachine


def make(**kw) -> EventStateMachine:
    kw.setdefault("warmup_steps", 0)
    return EventStateMachine("s", AdaptationConfig(**kw))


def run(sm, scores):
    return [sm.step(x) for x in scores]


def labels(ds):
    return [d.label[0] for d in ds]  # "N" / "A"


def test_defaults_match_spec():
    c = AdaptationConfig()
    assert (c.enter_threshold, c.exit_threshold, c.min_event_length) == (3.5, 2.0, 1)
    assert (c.cooldown_steps, c.warmup_steps, c.regime_shift_patience) == (3, 30, 50)


def test_config_validation_and_unknown_keys():
    with pytest.raises(ValueError):
        AdaptationConfig(enter_threshold=2.0, exit_threshold=3.0)
    with pytest.raises(ValueError):
        AdaptationConfig.from_dict({"adaptation": {"enter_treshold": 3}})
    assert AdaptationConfig.from_dict({"adaptation": {"enter_threshold": 4.0}}).enter_threshold == 4.0


def test_yaml_matches_defaults():
    pytest.importorskip("yaml")
    path = os.path.join(os.path.dirname(__file__), "..", "configs", "baseline.yaml")
    assert AdaptationConfig.from_yaml(path) == AdaptationConfig()


def test_opens_immediately_with_min_event_length_1():
    d = run(make(), [0, 3.6])
    assert d[1].label == "ANOMALOUS" and d[1].event_id == "s_evt_0001"
    assert "EVENT_START" in d[1].reason_codes


def test_hysteresis_hold_between_exit_and_enter():
    d = run(make(cooldown_steps=0), [0, 4, 3.0, 2.5, 1.0])
    assert labels(d) == list("NAAAN")
    assert "HYSTERESIS_HOLD" in d[2].reason_codes
    assert "EVENT_END" in d[4].reason_codes
    assert len({x.event_id for x in d[1:4]}) == 1


def test_score_in_hysteresis_band_does_not_open_event():
    assert labels(run(make(), [3.0, 3.4, 2.5])) == list("NNN")


def test_applied_threshold_switches_with_state():
    d = run(make(), [0, 4, 3.0, 1.0, 0])
    assert [x.threshold for x in d] == [3.5, 3.5, 2.0, 2.0, 3.5]


def test_cooldown_merges_gap_up_to_cooldown_steps():
    d = run(make(cooldown_steps=3), [4, 1, 1, 1, 4])  # gap of exactly 3 NORMAL points
    assert d[4].label == "ANOMALOUS"
    assert d[0].event_id == "s_evt_0001" and d[4].event_id == "s_evt_0002"
    assert d[4].incident_id == d[0].incident_id == "s_inc_0001"
    assert "COOLDOWN_MERGE" in d[4].reason_codes
    assert labels(d)[1:4] == list("NNN")  # gap stays NORMAL (causal)


def test_gap_longer_than_cooldown_opens_new_event():
    d = run(make(cooldown_steps=3), [4, 1, 1, 1, 1, 4])
    assert d[0].event_id == "s_evt_0001" and d[5].event_id == "s_evt_0002"
    assert d[0].incident_id == "s_inc_0001" and d[5].incident_id == "s_inc_0002"
    assert "COOLDOWN_MERGE" not in d[5].reason_codes


def test_cooldown_zero_never_merges():
    d = run(make(cooldown_steps=0), [4, 1, 4])
    assert d[2].event_id == "s_evt_0002"


def test_min_event_length_requires_consecutive_exceedances():
    d = run(make(min_event_length=3), [4, 4, 1, 4, 4, 4])
    assert labels(d) == list("NNNNNA")
    assert "PENDING_CONFIRMATION" in d[0].reason_codes
    assert d[5].event_id == "s_evt_0001"


def test_warmup_blocks_events_and_is_per_series():
    reg = EventRegistry(AdaptationConfig(warmup_steps=5))
    a, b = reg.get("a"), reg.get("b")
    da = [a.step(10) for _ in range(5)]
    assert all(d.label == "NORMAL" and "WARMUP" in d.reason_codes for d in da)
    assert b.step(10).label == "NORMAL"  # b has its own warmup counter
    assert a.step(10).label == "ANOMALOUS"
    assert a.step(10).event_id == "a_evt_0001"


def test_detector_warmup_flag_is_respected():
    d = make().step(10, is_warmup=True)
    assert d.label == "NORMAL" and "WARMUP" in d.reason_codes


def test_warmup_flag_mid_event_closes_it():
    sm = make()
    sm.step(10)
    assert sm.step(10, is_warmup=True).label == "NORMAL"
    assert not sm.in_event


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), None, "x"])
def test_non_finite_scores_are_sanitized(bad):
    cfg = AdaptationConfig(warmup_steps=0)
    d = EventStateMachine("s", cfg).step(bad)
    assert math.isfinite(d.anomaly_score) and d.anomaly_score == cfg.max_score
    assert "NON_FINITE_SCORE" in d.reason_codes


def test_negative_and_huge_scores_are_clipped():
    sm = make(max_score=100.0)
    assert sm.step(-5).anomaly_score == 0.0
    assert sm.step(1e9).anomaly_score == 100.0


def test_regime_shift_signal_after_patience():
    d = run(make(regime_shift_patience=5, cooldown_steps=0), [5] * 7)
    assert [x.regime_shift_suspected for x in d] == [False] * 4 + [True] * 3
    assert "REGIME_SHIFT_SUSPECTED" in d[4].reason_codes


def test_regime_run_resets_after_event_closes():
    d = run(make(regime_shift_patience=3, cooldown_steps=5), [5, 5, 1, 5, 5])
    assert not any(x.regime_shift_suspected for x in d)  # merged, but run restarted


def test_series_isolation():
    reg = EventRegistry(AdaptationConfig(warmup_steps=0))
    reg.get("a").step(10)
    assert reg.get("b").step(0).event_id is None
    assert reg.get("b").step(10).event_id == "b_evt_0001"


def test_causality_prefix_decisions_unchanged_by_future():
    scores = [0, 4, 3, 1, 1, 4, 0, 0, 5, 5, 1]
    full = run(make(), scores)
    for k in range(1, len(scores) + 1):
        prefix = run(make(), scores[:k])
        assert [(d.label, d.event_id, d.threshold) for d in prefix] == \
               [(d.label, d.event_id, d.threshold) for d in full[:k]]


def test_label_and_event_id_consistency():
    for d in run(make(), [0, 4, 3, 1, 0, 0, 0, 0, 5, 1]):
        assert (d.label == "ANOMALOUS") == (d.event_id is not None)


def test_huge_integer_scores_fail_closed_instead_of_crashing():
    sm = make(warmup_steps=0)
    d = sm.step(10**10000)
    assert d.anomaly_score == sm.cfg.max_score
    assert d.label == "ANOMALOUS"
    assert "NON_FINITE_SCORE" in d.reason_codes


def test_config_rejects_fractional_window_counts():
    with pytest.raises(ValueError):
        AdaptationConfig(likelihood_short_window=2.5)
    with pytest.raises(ValueError):
        AdaptationConfig(adwin_clock=0.5)
    with pytest.raises(ValueError):
        AdaptationConfig(enter_threshold="not-a-number")


def test_confirmed_event_start_is_first_exceedance_timestamp():
    sm = make(min_event_length=3)
    decisions = [sm.step(4.0, timestamp=f"t{i}") for i in range(3)]
    assert decisions[0].label == decisions[1].label == "NORMAL"
    assert decisions[2].label == "ANOMALOUS"
    assert decisions[2].event_start_timestamp == "t0"
    assert decisions[2].incident_start_timestamp == "t0"
