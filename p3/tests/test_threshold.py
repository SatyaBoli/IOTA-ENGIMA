import math, random
import pytest
from sentinel.config import AdaptationConfig
from sentinel.adaptation.threshold import LikelihoodThreshold, RobustZThreshold, make_threshold


def test_factory_and_robust_z_is_identity_with_sanitizing():
    t = make_threshold(AdaptationConfig())
    assert isinstance(t, RobustZThreshold) and t.transform(3.7) == 3.7
    assert t.transform(float("nan")) == AdaptationConfig().max_score and t.transform(-2) == 0.0


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        AdaptationConfig(threshold_mode="zscore")


def test_tail_probability_matches_known_values():
    q = LikelihoodThreshold.tail_probability
    assert q(0) == pytest.approx(0.5) and q(1.6449) == pytest.approx(0.05, abs=1e-4)


def _fed(cfg, n=200, seed=0):
    t, rng = LikelihoodThreshold(cfg), random.Random(seed)
    for _ in range(n):
        s = abs(rng.gauss(0, 1))
        t.transform(s); t.update(s, True)
    return t


def test_likelihood_silent_until_min_samples():
    t = LikelihoodThreshold(AdaptationConfig(threshold_mode="likelihood"))
    assert all(t.transform(50.0) == 0.0 for _ in range(10))


def test_likelihood_needs_sustained_rise_not_a_single_spike():
    cfg = AdaptationConfig(threshold_mode="likelihood")
    t = _fed(cfg)
    spike = t.transform(3.0)  # one moderately high score among normal ones
    for _ in range(cfg.likelihood_short_window):
        t.transform(0.8)
    assert spike < cfg.enter_threshold
    sustained = [t.transform(6.0) for _ in range(cfg.likelihood_short_window)]
    assert sustained[-1] >= cfg.enter_threshold


def test_likelihood_one_sided_and_rejected_scores_dont_enter_reference():
    cfg = AdaptationConfig(threshold_mode="likelihood")
    t = _fed(cfg)
    assert t.transform(0.0) == 0.0  # low scores never register
    n = len(t._long)
    t.update(99.0, False)
    assert len(t._long) == n


def test_forget_older_than_keeps_reference_statistics_consistent():
    t = _fed(AdaptationConfig(threshold_mode="likelihood"), n=300)
    t.forget_older_than(60)
    assert len(t._long) == 60
    vals = list(t._long)
    mean, sigma = t._stats()
    expected_mean = math.fsum(vals) / len(vals)
    expected_var = math.fsum((v - expected_mean) ** 2 for v in vals) / len(vals)
    assert mean == pytest.approx(expected_mean)
    assert sigma == pytest.approx(max(math.sqrt(expected_var), 1e-3))
