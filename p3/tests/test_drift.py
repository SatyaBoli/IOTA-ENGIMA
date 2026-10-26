import random
from sentinel.adaptation.drift import ADWIN


def stream(rng, n, mu, sd=0.05):
    return [min(max(rng.gauss(mu, sd), 0), 1) for _ in range(n)]


def test_no_false_alarm_on_stationary_stream():
    rng, a = random.Random(1), ADWIN(delta=0.002)
    assert not any(a.add(x) for x in stream(rng, 3000, 0.3))


def test_detects_mean_shift_and_drops_old_data():
    rng, a = random.Random(2), ADWIN(delta=0.002)
    for x in stream(rng, 600, 0.2): a.add(x)
    fired_at = next((i for i, x in enumerate(stream(rng, 600, 0.5)) if a.add(x)), None)
    assert fired_at is not None and fired_at < 150
    assert a.width < 400 and abs(a.mean - 0.5) < 0.1


def test_detects_slow_creep():
    rng, a, hit = random.Random(3), ADWIN(delta=0.002), False
    for i in range(1500):
        hit |= a.add(min(max(rng.gauss(0.2 + 0.0003 * i, 0.05), 0), 1))
    assert hit


def test_ignores_non_finite():
    a = ADWIN()
    assert a.add(float("nan")) is False and a.width == 0


def test_invalid_adwin_configuration_fails_early():
    import math
    import pytest

    with pytest.raises(ValueError):
        ADWIN(clock=0)
    with pytest.raises(ValueError):
        ADWIN(max_window=20, min_sub_window=11)
    with pytest.raises(ValueError):
        ADWIN(value_range=float("nan"))
    with pytest.raises(ValueError):
        ADWIN(delta=float("nan"))


def test_adwin_handles_unconvertible_and_overflowing_values():
    a = ADWIN()
    assert not a.add("not-a-score")
    assert not a.add(10**10000)
    assert a.width == 0
