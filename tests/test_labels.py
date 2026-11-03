import numpy as np

from quadrotor.labels import lowpass

DT = 0.0025


def test_lowpass_none_is_identity() -> None:
    x = np.random.default_rng(0).normal(size=(100, 3))
    assert np.array_equal(lowpass(x, DT, None), x)


def test_lowpass_keeps_slow_and_removes_fast_component() -> None:
    t = DT * np.arange(4000)
    slow, fast = np.sin(2 * np.pi * 1.0 * t), np.sin(2 * np.pi * 100.0 * t)
    y = lowpass((slow + fast)[:, None], DT, 10.0)[:, 0]
    inner = slice(400, -400)  # ignore filtfilt edge transients
    assert np.max(np.abs(y[inner] - slow[inner])) < 0.02
