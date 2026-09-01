"""Chronoamperometry/chronopotentiometry retention & stability metrics."""

import numpy as np
import pytest

from scripts.modules import chrono


def test_clean_exponential_decay_half_life_and_t90():
    tau_s = 3600.0 * 5.0  # 5 h time constant
    t = np.linspace(0, 3600.0 * 20, 400)
    v = 10.0 * np.exp(-t / tau_s)
    res = chrono.compute_retention(t, v, baseline_window_s=1.0)

    assert res.final_retention_pct == pytest.approx(
        100.0 * np.exp(-t[-1] / tau_s), rel=1e-2
    )
    expected_half_life_h = np.log(2) * tau_s / 3600.0
    expected_t90_h = -np.log(0.9) * tau_s / 3600.0
    assert res.half_life_s / 3600.0 == pytest.approx(expected_half_life_h, rel=0.05)
    assert res.time_to_90pct_s / 3600.0 == pytest.approx(expected_t90_h, rel=0.05)
    assert res.degradation_pct_per_hour > 0  # decaying, so a positive loss rate


def test_flat_trace_no_degradation():
    t = np.linspace(0, 3600.0 * 10, 100)
    v = np.full_like(t, 5.0)
    res = chrono.compute_retention(t, v)
    assert res.final_retention_pct == pytest.approx(100.0)
    assert res.degradation_pct_per_hour == pytest.approx(0.0, abs=1e-9)
    assert res.half_life_s is None
    assert res.time_to_90pct_s is None


def test_pure_noise_never_crosses_threshold_returns_none():
    rng = np.random.default_rng(0)
    t = np.linspace(0, 3600.0, 50)
    v = 10.0 + rng.normal(scale=0.01, size=50)  # tiny noise, no real decay
    res = chrono.compute_retention(t, v, baseline_window_s=60.0)
    assert res.half_life_s is None
    assert res.time_to_90pct_s is None


def test_zero_baseline_raises():
    t = np.array([0.0, 1.0, 2.0])
    v = np.array([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        chrono.compute_retention(t, v, baseline_window_s=1.0)


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        chrono.compute_retention(np.array([0.0]), np.array([1.0]))


def test_unsorted_time_is_handled():
    t = np.array([2.0, 0.0, 1.0, 3.0])
    v = np.array([8.0, 10.0, 9.0, 7.0])
    res = chrono.compute_retention(t, v, baseline_window_s=0.5)
    assert np.all(np.diff(res.time_s) >= 0)
    assert res.initial_value == pytest.approx(10.0)
    assert res.final_value == pytest.approx(7.0)
