"""Chronoamperometry / chronopotentiometry (stability) analysis.

A chrono-technique record is one value (current density, for
chronoamperometry, or potential, for chronopotentiometry) sampled at fixed
intervals while the *other* electrical quantity is held constant. The
standard way this is reported in the literature is not the raw trace but its
**retention** relative to an early baseline — how much of the initial
activity survives after N hours — plus a handful of scalar summaries: the
final retention, a degradation rate, and how long the sample took to fall to
a given retention threshold (typically 90 %, or 50 % as a "half-life").

This module is deliberately generic in the *quantity* it tracks (current
density or potential) — the retention/degradation-rate math is identical
either way, only the physical interpretation differs, which the caller
already knows from which mode (CA vs CP) it is in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import sweep


@dataclass
class RetentionResult:
    """Retention trace and scalar stability summaries for one sample.

    ``retention_pct`` is ``value / baseline * 100`` at every input point, so
    values above 100 % are possible (e.g. a short activation rise before
    decay sets in, or a chronopotentiometry trace whose overpotential itself
    *falls* over time) and are not clipped.
    """

    time_s: np.ndarray
    value: np.ndarray
    retention_pct: np.ndarray
    baseline: float
    initial_value: float
    final_value: float
    final_retention_pct: float
    degradation_pct_per_hour: float
    half_life_s: float | None
    time_to_90pct_s: float | None
    r_squared: float


def _linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Ordinary least-squares ``y = slope*x + intercept``, plus R².

    Returns ``(slope, intercept, r_squared)``; ``(0.0, mean(y), 0.0)`` when
    there are fewer than 2 points or ``x`` is constant (no fit is possible).
    """
    if len(x) < 2 or np.ptp(x) == 0:
        return 0.0, float(np.mean(y)) if len(y) else 0.0, 0.0
    slope, intercept = np.polyfit(x, y, 1)
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(intercept), r_squared


def _time_at_threshold(time_s: np.ndarray, retention_pct: np.ndarray,
                       target_pct: float) -> float | None:
    """First time the retention trace crosses ``target_pct`` from above,
    linearly interpolated between the bracketing samples. ``None`` if the
    trace never falls to (or starts below) that level.
    """
    below = retention_pct <= target_pct
    if not below.any():
        return None
    first = int(np.argmax(below))
    if first == 0:
        # Already at/below target at the very first sample -- nothing to
        # interpolate against (no earlier "above" sample exists).
        return float(time_s[0]) if retention_pct[0] <= target_pct else None
    t0, t1 = time_s[first - 1], time_s[first]
    r0, r1 = retention_pct[first - 1], retention_pct[first]
    if r1 == r0:
        return float(t1)
    frac = (target_pct - r0) / (r1 - r0)
    return float(t0 + frac * (t1 - t0))


def compute_retention(time_s, value, baseline_window_s: float = 60.0) -> RetentionResult:
    """Retention trace + stability summary for one chrono-technique record.

    ``baseline_window_s`` sets the reference: the baseline is the mean of
    every point within this many seconds of the first sample, not just the
    single first point, so one noisy/transient opening reading doesn't set a
    spurious 100 % reference (same reasoning as the onset detector's
    ``baseline_frac`` in :mod:`scripts.modules.tafel`). Points are sorted by
    time first (duplicate/unsorted timestamps are tolerated, matching the
    rest of this package's ``sweep`` conventions).
    """
    t, v = sweep.ascending_xy(time_s, value)
    if len(t) < 2:
        raise ValueError("Need at least 2 valid (time, value) points.")

    window_mask = t <= (t[0] + baseline_window_s)
    baseline = float(np.mean(v[window_mask])) if window_mask.any() else float(v[0])
    if baseline == 0:
        raise ValueError("Baseline value is zero; cannot compute retention (%).")

    retention_pct = v / baseline * 100.0
    time_hours = t / 3600.0
    slope_pct_per_hour, _, r_squared = _linear_fit(time_hours, retention_pct)

    return RetentionResult(
        time_s=t,
        value=v,
        retention_pct=retention_pct,
        baseline=baseline,
        initial_value=float(v[0]),
        final_value=float(v[-1]),
        final_retention_pct=float(retention_pct[-1]),
        degradation_pct_per_hour=-slope_pct_per_hour,
        half_life_s=_time_at_threshold(t, retention_pct, 50.0),
        time_to_90pct_s=_time_at_threshold(t, retention_pct, 90.0),
        r_squared=r_squared,
    )
