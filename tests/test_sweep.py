"""Sweep-geometry cleaning — the root-cause fix for the folded-sweep bugs."""

import numpy as np
import pytest

from scripts.modules import sweep


def test_single_monotonic_sweep_passes_through():
    pot = np.linspace(1.0, 0.0, 50)
    cur = -np.linspace(0, 1, 50)
    out_p, out_c = sweep.clean_sweep(pot, cur)
    assert len(out_p) == 50
    np.testing.assert_allclose(out_p, pot)
    np.testing.assert_allclose(out_c, cur)


def test_strips_leading_approach_leg():
    """The bundled ORR export's exact shape: a short ramp up to the vertex,
    then the real scan down."""
    approach = np.linspace(0.40, 0.50, 20)
    scan = np.linspace(0.50, -0.79, 250)
    pot = np.concatenate([approach, scan])
    cur = np.concatenate([np.full(20, 5e-6), np.linspace(5e-6, -2.7e-4, 250)])

    out_p, out_c = sweep.clean_sweep(pot, cur)

    assert len(out_p) < len(pot), "approach leg was not removed"
    assert len(out_p) >= 250
    # What remains must be a single direction.
    assert np.all(np.diff(out_p) <= 0)
    assert len(out_c) == len(out_p)


def test_main_sweep_indices_none_for_clean_data():
    assert sweep.main_sweep_indices(np.linspace(0, 1, 40)) is None


def test_monotonic_segments_counts_legs():
    pot = np.concatenate([np.linspace(0, 1, 10), np.linspace(1, -1, 20)])
    segments = sweep.monotonic_segments(pot)
    assert len(segments) == 2


def test_flat_steps_do_not_split_a_segment():
    """Quantised potential axes repeat values; that must not shred the sweep."""
    pot = np.repeat(np.linspace(0, 1, 20), 2)
    assert len(sweep.monotonic_segments(pot)) == 1


def test_orient_rest_first_puts_plateau_last():
    pot = np.linspace(0, -1, 40)
    cur = -np.linspace(0, 10, 40)  # grows toward the end: already correct
    p, c = sweep.orient_rest_first(pot, cur)
    np.testing.assert_allclose(c, cur)

    p2, c2 = sweep.orient_rest_first(pot[::-1], cur[::-1])
    assert abs(c2[-1]) > abs(c2[0]), "plateau end should sort last"
    np.testing.assert_allclose(c2, cur)


def test_orient_rest_first_survives_a_single_noisy_endpoint():
    """A block median is used precisely so one bad final sample cannot flip
    the whole record."""
    cur = -np.linspace(0, 10, 40)
    cur[-1] = -0.01  # one dropout at the plateau end
    pot = np.linspace(0, -1, 40)
    _, c = sweep.orient_rest_first(pot, cur)
    assert abs(c[0]) < abs(c[len(c) // 2]), "orientation flipped on one bad point"


def test_ascending_xy_sorts_and_collapses_duplicates():
    x = np.array([2.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 5.0, 20.0, 30.0])
    xs, ys = sweep.ascending_xy(x, y)
    np.testing.assert_allclose(xs, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(ys, [5.0, 15.0, 30.0])  # duplicates averaged


def test_safe_gradient_handles_duplicate_x():
    x = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 1.0, 1.0, 2.0, 3.0])
    g = sweep.safe_gradient(y, x)
    assert np.all(np.isfinite(g)), "duplicate x produced a non-finite gradient"


def test_safe_gradient_matches_numpy_on_clean_data():
    x = np.linspace(0, 10, 100)
    y = x ** 2
    np.testing.assert_allclose(sweep.safe_gradient(y, x), np.gradient(y, x))


def test_safe_gradient_stays_finite_where_numpy_returns_nan():
    """A fold repeats a potential exactly, so np.gradient divides by a zero
    step and poisons the result with nan. safe_gradient's contract is that it
    stays finite; removing the fold's *influence* is clean_sweep's job, and
    the two together are what orr.half_wave_derivative relies on."""
    scan = np.linspace(0.5, -0.5, 200)
    pot = np.concatenate([np.linspace(0.4, 0.5, 20), scan])
    cur = np.concatenate([np.full(20, 0.0), np.linspace(0, -1, 200)])

    with np.errstate(divide="ignore", invalid="ignore"):
        naive = np.gradient(cur, pot)
    assert not np.all(np.isfinite(naive)), "fixture no longer reproduces the fault"
    assert np.all(np.isfinite(sweep.safe_gradient(cur, pot)))


def test_clean_then_gradient_removes_the_fold_spike():
    """The combination callers actually use."""
    scan = np.linspace(0.5, -0.5, 200)
    pot = np.concatenate([np.linspace(0.4, 0.5, 20), scan])
    cur = np.concatenate([np.full(20, 0.0), np.linspace(0, -1, 200)])

    folded = np.max(np.abs(sweep.safe_gradient(cur, pot)))
    p_clean, c_clean = sweep.clean_sweep(pot, cur)
    cleaned = np.max(np.abs(sweep.safe_gradient(c_clean, p_clean)))
    assert cleaned < folded / 5, f"cleaned={cleaned:.3g} folded={folded:.3g}"


def test_interp_at_is_order_independent():
    x = np.linspace(1.0, 0.0, 50)  # descending
    y = 2.0 * x
    assert sweep.interp_at(0.5, x, y) == pytest.approx(1.0, abs=1e-9)


def test_clean_sweep_drops_non_finite_rows_in_lockstep():
    pot = np.array([0.0, 0.1, np.nan, 0.3, 0.4])
    cur = np.array([1.0, 2.0, 3.0, np.inf, 5.0])
    p, c = sweep.clean_sweep(pot, cur, strip_approach=False)
    assert len(p) == len(c) == 3
    assert np.all(np.isfinite(p)) and np.all(np.isfinite(c))
