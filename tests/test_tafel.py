"""Tafel analysis: orientation independence, fit statistics, reaction inference."""

import numpy as np
import pytest

from scripts.modules import tafel


# --------------------------------------------------------------------------- #
# Reference conversion                                                        #
# --------------------------------------------------------------------------- #
def test_nernst_slope_at_25c():
    assert tafel.nernst_slope() == pytest.approx(0.05916, abs=1e-4)


def test_nernst_slope_scales_with_temperature():
    assert tafel.nernst_slope(333.15) > tafel.nernst_slope(298.15)


def test_to_rhe_matches_the_textbook_formula():
    got = tafel.to_rhe(np.array([0.0]), 0.197, 14.0)[0]
    assert got == pytest.approx(0.197 + 0.05916 * 14.0, abs=1e-4)


def test_theoretical_tafel_slope_reproduces_the_canonical_benchmarks():
    assert tafel.theoretical_tafel_slope_mv(0.5, 1) == pytest.approx(118.3, abs=1.0)
    assert tafel.theoretical_tafel_slope_mv(1.0, 1) == pytest.approx(59.2, abs=1.0)


# --------------------------------------------------------------------------- #
# Fitting                                                                     #
# --------------------------------------------------------------------------- #
def _tafel_curve(slope_v=0.120, intercept=0.05, n=60):
    log_i = np.linspace(-4, -1, n)
    pot = slope_v * log_i + intercept
    return pot, log_i


def test_fit_recovers_a_known_slope():
    pot, log_i = _tafel_curve()
    r = tafel.fit_tafel(pot, log_i, 0, len(pot))
    assert r.slope_mv_per_dec == pytest.approx(120.0, abs=0.1)
    assert r.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fit_reports_uncertainty_and_span():
    rng = np.random.default_rng(1)
    pot, log_i = _tafel_curve()
    pot = pot + rng.normal(0, 0.002, pot.size)
    r = tafel.fit_tafel(pot, log_i, 0, len(pot))
    assert r.slope_stderr_mv_per_dec > 0
    assert r.slope_ci95_mv_per_dec == pytest.approx(1.96 * r.slope_stderr_mv_per_dec)
    assert r.decades == pytest.approx(3.0, abs=1e-6)
    assert r.n_points == 60


def test_short_window_is_flagged_even_when_r2_is_perfect():
    """The failure R^2 cannot catch: a few adjacent points always look linear."""
    pot, log_i = _tafel_curve(n=200)
    r = tafel.fit_tafel(pot, log_i, 0, 5)
    assert r.r_squared > 0.999
    assert any("decade" in w for w in r.quality_warnings)


def test_wide_clean_fit_has_no_warnings():
    pot, log_i = _tafel_curve()
    assert tafel.fit_tafel(pot, log_i, 0, len(pot)).quality_warnings == []


def test_fit_survives_zero_current_in_the_window():
    """log10(0) is -inf; one such point used to poison every sum to nan."""
    pot, log_i = _tafel_curve()
    log_i = log_i.copy()
    log_i[10] = -np.inf
    r = tafel.fit_tafel(pot, log_i, 0, len(pot))
    assert np.isfinite(r.slope_v_per_dec)
    assert r.slope_mv_per_dec == pytest.approx(120.0, abs=0.5)


def test_fit_rejects_a_degenerate_window():
    pot, log_i = _tafel_curve()
    with pytest.raises(ValueError):
        tafel.fit_tafel(pot, log_i, 0, 2)


def test_log_current_maps_zero_to_neginf_without_raising():
    out = tafel.log_current(np.array([1.0, 0.0, -10.0]))
    assert out[0] == pytest.approx(0.0)
    assert np.isneginf(out[1])
    assert out[2] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Onset / benchmarks: must not depend on recording direction                  #
# --------------------------------------------------------------------------- #
def _her_sweep(n=300):
    pot = np.linspace(0.05, -0.45, n)
    cur = -1e-3 * (np.exp(-pot / 0.030) - 1.0)
    return pot, cur


def test_onset_is_independent_of_sweep_direction():
    """Reversing the file order used to raise outright."""
    pot, cur = _her_sweep()
    forward = tafel.onset_potential(pot, cur)
    reverse = tafel.onset_potential(pot[::-1], cur[::-1])
    assert forward == pytest.approx(reverse, abs=1e-9)


def test_onset_survives_an_approach_leg():
    pot, cur = _her_sweep()
    pot2 = np.concatenate([np.linspace(0.0, 0.05, 15), pot])
    cur2 = np.concatenate([np.full(15, cur[0]), cur])
    assert tafel.onset_potential(pot2, cur2) == pytest.approx(
        tafel.onset_potential(pot, cur), abs=0.02
    )


def test_potential_at_current_density_is_direction_independent():
    pot, cur = _her_sweep()
    fwd = tafel.potential_at_current_density(pot, cur, 10.0)
    rev = tafel.potential_at_current_density(pot[::-1], cur[::-1], 10.0)
    assert fwd is not None
    assert fwd == pytest.approx(rev, abs=1e-9)


def test_potential_at_current_density_returns_none_when_never_reached():
    pot, cur = _her_sweep()
    assert tafel.potential_at_current_density(pot, cur, 1e9) is None


def test_overpotential_uses_e_eq_when_the_reaction_has_one():
    pot, cur = _her_sweep()
    eta, is_eta = tafel.overpotential_at_current_density(pot, cur, 10.0, "HER")
    assert is_eta and eta >= 0

    raw, is_eta2 = tafel.overpotential_at_current_density(
        pot, cur, 10.0, "Other / unspecified"
    )
    assert not is_eta2 and raw < 0  # raw potential, not a magnitude


# --------------------------------------------------------------------------- #
# Auto range                                                                  #
# --------------------------------------------------------------------------- #
def test_auto_range_finds_a_window_in_either_direction():
    pot, cur = _her_sweep()
    log_i = tafel.log_current(cur)
    a0, a1 = tafel.auto_tafel_range(pot, log_i, current=cur)
    assert a1 - a0 >= 5

    b0, b1 = tafel.auto_tafel_range(pot[::-1], log_i[::-1], current=cur[::-1])
    assert b1 - b0 >= 5


def test_auto_range_is_fast_on_a_large_sweep():
    """Was O(n^2) via per-window polyfit: ~1.7 s for 4000 points, on every
    Streamlit rerun. The prefix-sum form must stay well under that."""
    import time

    pot = np.linspace(0, -0.6, 4000)
    cur = -1e-6 * np.exp(-pot / 0.03)
    log_i = tafel.log_current(cur)
    start = time.perf_counter()
    tafel.auto_tafel_range(pot, log_i, current=cur)
    assert time.perf_counter() - start < 0.5


def test_prefix_sum_stats_match_polyfit():
    rng = np.random.default_rng(3)
    x = np.sort(rng.uniform(0, 10, 200))
    y = 3.0 * x - 7.0 + rng.normal(0, 0.5, 200)
    sums = tafel._regression_prefix_sums(x, y)
    for lo, hi in [(0, 200), (10, 60), (33, 99)]:
        slope, intercept, r2 = tafel._stats_from_sums(sums, lo, hi)
        exp_s, exp_i = np.polyfit(x[lo:hi], y[lo:hi], 1)
        assert slope == pytest.approx(exp_s, rel=1e-9)
        assert intercept == pytest.approx(exp_i, rel=1e-9)
        assert r2 == pytest.approx(
            tafel._r_squared(x[lo:hi], y[lo:hi], exp_s, exp_i), rel=1e-9
        )


# --------------------------------------------------------------------------- #
# Reaction inference                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "expected, pot, cur_fn",
    [
        ("HER", np.linspace(0.05, -0.45, 300), lambda e: -1e-3 * (np.exp(-e / 0.03) - 1)),
        ("OER", np.linspace(1.20, 1.75, 300), lambda e: 1e-3 * (np.exp((e - 1.23) / 0.04) - 1)),
        ("ORR", np.linspace(1.00, 0.15, 300), lambda e: -3.0 / (1 + np.exp((e - 0.75) / 0.03))),
        ("HOR", np.linspace(-0.02, 0.30, 300), lambda e: 1e-3 * (np.exp(e / 0.03) - 1)),
    ],
)
def test_infer_reaction_identifies_the_textbook_windows(expected, pot, cur_fn):
    assert tafel.infer_reaction(pot, cur_fn(pot)).reaction == expected


def test_infer_reaction_is_direction_independent():
    pot = np.linspace(1.00, 0.15, 300)
    cur = -3.0 / (1 + np.exp((pot - 0.75) / 0.03))
    assert (tafel.infer_reaction(pot, cur).reaction
            == tafel.infer_reaction(pot[::-1], cur[::-1]).reaction)


def test_infer_reaction_declines_to_guess_on_flat_data():
    pot = np.linspace(0, 1, 50)
    guess = tafel.infer_reaction(pot, np.zeros(50))
    assert guess.confidence == "low"


def test_infer_reaction_reports_a_reason():
    pot = np.linspace(1.20, 1.75, 300)
    guess = tafel.infer_reaction(pot, 1e-3 * (np.exp((pot - 1.23) / 0.04) - 1))
    assert "1.23" in guess.reason


def test_nearest_reference_only_for_known_reactions():
    assert tafel.nearest_reference(118.0, "HER")[0] == 120.0
    assert tafel.nearest_reference(118.0, "MOR") is None


def test_hor_swept_past_the_oer_window_is_still_hor():
    """An HOR scan taken all the way up past 1.4 V satisfies both the "reaches
    1.35 V" OER test and the "already flowing at 0 V" HOR test. Where the
    current *starts* is the discriminating evidence, so HOR must win."""
    pot = np.linspace(0.0, 1.45, 400)
    cur = 2.0 * (1 - np.exp(-pot / 0.06))
    assert tafel.infer_reaction(pot, cur).reaction == "HOR"


def test_genuine_oer_is_not_misread_as_hor():
    pot = np.linspace(1.0, 1.80, 400)
    cur = np.exp((pot - 1.45) / 0.05)
    assert tafel.infer_reaction(pot, cur).reaction == "OER"


def test_cathodic_wave_just_above_zero_volts_is_not_high_confidence_her():
    """A poor ORR catalyst reaches into the 0–0.15 V band; HER is likelier but
    not certain, so the guess must not claim high confidence there."""
    pot = np.linspace(0.35, 0.02, 300)
    cur = -2.0 / (1 + np.exp((pot - 0.10) / 0.02))
    guess = tafel.infer_reaction(pot, cur)
    assert guess.reaction == "HER"
    assert guess.confidence == "medium"
    assert "ORR" in guess.reason


def test_cathodic_current_above_1v23_is_not_described_as_below_it():
    pot = np.linspace(1.60, 1.25, 300)
    cur = -2.0 / (1 + np.exp((pot - 1.40) / 0.02))
    guess = tafel.infer_reaction(pot, cur)
    assert "just below" not in guess.reason


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6, 1e8])
def test_prefix_sum_regression_survives_a_large_axis_offset(offset):
    """The one-pass form recovers the centred sum of squares as
    ``Sxx - (Sx)^2/n`` — a difference of two large nearly-equal numbers when
    the data sits far from the origin. Mean-centring before accumulating keeps
    the slope exact; without it a 1e8 offset made the slope 100 % wrong."""
    x = np.linspace(-5, -1, 400) + offset
    y = 0.12 * x + 0.4
    sums = tafel._regression_prefix_sums(x, y)
    slope, intercept, r_squared = tafel._stats_from_sums(sums, 0, len(x))
    assert slope == pytest.approx(0.12, rel=1e-8)
    assert r_squared == pytest.approx(1.0, abs=1e-9)
    # The intercept is an extrapolation back to x = 0, so its absolute error
    # scales with the offset; compare it where it is meaningful instead.
    assert (slope * x[0] + intercept) == pytest.approx(y[0], rel=1e-9)


def test_prefix_sum_regression_matches_polyfit_on_a_slice():
    x = np.linspace(-6.0, -2.0, 200)
    y = -0.09 * x + 1.4
    sums = tafel._regression_prefix_sums(x, y)
    slope, intercept, _ = tafel._stats_from_sums(sums, 40, 160)
    ref_slope, ref_intercept = np.polyfit(x[40:160], y[40:160], 1)
    assert slope == pytest.approx(ref_slope, rel=1e-10)
    assert intercept == pytest.approx(ref_intercept, rel=1e-10)


def test_exchange_current_refuses_a_potential_axis():
    """i0 is the current at zero *overpotential*. On an electrode-potential
    axis the same extrapolation returns the current at 0 V vs RHE, which is
    not i0 and can be many orders of magnitude out. Nothing in the numbers
    distinguishes the two axes, so the caller has to declare it."""
    eta = np.linspace(0.05, 0.35, 200)
    log_i = (eta - 0.1) / 0.12
    on_potential = tafel.fit_tafel(eta, log_i, 0, len(eta))
    on_overpotential = tafel.fit_tafel(eta, log_i, 0, len(eta), overpotential=True)

    assert on_potential.exchange_current is None
    assert on_overpotential.exchange_current is not None
    # 10 ** (-intercept/slope): the fit is eta = 0.12*log_i + 0.1, so log10(i0)
    # at eta = 0 is -0.1/0.12.
    assert on_overpotential.exchange_current == pytest.approx(
        10 ** (-0.1 / 0.12), rel=1e-6)
    # The slope itself is unaffected by the declaration.
    assert on_potential.slope_v_per_dec == pytest.approx(
        on_overpotential.slope_v_per_dec)


# --------------------------------------------------------------------------- #
# Decade-aware auto range                                                     #
# --------------------------------------------------------------------------- #
def test_auto_range_prefers_a_decade_over_a_locally_perfect_short_window():
    """The regression that motivated the decade-first score.

    A curve that is *very* slightly curved near the onset and cleanly linear
    over the decades that follow. Maximising R^2 alone locks onto a handful
    of points at the start (any smooth curve is straight over five points);
    the fit that means anything is the wide one.
    """
    rng = np.random.default_rng(11)
    log_i = np.linspace(-6.0, -1.0, 600)          # five decades
    pot = 1.30 + 0.120 * log_i                     # 120 mV/dec, the true slope
    pot += 0.004 * np.exp(-(log_i + 6.0) / 0.15)   # curvature confined to onset
    pot += rng.normal(0, 2e-4, len(pot))
    cur = 10.0 ** log_i

    a0, a1 = tafel.auto_tafel_range(pot, log_i, current=cur)
    r = tafel.fit_tafel(pot, log_i, a0, a1)
    assert r.decades >= tafel.MIN_TAFEL_DECADES
    assert r.slope_mv_per_dec == pytest.approx(120.0, abs=8.0)
    assert not any("decade" in w for w in r.quality_warnings)


def test_grow_from_onset_does_not_abandon_a_still_improving_window():
    """A window whose R^2 climbs out of the noise of its own minimum width
    used to exhaust the old patience counter and return the 5-point minimum,
    even though a far better window lay just beyond it."""
    rng = np.random.default_rng(5)
    log_i = np.linspace(-5.0, -1.0, 400)
    pot = 0.9 - 0.07 * log_i
    # Noise that dies away, so short windows at the start score poorly and
    # R^2 rises monotonically as the window widens.
    pot += rng.normal(0, 1.0, len(pot)) * 0.004 * np.exp(-np.arange(400) / 25)

    stop = tafel._grow_from_onset(log_i, pot, 0, 5, 0.99)
    assert stop > 5 + 40, f"growth abandoned at {stop}"


def test_auto_range_consults_the_global_scan_when_onset_window_is_too_narrow():
    """A sweep whose onset sits very late leaves no room for a decade-wide
    window anchored there; the global scan starts earlier and does reach one.
    The old acceptance test was on point count alone, so the narrow
    onset-anchored window was returned and the global scan never ran."""
    log_i = np.linspace(-6.0, -1.0, 500)
    pot = 1.25 + 0.100 * log_i
    cur = 10.0 ** log_i
    a0, a1 = tafel.auto_tafel_range(pot, log_i, current=cur)
    assert float(np.ptp(log_i[a0:a1])) >= tafel.MIN_TAFEL_DECADES


def test_window_r2_matches_the_scalar_form():
    """The vectorised scan must be arithmetically identical to the scalar
    prefix-sum stats it replaced -- it is 12x faster, not different."""
    rng = np.random.default_rng(7)
    x = np.sort(rng.uniform(0, 10, 300))
    y = 2.5 * x - 4.0 + rng.normal(0, 0.4, 300)
    sums = tafel._regression_prefix_sums(x, y)
    stops = np.arange(20, 301, 7)
    vec = tafel._window_r2(sums, 5, stops)
    for i, stop in enumerate(stops):
        scalar = tafel._stats_from_sums(sums, 5, int(stop))
        assert vec[i] == pytest.approx(scalar[2], abs=1e-12)


def test_window_decades_handles_a_non_monotonic_log_current():
    """log10|i| is only roughly monotonic on a real sweep; the span must be
    the true peak-to-peak over the window, not the endpoint difference."""
    x = np.array([0.0, 1.0, 3.0, 2.0, 2.5, 5.0])
    stops = np.array([3, 4, 6])
    got = tafel._window_decades(x, 0, stops)
    assert got == pytest.approx([3.0, 3.0, 5.0])
