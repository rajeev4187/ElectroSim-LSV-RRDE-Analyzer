"""ORR / RRDE: guarded kinetic current, masked ring-disk ratios, K-L fitting."""

import numpy as np
import pytest

from scripts.modules import orr


def _orr_sweep(n=300, j_lim=-6.0, e_half=0.75):
    pot = np.linspace(1.0, 0.1, n)
    disk = j_lim / (1.0 + np.exp((pot - e_half) / 0.03))
    return pot, disk


# --------------------------------------------------------------------------- #
# Onset / half-wave                                                           #
# --------------------------------------------------------------------------- #
def test_onset_and_half_wave_recovers_a_known_e_half():
    pot, disk = _orr_sweep(e_half=0.75)
    res = orr.onset_and_half_wave(pot, disk, half_wave_search_range=(0.4, 0.9))
    assert res.half_wave_potential == pytest.approx(0.75, abs=0.02)
    assert res.onset_potential > res.half_wave_potential
    assert res.limiting_current == pytest.approx(-6.0, abs=0.1)


@pytest.mark.parametrize("method", ["steepest", "interpolated", "second_derivative"])
def test_all_three_half_wave_methods_agree_on_a_clean_curve(method):
    pot, disk = _orr_sweep(e_half=0.75)
    res = orr.onset_and_half_wave(pot, disk, half_wave_search_range=(0.4, 0.9),
                                  method=method)
    assert res.half_wave_potential == pytest.approx(0.75, abs=0.05)


def test_unknown_half_wave_method_raises():
    pot, disk = _orr_sweep()
    with pytest.raises(ValueError):
        orr.onset_and_half_wave(pot, disk, method="nonsense")


def test_onset_is_direction_independent():
    pot, disk = _orr_sweep()
    a = orr.onset_and_half_wave(pot, disk, half_wave_search_range=(0.4, 0.9))
    b = orr.onset_and_half_wave(pot[::-1], disk[::-1], half_wave_search_range=(0.4, 0.9))
    assert a.onset_potential == pytest.approx(b.onset_potential, abs=1e-9)
    assert a.half_wave_potential == pytest.approx(b.half_wave_potential, abs=1e-9)


def test_approach_leg_does_not_move_e_half():
    pot, disk = _orr_sweep()
    pot2 = np.concatenate([np.linspace(0.9, 1.0, 20), pot])
    disk2 = np.concatenate([np.full(20, disk[0]), disk])
    a = orr.onset_and_half_wave(pot, disk, half_wave_search_range=(0.4, 0.9))
    b = orr.onset_and_half_wave(pot2, disk2, half_wave_search_range=(0.4, 0.9))
    assert a.half_wave_potential == pytest.approx(b.half_wave_potential, abs=0.02)


def test_derivative_has_no_spike_across_a_fold():
    pot, disk = _orr_sweep()
    pot2 = np.concatenate([np.linspace(0.9, 1.0, 20), pot])
    disk2 = np.concatenate([np.full(20, disk[0]), disk])
    _, d_clean = orr.half_wave_derivative(pot, disk)
    _, d_folded = orr.half_wave_derivative(pot2, disk2)
    assert np.max(np.abs(d_folded)) < 1.5 * np.max(np.abs(d_clean))


# --------------------------------------------------------------------------- #
# Kinetic current                                                             #
# --------------------------------------------------------------------------- #
def test_kinetic_current_matches_the_formula_away_from_the_plateau():
    j, jd = -1.0, -6.0
    got = orr.mass_transport_corrected_current(np.array([j]), jd)[0]
    assert got == pytest.approx(j * jd / (jd - j))


def test_kinetic_current_masks_the_plateau_instead_of_exploding():
    """j_k has a pole at j = j_d; unguarded it reached 126x |j_d| on real
    data and flipped sign on 63 points."""
    jd = -6.0
    j = np.linspace(0, -6.5, 400)  # runs past the plateau, as noisy data does
    jk = orr.mass_transport_corrected_current(j, jd)
    finite = np.isfinite(jk)
    assert finite.sum() > 0
    assert np.all(np.sign(jk[finite]) == np.sign(jd)), "sign-flipped points survived"
    assert np.max(np.abs(jk[finite])) < 25 * abs(jd)


def test_kinetic_current_rejects_a_zero_limiting_current():
    assert np.all(np.isnan(orr.mass_transport_corrected_current(np.array([1.0]), 0.0)))


# --------------------------------------------------------------------------- #
# Ring/disk                                                                   #
# --------------------------------------------------------------------------- #
def test_electron_number_is_four_with_no_ring_current():
    n = orr.electron_number(np.array([-1.0]), np.array([0.0]), 0.222)
    assert n[0] == pytest.approx(4.0)


def test_peroxide_percent_is_hundred_for_a_pure_two_electron_path():
    # |Ir|/N == |Id| -> half the flux is peroxide -> n = 2, %H2O2 = 100
    disk, ring = np.array([-1.0]), np.array([0.222])
    assert orr.peroxide_percent(disk, ring, 0.222)[0] == pytest.approx(100.0)
    assert orr.electron_number(disk, ring, 0.222)[0] == pytest.approx(2.0)


def test_pre_onset_region_is_masked_not_pinned_at_100_percent():
    """159 pre-onset points used to plot as a confident flat 100 % line."""
    disk = np.concatenate([np.full(100, -1e-4), np.linspace(-1e-4, -6.0, 100)])
    ring = np.concatenate([np.full(100, 2e-3), np.linspace(2e-3, 0.4, 100)])
    pct = orr.peroxide_percent(disk, ring, 0.222)
    assert np.all(np.isnan(pct[:90])), "pre-onset points were not masked"
    assert np.isfinite(pct[-10:]).all(), "real data got masked away too"


def test_zero_collection_efficiency_raises():
    with pytest.raises(ValueError):
        orr.electron_number(np.array([-1.0]), np.array([0.1]), 0.0)


def test_ring_disk_average_over_a_window():
    pot = np.linspace(0.8, 0.2, 200)
    disk = np.full(200, -6.0)
    ring = np.full(200, 0.222 * 2.0)  # -> n = 3
    n_mean, pct_mean, count = orr.ring_disk_average(pot, disk, ring, 0.222,
                                                    window=(0.3, 0.6))
    assert count > 0
    assert n_mean == pytest.approx(3.0, abs=1e-6)
    assert pct_mean == pytest.approx(50.0, abs=1e-6)


def test_ring_disk_average_reports_empty_window():
    pot = np.linspace(0.8, 0.2, 50)
    n_mean, pct_mean, count = orr.ring_disk_average(
        pot, np.full(50, -6.0), np.full(50, 0.4), 0.222, window=(5.0, 6.0)
    )
    assert count == 0 and np.isnan(n_mean) and np.isnan(pct_mean)


# --------------------------------------------------------------------------- #
# Koutecky-Levich                                                             #
# --------------------------------------------------------------------------- #
def test_kl_recovers_a_known_electron_number():
    d, nu, c = 1.9e-5, 1.0e-2, 1.2e-6
    n_true = 4.0
    rpms = np.array([400.0, 800.0, 1200.0, 1600.0, 2000.0])
    b = 0.62 * n_true * orr.FARADAY_C_PER_MOL * d ** (2 / 3) * nu ** (-1 / 6) * c
    jk = 0.05
    j = 1.0 / (1.0 / jk + 1.0 / (b * np.sqrt(orr.angular_velocity(rpms))))

    fit = orr.fit_koutecky_levich(rpms, j)
    assert fit.r_squared > 0.999
    assert fit.is_reliable
    assert orr.levich_slope_to_n(fit.slope, d, nu, c) == pytest.approx(4.0, abs=0.05)
    assert fit.kinetic_current_density == pytest.approx(jk, rel=1e-6)


def test_kl_gives_the_same_answer_for_cathodic_sign_convention():
    """The fit runs on |j|, so n and j_k come out positive either way."""
    d, nu, c = 1.9e-5, 1.0e-2, 1.2e-6
    rpms = np.array([400.0, 800.0, 1200.0, 1600.0, 2000.0])
    b = 0.62 * 4.0 * orr.FARADAY_C_PER_MOL * d ** (2 / 3) * nu ** (-1 / 6) * c
    j = 1.0 / (1.0 / 0.05 + 1.0 / (b * np.sqrt(orr.angular_velocity(rpms))))

    pos = orr.fit_koutecky_levich(rpms, j)
    neg = orr.fit_koutecky_levich(rpms, -j)
    assert pos.slope == pytest.approx(neg.slope)
    assert neg.kinetic_current_density > 0
    assert orr.levich_slope_to_n(neg.slope, d, nu, c) > 0


def test_kl_reports_uncertainty():
    rng = np.random.default_rng(5)
    rpms = np.array([400.0, 800.0, 1200.0, 1600.0, 2000.0])
    x = 1.0 / np.sqrt(orr.angular_velocity(rpms))
    j = 1.0 / (20.0 + 3.0 * x + rng.normal(0, 0.05, x.size))
    fit = orr.fit_koutecky_levich(rpms, j)
    assert np.isfinite(fit.slope_stderr) and fit.slope_stderr > 0


def test_kl_needs_enough_rotation_rates():
    with pytest.raises(ValueError):
        orr.fit_koutecky_levich([400.0, 800.0], [1.0, 2.0])


def test_kl_flags_a_noise_dominated_fit_as_unreliable():
    """The real failure: near the onset, 1/j amplifies noise and the fit
    produced n = 19.9 with R^2 = 0.0007 inside the app's default window."""
    rng = np.random.default_rng(7)
    rpms = np.array([400.0, 800.0, 1200.0, 1600.0, 2000.0])
    j = rng.normal(0, 1, 5) * 1e-6 + 1e-5
    fit = orr.fit_koutecky_levich(rpms, np.abs(j))
    assert not fit.is_reliable or fit.r_squared >= 0.95


def test_levich_current_density_is_positive_and_grows_with_rpm():
    d, nu, c = 1.9e-5, 1.0e-2, 1.2e-6
    lo = orr.levich_current_density(4, 400, d, nu, c)
    hi = orr.levich_current_density(4, 1600, d, nu, c)
    assert 0 < lo < hi
    assert hi / lo == pytest.approx(2.0, rel=1e-6)  # sqrt(1600/400)


def test_levich_slope_to_n_returns_a_magnitude():
    assert orr.levich_slope_to_n(-1.0, 1.9e-5, 1.0e-2, 1.2e-6) > 0


def test_angular_velocity_conversion():
    assert orr.angular_velocity(1600.0) == pytest.approx(2 * np.pi * 1600 / 60)


def test_all_zero_disk_current_yields_no_peroxide_number():
    """With no disk current there is nothing for the ring signal to be a
    fraction *of*; the formula would otherwise report a confident 100 %."""
    disk = np.zeros(50)
    ring = np.full(50, 0.01)
    assert np.all(np.isnan(orr.peroxide_percent(disk, ring, 0.222)))
    assert np.all(np.isnan(orr.electron_number(disk, ring, 0.222)))


def test_duplicate_rotation_rates_are_averaged_not_double_counted():
    """Three points at one x is not a line. Repeats collapse to their mean,
    and the fit then fails the minimum-rates check honestly."""
    with pytest.raises(ValueError):
        orr.fit_koutecky_levich([1600, 1600, 1600], [-4.0, -4.2, -4.1])

    dup = orr.fit_koutecky_levich(
        [400, 400, 900, 1600], [-2.0, -2.2, -3.0, -4.0])
    avg = orr.fit_koutecky_levich([400, 900, 1600], [-2.1, -3.0, -4.0])
    assert dup.n_rotation_rates == 3
    assert dup.slope == pytest.approx(avg.slope, rel=1e-9)


def test_negative_intercept_has_no_kinetic_current_density():
    fit = orr.KoutieckyLevichFit(
        potential=0.7, slope=1.0, intercept=-0.5,
        r_squared=0.999, n_rotation_rates=4,
    )
    assert fit.kinetic_current_density is None
    assert not fit.is_reliable


def test_transport_parameters_must_be_positive():
    for bad in (0.0, -1e-5, float("nan")):
        with pytest.raises(ValueError):
            orr.levich_current_density(4.0, 1600, bad, 1.0e-2, 1.2e-6)
        with pytest.raises(ValueError):
            orr.levich_slope_to_n(1.0, 1.9e-5, bad, 1.2e-6)
