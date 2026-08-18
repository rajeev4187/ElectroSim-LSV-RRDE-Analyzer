"""End-to-end checks against the bundled sample data.

These are the regression tests for the defects that were *measured* on real
instrument exports rather than constructed: the bundled ORR files open with a
21-point approach leg, which is what exposed the folded-sweep bugs in the
first place.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.modules import data_io, orr, sweep, tafel

ROOT = Path(__file__).resolve().parents[1]
ORR_DIR = ROOT / "sample-data" / "Example ORR 0-1 M KOH"
EIS_XLSX = ROOT / "sample-data" / "EIS example.xlsx"
LSV_XLSX = ROOT / "sample-data" / "LSV Example.xlsx"

AREA_CM2 = 0.19625
E_REF_HG_HGO, PH_KOH = 0.140, 13.0

pytestmark = pytest.mark.skipif(
    not ORR_DIR.exists(), reason="bundled sample data not present"
)


def _disk_files():
    return sorted(ORR_DIR.glob("**/Disk Current vs Disk Potential*.csv"))


def _load(path):
    df = pd.read_csv(path)
    pot = tafel.to_rhe(df.iloc[:, 0].to_numpy(float), E_REF_HG_HGO, PH_KOH)
    cur = df.iloc[:, 1].to_numpy(float)
    return pot, cur


def test_sample_data_is_present():
    assert len(_disk_files()) >= 3


def test_bundled_sweeps_really_do_contain_an_approach_leg():
    """If this ever fails the fixtures changed and the tests below lose their
    point."""
    pot, _ = _load(_disk_files()[0])
    assert len(sweep.monotonic_segments(pot)) > 1
    assert sweep.main_sweep_indices(pot) is not None


def test_cleaning_removes_the_leg_and_leaves_one_direction():
    pot, cur = _load(_disk_files()[0])
    p, c = sweep.clean_sweep(pot, cur)
    assert len(p) < len(pot)
    assert len(sweep.monotonic_segments(p)) == 1
    assert len(c) == len(p)


def test_derivative_spike_from_the_fold_is_gone():
    """Measured before the fix: 121.7 vs a true maximum of 4.66."""
    pot, cur = _load(_disk_files()[0])
    j = cur / AREA_CM2 * 1000.0
    _, deriv_raw = orr.half_wave_derivative(pot, j)
    p, c = sweep.clean_sweep(pot, j)
    _, deriv_clean = orr.half_wave_derivative(p, c)
    assert np.max(np.abs(deriv_raw)) == pytest.approx(
        np.max(np.abs(deriv_clean)), rel=0.05
    )
    assert np.max(np.abs(deriv_raw)) < 20


def test_onset_and_e_half_are_physically_sensible():
    pot, cur = _load([f for f in _disk_files() if "1600" in f.name][0])
    j = cur / AREA_CM2 * 1000.0
    res = orr.onset_and_half_wave(pot, j, half_wave_search_range=(0.4, 0.8))
    assert 0.4 < res.half_wave_potential < 0.9
    assert res.onset_potential > res.half_wave_potential
    assert res.limiting_current < 0  # cathodic


def test_onset_is_the_same_whichever_way_the_file_is_ordered():
    pot, cur = _load(_disk_files()[0])
    j = cur / AREA_CM2 * 1000.0
    a = orr.onset_and_half_wave(pot, j, half_wave_search_range=(0.4, 0.8))
    b = orr.onset_and_half_wave(pot[::-1], j[::-1], half_wave_search_range=(0.4, 0.8))
    assert a.onset_potential == pytest.approx(b.onset_potential, abs=1e-9)


def test_tafel_onset_no_longer_fails_on_a_reversed_file():
    """Reversing the record used to raise 'No clear onset found'."""
    pot, cur = _load(_disk_files()[0])
    forward = tafel.onset_potential(pot, cur)
    reverse = tafel.onset_potential(pot[::-1], cur[::-1])
    assert forward == pytest.approx(reverse, abs=1e-9)


def test_kinetic_current_has_no_sign_flipped_points():
    """63 unphysical sign-flipped points before the guard."""
    pot, cur = _load([f for f in _disk_files() if "1600" in f.name][0])
    p, j = sweep.clean_sweep(pot, cur / AREA_CM2 * 1000.0)
    res = orr.onset_and_half_wave(p, j, half_wave_search_range=(0.4, 0.8))
    jk = orr.mass_transport_corrected_current(j, res.limiting_current)
    finite = np.isfinite(jk)
    assert finite.sum() > 20
    assert np.all(np.sign(jk[finite]) == np.sign(res.limiting_current))


def test_peroxide_yield_is_masked_before_onset():
    """159 pre-onset points used to read a flat, confident 100 %."""
    disk_path = [f for f in _disk_files() if "1600" in f.name][0]
    ring_path = disk_path.with_name(disk_path.name.replace("Disk Current", "Ring Current"))
    if not ring_path.exists():
        pytest.skip("no matching ring file")

    pot, disk = _load(disk_path)
    _, ring = _load(ring_path)
    j_d = disk / AREA_CM2 * 1000.0
    j_r = ring / AREA_CM2 * 1000.0

    res = orr.onset_and_half_wave(pot, j_d, half_wave_search_range=(0.4, 0.8))
    pct = orr.peroxide_percent(j_d, j_r, 0.222)
    pre_onset = pot > res.onset_potential + 0.05
    assert pre_onset.sum() > 50
    assert np.all(np.isnan(pct[pre_onset])), "pre-onset garbage is still plotted"
    assert np.isfinite(pct).any(), "everything got masked"


def test_electron_number_at_the_plateau_is_in_range():
    disk_path = [f for f in _disk_files() if "1600" in f.name][0]
    ring_path = disk_path.with_name(disk_path.name.replace("Disk Current", "Ring Current"))
    if not ring_path.exists():
        pytest.skip("no matching ring file")
    pot, disk = _load(disk_path)
    _, ring = _load(ring_path)
    n_arr = orr.electron_number(disk / AREA_CM2, ring / AREA_CM2, 0.222)
    plateau = np.isfinite(n_arr) & (pot < 0.4)
    assert plateau.sum() > 0
    assert np.all((n_arr[plateau] >= 0) & (n_arr[plateau] <= 4))


def test_koutecky_levich_across_the_real_rotation_rates():
    rows = []
    for path in _disk_files():
        rpm_token = path.stem.split("(")[-1].split()[0]
        pot, cur = _load(path)
        p, j = sweep.clean_sweep(pot, cur / AREA_CM2)  # A/cm^2
        rows.append((float(rpm_token), p, j))
    if len(rows) < 3:
        pytest.skip("need 3+ rotation rates")

    rpms = [r[0] for r in rows]
    j_at = [float(sweep.interp_at(0.40, p, j)) for _, p, j in rows]
    fit = orr.fit_koutecky_levich(rpms, np.array(j_at))
    n = orr.levich_slope_to_n(fit.slope, 1.9e-5, 1.0e-2, 1.2e-6)

    assert n is not None and n > 0
    assert 0 < n < 8, f"n = {n} is not physically plausible"


def test_reaction_inference_on_the_bundled_files():
    """The LSV workbook is an OER data set; the app used to label it HER
    because HER was simply the first item in the dropdown."""
    pot, cur = _load(_disk_files()[0])
    assert tafel.infer_reaction(pot, cur).reaction == "ORR"

    if LSV_XLSX.exists():
        for ds in data_io.load_lsv_datasets(LSV_XLSX, sheet=0):
            assert tafel.infer_reaction(ds.potential, ds.current).reaction == "OER"


def test_bundled_eis_workbook_still_fits_the_same_ru():
    if not EIS_XLSX.exists():
        pytest.skip("no bundled EIS workbook")
    from scripts.modules import eis

    d = data_io.load_eis_datasets(EIS_XLSX, sheet=0)[0]
    assert eis.auto_arc_range(d.z_real, d.z_imag) == (0, 18)
    assert eis.fit_ru_circle(d.z_real, d.z_imag).ru == pytest.approx(27.5336, abs=1e-3)
