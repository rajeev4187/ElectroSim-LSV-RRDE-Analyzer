"""EIS arc fitting, iR correction, and the data loaders' safety guards."""

import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from scripts.modules import correction, data_io, eis


# --------------------------------------------------------------------------- #
# EIS                                                                         #
# --------------------------------------------------------------------------- #
def _nyquist_arc(ru=25.0, rct=40.0, span_deg=180, n=40, noise=0.0, seed=0):
    """Upper-half Nyquist arc with a known left intercept (= ru)."""
    rng = np.random.default_rng(seed)
    a, r = ru + rct / 2.0, rct / 2.0
    theta = np.linspace(np.pi, np.pi - np.deg2rad(span_deg), n)
    x = a + r * np.cos(theta) + rng.normal(0, noise, n)
    y = r * np.sin(theta) + rng.normal(0, noise, n)
    return x, y


@pytest.mark.parametrize("span", [180, 120, 90, 60, 45])
def test_circle_fit_recovers_ru_across_arc_spans(span):
    x, y = _nyquist_arc(span_deg=span, noise=0.15)
    res = eis.fit_ru_circle(x, y, start=0, stop=len(x))
    assert res.ru == pytest.approx(25.0, abs=0.5)


def test_circle_fit_recovers_rct():
    x, y = _nyquist_arc()
    res = eis.fit_ru_circle(x, y, start=0, stop=len(x))
    assert res.rct == pytest.approx(40.0, abs=0.5)
    assert res.r_low == pytest.approx(65.0, abs=0.5)


def test_auto_arc_range_excludes_the_diffusion_tail():
    """A spectrum starting at high frequency (|Z''| ~ 0) used to defeat the
    global-argmin trough search and fit the whole tail, giving Ru = -2.34
    against a true 25."""
    theta = np.linspace(np.pi, 0.02, 30)
    a, r = 45.0, 20.0
    arc_x, arc_y = a + r * np.cos(theta), r * np.sin(theta)
    tail_x = np.linspace(arc_x[-1], arc_x[-1] + 30, 12)
    tail_y = np.linspace(0.3, 25, 12)
    zx = np.concatenate([arc_x, tail_x])
    zy = np.concatenate([arc_y, tail_y])

    start, stop = eis.auto_arc_range(zx, zy)
    assert stop < len(zx), "diffusion tail was not excluded"
    assert eis.fit_ru_circle(zx, zy).ru == pytest.approx(25.0, abs=1.0)


def test_auto_arc_range_keeps_everything_on_a_bare_arc():
    x, y = _nyquist_arc(span_deg=120)
    start, stop = eis.auto_arc_range(x, y)
    assert start == 0 and stop >= len(x) - 3


def test_fit_ru_handles_either_imaginary_sign_convention():
    x, y = _nyquist_arc()
    assert (eis.fit_ru_circle(x, y).ru
            == pytest.approx(eis.fit_ru_circle(x, -y).ru, abs=1e-9))


def test_fit_ru_needs_three_points():
    with pytest.raises(ValueError):
        eis.fit_ru_circle(np.array([1.0, 2.0]), np.array([1.0, 2.0]))


def test_min_imag_and_manual_methods():
    x, y = _nyquist_arc()
    assert eis.fit_ru_min_imag(x, y).method == "min_imag"
    assert eis.manual_ru(12.5).ru == 12.5


# --------------------------------------------------------------------------- #
# iR correction                                                               #
# --------------------------------------------------------------------------- #
def test_correction_subtracts_i_times_ru_in_volts():
    pot = np.array([0.0, 0.5])
    cur = np.array([10.0, 10.0])  # mA
    res = correction.apply_ir_correction(pot, cur, ru=10.0, factor_percent=100,
                                         current_unit="mA", ru_unit="Ω")
    # 10 mA * 10 ohm = 0.1 V
    np.testing.assert_allclose(res.ir_drop, [0.1, 0.1])
    np.testing.assert_allclose(res.potential_corrected, [-0.1, 0.4])


def test_correction_shifts_cathodic_current_the_other_way():
    res = correction.apply_ir_correction(np.array([0.0]), np.array([-10.0]),
                                         ru=10.0, factor_percent=100,
                                         current_unit="mA")
    assert res.potential_corrected[0] == pytest.approx(0.1)


def test_factor_is_clamped():
    assert correction.clamp_factor_percent(500) == 100
    assert correction.clamp_factor_percent(-5) == 5


def test_area_reconciles_density_with_plain_ohms():
    assert correction.reconcile_ru(10.0, "Ω", "mA/cm²", 0.5) == pytest.approx(5.0)
    assert correction.reconcile_ru(10.0, "Ω·cm²", "mA", 0.5) == pytest.approx(20.0)
    assert correction.reconcile_ru(10.0, "Ω", "mA") == 10.0  # already consistent


def test_missing_area_raises():
    with pytest.raises(ValueError):
        correction.reconcile_ru(10.0, "Ω", "mA/cm²", None)


def test_unknown_units_raise():
    with pytest.raises(ValueError):
        correction.reconcile_ru(1.0, "kOhm", "mA")
    with pytest.raises(ValueError):
        correction.apply_ir_correction(np.array([0.0]), np.array([1.0]),
                                       ru=1.0, current_unit="furlongs")


def test_monotonic_correction_is_not_flagged():
    pot = np.linspace(0, 1, 200)
    assert not correction.assess_correction(pot, pot - 0.01).over_compensated


def test_foldback_is_detected():
    pot = np.linspace(0, 1, 200)
    corrupted = pot.copy()
    corrupted[100:] = pot[100:] - np.linspace(0, 0.9, 100)  # reverses
    assert correction.assess_correction(pot, corrupted).over_compensated


def test_direction_detection_survives_an_approach_leg():
    """Endpoint-based direction detection classified this backwards, which
    made the fold-back metric report ~100 % on a good correction."""
    pot = np.concatenate([np.linspace(0.1, 0.0, 10), np.linspace(0.0, 1.0, 200)])
    assessment = correction.assess_correction(pot, pot - 0.01)
    assert assessment.direction == 1
    assert not assessment.over_compensated


def test_recommend_factor_backs_off_when_ru_is_too_large():
    pot = np.linspace(0, 1, 200)
    cur = np.linspace(0, 100, 200)  # mA
    safe = correction.recommend_factor(pot, cur, ru=50.0, current_unit="mA")
    assert correction.MIN_FACTOR_PERCENT <= safe <= correction.MAX_FACTOR_PERCENT


# --------------------------------------------------------------------------- #
# Loaders                                                                     #
# --------------------------------------------------------------------------- #
def _csv(text):
    return io.BytesIO(text.encode())


def test_load_lsv_matches_columns_by_header():
    d = data_io.load_lsv(_csv("Potential (V),Current (mA)\n0.1,1.0\n0.2,2.0\n"), sheet=None)
    np.testing.assert_allclose(d.potential, [0.1, 0.2])
    np.testing.assert_allclose(d.current, [1.0, 2.0])


def test_loader_falls_back_to_positional_columns():
    d = data_io.load_lsv(_csv("alpha,beta\n0.1,1.0\n0.2,2.0\n"), sheet=None)
    np.testing.assert_allclose(d.potential, [0.1, 0.2])


def test_loader_drops_non_numeric_rows():
    d = data_io.load_lsv(_csv("Potential,Current\n0.1,1.0\nbad,2.0\n0.3,3.0\n"),
                         sheet=None)
    assert len(d) == 2


def test_loader_rejects_a_file_with_no_numbers():
    with pytest.raises(ValueError):
        data_io.load_lsv(_csv("Potential,Current\na,b\nc,d\n"), sheet=None)


def test_column_pairs_split_multiple_datasets():
    text = "P1,C1,P2,C2\n0.1,1,0.5,5\n0.2,2,0.6,6\n"
    out = data_io.load_lsv_datasets(_csv(text), sheet=None)
    assert len(out) == 2
    np.testing.assert_allclose(out[1].potential, [0.5, 0.6])


def test_safe_label_strips_control_characters():
    assert "\n" not in data_io._safe_label("bad\nlabel")
    assert len(data_io._safe_label("x" * 500)) <= 60


def test_oversized_table_is_rejected():
    df = pd.DataFrame(np.zeros((3, data_io.MAX_COLS + 1)))
    with pytest.raises(data_io.DataValidationError):
        data_io._check_shape(df)


def test_non_zip_is_rejected_as_an_xlsx():
    with pytest.raises(data_io.DataValidationError):
        data_io._guard_excel_archive(io.BytesIO(b"not a zip"))


def test_zip_bomb_guard_trips_on_declared_size(monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("big.bin", b"0" * 1024)
    buf.seek(0)
    monkeypatch.setattr(data_io, "MAX_UNCOMPRESSED_BYTES", 100)
    with pytest.raises(data_io.DataValidationError):
        data_io._guard_excel_archive(buf)


def test_list_sheets_closes_its_handle(tmp_path):
    """Leaving ExcelFile to the GC emitted a ResourceWarning per call and
    held a lock on the file on Windows."""
    path = tmp_path / "book.xlsx"
    pd.DataFrame({"Potential": [0.1], "Current": [1.0]}).to_excel(path, index=False)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", ResourceWarning)
        assert data_io.list_sheets(path)
    path.unlink()  # would raise PermissionError on Windows if still open


def test_arc_coverage_measures_the_fitted_span():
    """A Nyquist arc's high-frequency end sits on the branch cut of arctan2,
    so a naive max-minus-min angle reads a 45° arc as a full turn."""
    for fraction in (0.05, 0.25, 0.5, 1.0):
        theta = np.linspace(np.pi, np.pi * (1 - fraction), 40)
        zr = 75 + 50 * np.cos(theta)
        zi = 50 * np.sin(theta)
        result = eis.fit_ru_circle(zr, zi)
        assert result.arc_coverage_deg == pytest.approx(fraction * 180, abs=0.5)
        assert result.ru == pytest.approx(25.0, abs=0.05)


def test_short_arcs_are_flagged_as_extrapolated():
    short = np.linspace(np.pi, np.pi * 0.9, 30)
    wide = np.linspace(np.pi, np.pi * 0.4, 30)
    short_fit = eis.fit_ru_circle(75 + 50 * np.cos(short), 50 * np.sin(short))
    wide_fit = eis.fit_ru_circle(75 + 50 * np.cos(wide), 50 * np.sin(wide))
    assert short_fit.is_extrapolated
    assert not wide_fit.is_extrapolated


def _semicircle(ru=25.0, rct=100.0, n=60):
    """Textbook capacitive arc, high -> low frequency."""
    theta = np.linspace(np.pi, 0.0, n)
    centre, radius = ru + rct / 2, rct / 2
    return centre + radius * np.cos(theta), radius * np.sin(theta)


def _with_inductive_lead(zr, zi):
    """Prepend a high-frequency inductive branch: Z' dips below Ru while Z''
    crosses to the other side of the real axis."""
    lead_zr = np.array([19.0, 21.0, 22.8, 24.0, 24.8])
    lead_zi = np.array([-13.0, -9.5, -6.0, -3.0, -0.8])
    return np.concatenate([lead_zr, zr]), np.concatenate([lead_zi, zi])


def test_inductive_lead_is_excluded_from_the_arc_fit():
    """abs() folds an inductive lead onto the capacitive side, where it reads
    as extra arc curving the wrong way and drags Ru with it."""
    zr, zi = _with_inductive_lead(*_semicircle())
    folded = eis.fit_ru_circle(zr, zi, start=0, stop=len(zr))
    fixed = eis.fit_ru_circle(zr, zi)
    assert abs(folded.ru - 25.0) > 1.0        # the defect this guards
    assert fixed.ru == pytest.approx(25.0, abs=0.05)
    assert fixed.arc_slice[0] == 5


def test_inductive_lead_detection_is_sign_convention_agnostic():
    zr, zi = _with_inductive_lead(*_semicircle())
    assert eis.inductive_lead_count(zi) == 5
    assert eis.inductive_lead_count(-zi) == 5
    assert eis.fit_ru_circle(zr, -zi).ru == pytest.approx(25.0, abs=0.05)


def test_a_clean_spectrum_loses_no_points_to_inductance_detection():
    zr, zi = _semicircle()
    assert eis.inductive_lead_count(zi) == 0
    assert eis.fit_ru_circle(zr, zi).arc_slice[0] == 0


def test_mostly_opposite_sign_data_is_left_alone():
    """If "most" of a spectrum reads as inductive the sign test has failed;
    trust nothing rather than eating the arc."""
    _, zi = _semicircle()
    assert eis.inductive_lead_count(-zi) == 0
