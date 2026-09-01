"""Instrument-file format detection and parsing.

The Biologic .mpr/.mpt path is exercised only via extension recognition and
the not-yet-supported-format error path here — real Biologic parsing goes
through the third-party `galvani` library and was hand-verified against a
real .mpr file during development (not committed to the repo).

The Pine Research AfterMath (.paax) path is genuinely plain XML (confirmed
against real exports), so it's tested here against a small synthetic
document built from the exact tag/attribute shapes verified in those real
files — this catches a real regression without needing a multi-MB fixture.
"""

import io

import numpy as np
import pytest

from scripts.modules import instrument_formats

# A minimal .paax-shaped document covering the four stream kinds this
# module reads: a paginated time-series potential/current pair (chrono), a
# current-vs-potential sweep (LSV/CV), and a Nyquist impedance curve (EIS).
# Field/attribute names and nesting match real AfterMath exports exactly;
# unrelated GUI/metadata nodes real files carry are omitted as irrelevant.
_SAMPLE_PAAX = b"""<?xml version="1.0" encoding="ASCII"?>
<DAB_node_list>
<stream>
<identifier name="potential"></identifier>
<name><ascii>Measured Potential</ascii></name>
<data>
<X_units qty_base="[T]" qty_kind="time"></X_units>
<Y_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.1, 0.2</Y_data></points>
<points quantity="2"><X_data>2, 3</X_data><Y_data>0.3, 0.4</Y_data></points>
</data>
</stream>
<stream>
<identifier name="current"></identifier>
<name><ascii>Measured Current</ascii></name>
<data>
<X_units qty_base="[T]" qty_kind="time"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="4"><X_data>0, 1, 2, 3</X_data><Y_data>0.001, 0.002, 0.003, 0.004</Y_data></points>
</data>
</stream>
<stream>
<identifier name="current_vs_potential"></identifier>
<name><ascii>Current vs Potential</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="3"><X_data>-0.5, 0, 0.5</X_data><Y_data>0.0001, 0.0002, 0.0003</Y_data></points>
</data>
</stream>
<stream>
<identifier name="impedance"></identifier>
<name><ascii>Nyquist (-Zi vs Zr)</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-2]" qty_kind="resistance"></X_units>
<Y_units qty_base="[L2,M,T-3,I-2]" qty_kind="resistance"></Y_units>
<points quantity="2"><X_data>10, 20</X_data><Y_data>5, 3</Y_data></points>
</data>
</stream>
</DAB_node_list>
"""


def test_known_supported_extensions():
    for name, expected in [
        ("run1.mpr", "Biologic EC-Lab (binary export)"),
        ("run1.mpt", "Biologic EC-Lab (ASCII export)"),
        ("run1.paax", "Pine Research AfterMath"),
    ]:
        assert instrument_formats.is_supported(name)
        assert instrument_formats.known_format_name(name) == expected


def test_known_but_unsupported_extension():
    assert not instrument_formats.is_supported("run1.bin")
    assert instrument_formats.known_format_name("run1.bin") == "CH Instruments"


def test_unknown_extension():
    assert instrument_formats.known_format_name("run1.csv") is None
    assert not instrument_formats.is_supported("run1.csv")


def test_load_table_raises_for_unsupported_format():
    with pytest.raises(instrument_formats.UnsupportedFormatError):
        instrument_formats.load_table(io.BytesIO(b"\x00"), "run1.bin")


def test_load_table_raises_for_unknown_format():
    with pytest.raises(instrument_formats.UnsupportedFormatError):
        instrument_formats.load_table(io.BytesIO(b"\x00"), "run1.xyz")


def test_paax_table_concatenates_pages_and_scales_current_to_mA():
    df = instrument_formats.load_table(io.BytesIO(_SAMPLE_PAAX), "run1.paax")
    np.testing.assert_allclose(df["time/s"].to_numpy(), [0, 1, 2, 3])
    np.testing.assert_allclose(df["Ewe/V"].to_numpy(), [0.1, 0.2, 0.3, 0.4])
    # Amperes in the file (qty_base="[I]") -> milliamps in the project.
    np.testing.assert_allclose(df["I/mA"].to_numpy(), [1.0, 2.0, 3.0, 4.0])


def test_paax_lsv_reads_current_vs_potential_stream():
    potential, current = instrument_formats.load_lsv(
        io.BytesIO(_SAMPLE_PAAX), "run1.paax"
    )
    np.testing.assert_allclose(potential, [-0.5, 0, 0.5])
    np.testing.assert_allclose(current, [0.1, 0.2, 0.3])  # scaled A -> mA


def test_paax_eis_reads_nyquist_stream():
    z_real, z_imag = instrument_formats.load_eis(io.BytesIO(_SAMPLE_PAAX), "run1.paax")
    np.testing.assert_allclose(z_real, [10, 20])
    np.testing.assert_allclose(z_imag, [5, 3])


def test_paax_table_raises_without_a_time_series_stream():
    xml = b"""<?xml version="1.0" encoding="ASCII"?>
<DAB_node_list>
<stream>
<identifier name="current_vs_potential"></identifier>
<name><ascii>Current vs Potential</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.001, 0.002</Y_data></points>
</data>
</stream>
</DAB_node_list>
"""
    with pytest.raises(ValueError):
        instrument_formats.load_table(io.BytesIO(xml), "run1.paax")
