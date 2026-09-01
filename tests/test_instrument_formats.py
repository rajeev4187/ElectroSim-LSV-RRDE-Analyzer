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


# A two-experiment .paax document, mirroring the real DAB tree shape found
# in every sample export on hand: DAB_node elements are flat siblings
# (study/experiment/X_results are *not* XML-nested in each other) linked
# purely by a "parent" index attribute, except that a <stream> genuinely
# does live nested inside its own <DAB_node type="X_results"> as
# <streams><stream>...</stream></streams>. The *study* node (index 2 / 12)
# carries the real, useful name; its child *experiment* node (index 3 / 13)
# always carries the literal placeholder "experiment" -- confirmed across
# every real export on hand, never a custom name -- so the label has to
# come from the study, not the experiment.
_MULTI_EXPERIMENT_PAAX = b"""<?xml version="1.0" encoding="ASCII"?>
<DAB_node_list>
<DAB_node type="study" index="2" parent="1">
<name length="13" encoding="mixed">CV%20Run%20A</name></DAB_node>
<DAB_node type="experiment" index="3" parent="2">
<name length="10" encoding="mixed">experiment</name></DAB_node>
<DAB_node type="X_results" index="4" parent="3">
<name length="7" encoding="mixed">results</name>
<streams>
<stream>
<identifier name="current_vs_potential"></identifier>
<name><ascii>Current vs Potential</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.001, 0.002</Y_data></points>
</data>
</stream>
</streams>
</DAB_node>
<DAB_node type="study" index="12" parent="1">
<name length="0" encoding="mixed"></name></DAB_node>
<DAB_node type="experiment" index="13" parent="12">
<name length="10" encoding="mixed">experiment</name></DAB_node>
<DAB_node type="X_results" index="14" parent="13">
<name length="7" encoding="mixed">results</name>
<streams>
<stream>
<identifier name="current_vs_potential"></identifier>
<name><ascii>Current vs Potential</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>10, 11</X_data><Y_data>0.003, 0.004</Y_data></points>
</data>
</stream>
</streams>
</DAB_node>
</DAB_node_list>
"""


def test_paax_list_lsv_candidates_one_per_experiment():
    candidates = instrument_formats.list_lsv_candidates(
        io.BytesIO(_MULTI_EXPERIMENT_PAAX), "multi.paax"
    )
    assert len(candidates) == 2
    labels = [c["label"] for c in candidates]
    # The named study's own name, URL-decoded -- not the experiment node's
    # own placeholder "experiment", and not a bare index either.
    assert "CV Run A" in labels
    # The unnamed study falls back to a bare index rather than "experiment".
    assert any(lbl.startswith("Experiment ") for lbl in labels)
    by_label = {c["label"]: c for c in candidates}
    np.testing.assert_allclose(by_label["CV Run A"]["potential"], [0, 1])
    np.testing.assert_allclose(by_label["CV Run A"]["current"], [1.0, 2.0])


def test_paax_rrde_candidates_keep_both_disk_and_ring():
    # A real RRDE experiment records TWO streams under the identical
    # "current_vs_potential" identifier within one experiment -- Pine names
    # them "Disk Current vs Disk Potential (2000 RPM)" / "Ring Current vs
    # Disk Potential (2000 RPM)", sharing the disk's own potential axis.
    # list_lsv_candidates (single-match-per-experiment) would silently keep
    # only whichever sorts first and drop the other; list_rrde_candidates
    # must keep both, and recover the rpm baked into the name.
    xml = b"""<?xml version="1.0" encoding="ASCII"?>
<DAB_node_list>
<DAB_node type="study" index="2" parent="1">
<name length="15" encoding="mixed">RRDE%20Run%20A</name></DAB_node>
<DAB_node type="experiment" index="3" parent="2">
<name length="10" encoding="mixed">experiment</name></DAB_node>
<DAB_node type="X_results" index="4" parent="3">
<name length="7" encoding="mixed">results</name>
<streams>
<stream>
<identifier name="current_vs_potential"></identifier>
<name><ascii>Disk Current vs Disk Potential (2000 RPM)</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.010, 0.020</Y_data></points>
</data>
</stream>
<stream>
<identifier name="current_vs_potential"></identifier>
<name><ascii>Ring Current vs Disk Potential (2000 RPM)</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.001, 0.002</Y_data></points>
</data>
</stream>
</streams>
</DAB_node>
</DAB_node_list>
"""
    candidates = instrument_formats.list_rrde_candidates(io.BytesIO(xml), "rrde.paax")
    assert len(candidates) == 1
    c = candidates[0]
    assert c["rpm"] == 2000.0
    # Amperes in the file -> milliamps in the project (see
    # _PAAX_SCALE_TO_PROJECT_UNITS).
    np.testing.assert_allclose(c["disk_current"], [10.0, 20.0])
    assert c["ring_current"] is not None
    np.testing.assert_allclose(c["ring_current"], [1.0, 2.0])


def test_paax_rrde_candidates_disk_only_when_no_ring_stream():
    candidates = instrument_formats.list_rrde_candidates(
        io.BytesIO(_MULTI_EXPERIMENT_PAAX), "multi.paax"
    )
    assert len(candidates) == 2
    for c in candidates:
        assert c["ring_current"] is None
        assert c["rpm"] is None  # no "NNN RPM" in these labels


def test_paax_lsv_candidates_include_dpv_but_not_lissajous():
    # differential_current_vs_potential (DPV) is a genuine polarization
    # curve under a different Pine identifier than plain CV/LSV's
    # current_vs_potential -- confirmed against a real DPV export -- and
    # must be picked up. lissajous (EIS's raw AC current-vs-potential loop)
    # has the exact same x=potential/y=current *shape* but is not a
    # polarization sweep at all -- confirmed against a real EIS export,
    # where it appears hundreds of times per file -- and must NOT be picked
    # up just because the shape matches; only the identifier allow-list
    # should decide this.
    xml = b"""<?xml version="1.0" encoding="ASCII"?>
<DAB_node_list>
<DAB_node type="study" index="2" parent="1">
<name length="3" encoding="mixed">DPV</name></DAB_node>
<DAB_node type="experiment" index="3" parent="2">
<name length="10" encoding="mixed">experiment</name></DAB_node>
<DAB_node type="X_results" index="4" parent="3">
<name length="7" encoding="mixed">results</name>
<streams>
<stream>
<identifier name="differential_current_vs_potential"></identifier>
<name><ascii>Differential Current vs Potential</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.001, 0.002</Y_data></points>
</data>
</stream>
<stream>
<identifier name="lissajous"></identifier>
<name><ascii>Lissajous</ascii></name>
<data>
<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>5, 6</X_data><Y_data>0.5, 0.6</Y_data></points>
</data>
</stream>
</streams>
</DAB_node>
</DAB_node_list>
"""
    candidates = instrument_formats.list_lsv_candidates(io.BytesIO(xml), "dpv.paax")
    assert len(candidates) == 1
    np.testing.assert_allclose(candidates[0]["potential"], [0, 1])  # DPV, not lissajous


def test_paax_list_eis_candidates_labelled_by_study():
    xml = _MULTI_EXPERIMENT_PAAX.replace(
        b'<identifier name="current_vs_potential"></identifier>\n'
        b"<name><ascii>Current vs Potential</ascii></name>\n"
        b"<data>\n"
        b'<X_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></X_units>\n'
        b'<Y_units qty_base="[I]" qty_kind="current"></Y_units>',
        b'<identifier name="impedance"></identifier>\n'
        b"<name><ascii>Nyquist (-Zi vs Zr)</ascii></name>\n"
        b"<data>\n"
        b'<X_units qty_base="[L2,M,T-3,I-2]" qty_kind="resistance"></X_units>\n'
        b'<Y_units qty_base="[L2,M,T-3,I-2]" qty_kind="resistance"></Y_units>',
    )
    candidates = instrument_formats.list_eis_candidates(io.BytesIO(xml), "multi.paax")
    assert len(candidates) == 2
    assert {"CV Run A"} <= {c["label"] for c in candidates}


def test_paax_list_chrono_candidates_handles_potential_only_experiment():
    # One experiment has both Measured Potential and Measured Current (a
    # typical chrono run); a second has only Measured Potential (e.g. an
    # OCP hold before the real technique) -- the candidate must still
    # appear, with current left as None rather than being dropped.
    xml = b"""<?xml version="1.0" encoding="ASCII"?>
<DAB_node_list>
<DAB_node type="study" index="2" parent="1">
<name length="4" encoding="mixed">CA%20</name></DAB_node>
<DAB_node type="experiment" index="3" parent="2">
<name length="10" encoding="mixed">experiment</name></DAB_node>
<DAB_node type="X_results" index="4" parent="3">
<name length="7" encoding="mixed">results</name>
<streams>
<stream>
<identifier name="potential"></identifier>
<name><ascii>Measured Potential</ascii></name>
<data>
<X_units qty_base="[T]" qty_kind="time"></X_units>
<Y_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.5, 0.6</Y_data></points>
</data>
</stream>
<stream>
<identifier name="current"></identifier>
<name><ascii>Measured Current</ascii></name>
<data>
<X_units qty_base="[T]" qty_kind="time"></X_units>
<Y_units qty_base="[I]" qty_kind="current"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.001, 0.002</Y_data></points>
</data>
</stream>
</streams>
</DAB_node>
<DAB_node type="study" index="12" parent="1">
<name length="6" encoding="mixed">OCP%20</name></DAB_node>
<DAB_node type="experiment" index="13" parent="12">
<name length="10" encoding="mixed">experiment</name></DAB_node>
<DAB_node type="X_results" index="14" parent="13">
<name length="7" encoding="mixed">results</name>
<streams>
<stream>
<identifier name="potential"></identifier>
<name><ascii>Measured OCP</ascii></name>
<data>
<X_units qty_base="[T]" qty_kind="time"></X_units>
<Y_units qty_base="[L2,M,T-3,I-1]" qty_kind="potential"></Y_units>
<points quantity="2"><X_data>0, 1</X_data><Y_data>0.1, 0.1</Y_data></points>
</data>
</stream>
</streams>
</DAB_node>
</DAB_node_list>
"""
    candidates = instrument_formats.list_chrono_candidates(io.BytesIO(xml), "multi.paax")
    assert len(candidates) == 2
    by_label = {c["label"]: c for c in candidates}
    assert by_label["CA"]["current"] is not None
    assert by_label["OCP"]["current"] is None
    np.testing.assert_allclose(by_label["OCP"]["potential"], [0.1, 0.1])


def test_non_paax_sources_yield_a_single_candidate(monkeypatch):
    # A non-.paax file (Biologic .mpr/.mpt) never has multiple "experiments"
    # to choose between -- the candidate-listing API still has to return
    # exactly one entry, wrapping whatever load_lsv/load_eis/load_table
    # already return, so a caller never has to special-case the format.
    monkeypatch.setattr(
        instrument_formats, "load_lsv",
        lambda file_or_path, filename: (np.array([0.0, 1.0]), np.array([2.0, 3.0])),
    )
    candidates = instrument_formats.list_lsv_candidates(io.BytesIO(b""), "run1.mpr")
    assert len(candidates) == 1
    np.testing.assert_allclose(candidates[0]["potential"], [0.0, 1.0])
    np.testing.assert_allclose(candidates[0]["current"], [2.0, 3.0])


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
