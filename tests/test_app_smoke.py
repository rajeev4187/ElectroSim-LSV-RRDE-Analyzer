"""Smoke tests that actually render the Streamlit app.

Unit tests cover the analysis modules; these cover the app shell, where the
failures are of a different kind — a deprecated widget argument, a missing
session-state key, a helper renamed on one side only. ``AppTest`` runs the
real script in-process and collects anything Streamlit would have shown the
user as an error.
"""

import warnings
from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"

pytestmark = pytest.mark.skipif(not APP.exists(), reason="app.py not found")


def _run(timeout=600):
    at = AppTest.from_file(str(APP), default_timeout=timeout)
    at.run()
    return at


def _problems(at):
    return [str(e.value) for e in at.exception] + [str(e.value) for e in at.error]


def test_app_renders_without_error():
    at = _run()
    assert not _problems(at), _problems(at)


def test_app_renders_with_the_bundled_samples_loaded():
    at = _run()
    for radio in at.radio:
        if radio.key in ("eis_source", "lsv_source"):
            radio.set_value("Use bundled sample")
    at.run()
    assert not _problems(at), _problems(at)
    # The EIS fit must have produced its metrics row.
    assert any("Ru" in (m.label or "") for m in at.metric)


def test_bundled_sample_reports_the_expected_ru():
    at = _run()
    for radio in at.radio:
        if radio.key == "eis_source":
            radio.set_value("Use bundled sample")
    at.run()
    values = {m.label: m.value for m in at.metric}
    ru = next((v for k, v in values.items() if k.startswith("Ru (Ω")), None)
    assert ru is not None, values
    assert float(ru) == pytest.approx(27.534, abs=0.01)


def test_no_deprecated_streamlit_arguments_remain():
    """``use_container_width`` was removed from Streamlit's public API; the
    app emitted 24 deprecation warnings per render before this was fixed."""
    source = APP.read_text(encoding="utf-8")
    assert "use_container_width" not in source


def test_rendering_emits_no_resource_warnings():
    """Unclosed workbook handles both warned and, on Windows, locked the
    sample files."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        at = _run()
        for radio in at.radio:
            if radio.key in ("eis_source", "lsv_source"):
                radio.set_value("Use bundled sample")
        at.run()

    offenders = [
        str(w.message) for w in caught
        if issubclass(w.category, ResourceWarning) and "sample-data" in str(w.message)
    ]
    assert not offenders, offenders


def test_style_helpers_are_self_consistent():
    """The appearance panel and the renderer must agree on their keys."""
    import app

    style = app._default_style()
    for key in ("font_family", "font_size", "palette", "legend_position",
                "line_width", "marker_size", "marker_symbol", "fit_dash",
                "show_grid", "mirror_axes", "n_ticks"):
        assert key in style, key

    for name, colours in app._PALETTES.items():
        assert len(colours) >= 6, name
        assert all(c.startswith("#") and len(c) == 7 for c in colours), name

    for position in app._LEGEND_POSITIONS:
        assert position == "hidden" or position in app._LEGEND_ANCHORS, position


def test_apply_plot_style_produces_a_renderable_figure():
    import app
    import plotly.graph_objects as go

    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    app.apply_plot_style(fig, app._default_style(), "x", "y", title="t")
    assert fig.layout.font.family == "Arial"
    assert fig.to_json()  # serialisable, i.e. exportable


def test_export_formats_are_declared_consistently():
    import app

    for label, (ext, mime) in app._EXPORT_FORMATS.items():
        assert ext and mime and "/" in mime, label
    assert 300 in app._EXPORT_DPI_CHOICES
