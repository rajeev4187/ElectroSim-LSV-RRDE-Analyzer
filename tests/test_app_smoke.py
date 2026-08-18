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


def test_preset_export_size_hits_the_requested_column_width_at_every_dpi():
    """Journals specify a width in centimetres. Kaleido sizes figures in CSS
    pixels (1/96 in) and then multiplies by scale = dpi/96, so the physical
    width follows from the CSS size alone. Converting cm to pixels *at dpi*
    double-counts the resolution -- a 150 dpi single-column figure came out
    13.4 cm instead of the 8.6 cm asked for."""
    import app

    for width_cm in app._JOURNAL_WIDTHS_CM.values():
        if width_cm is None:  # "As shown on screen"
            continue
        css_w, css_h = app._preset_export_size(width_cm, 520, 520)
        assert css_h == css_w  # a square figure stays square
        for dpi in app._EXPORT_DPI_CHOICES:
            scale = max(1.0, dpi / app._CSS_PX_PER_INCH)
            printed_cm = (css_w * scale) / dpi * 2.54
            assert printed_cm == pytest.approx(width_cm, abs=0.05), (
                f"{width_cm} cm preset printed {printed_cm:.2f} cm at {dpi} dpi"
            )

    wide_w, wide_h = app._preset_export_size(17.8, 930, 680)
    assert wide_h / wide_w == pytest.approx(680 / 930, rel=1e-3)


def test_html_export_escapes_markup_in_user_supplied_names():
    """Sample names, legend names and file names are user strings that end up
    inside the ``<script>`` payload of the HTML export. Plotly escapes them,
    but the export is a file the user then sends to a co-author, so pin the
    behaviour: a downgrade that stopped escaping would be a real hole."""
    import plotly.graph_objects as go

    hostile = '</script><img src=x onerror="alert(1)">'
    fig = go.Figure(go.Scatter(x=[1, 2], y=[1, 2], name=hostile))
    fig.update_layout(title=hostile)
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)
    assert "</script><img" not in html
    assert "onerror=\"alert(1)\"" not in html


def _kaleido_available() -> bool:
    """Whether static image export actually works here (kaleido needs a
    headless browser, which is not present on every machine or CI runner)."""
    import plotly.graph_objects as go

    try:
        go.Figure().to_image(format="png", width=200, height=200)
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kaleido_available(),
                    reason="kaleido/headless browser not available")
@pytest.mark.parametrize("dpi", [150, 300, 600])
@pytest.mark.parametrize("fmt", ["tiff", "png", "jpeg", "svg", "pdf"])
def test_export_renders_at_the_requested_physical_size(fmt, dpi):
    """The whole export path, end to end: a single-column preset must come out
    8.6 cm wide at every dpi, and every raster format must carry a dpi tag.

    Both halves regressed before: the presets converted cm to pixels *at dpi*
    while kaleido separately multiplies by dpi/96, so a 150 dpi figure came out
    13.4 cm; and only TIFF went through Pillow, so PNG/JPEG claimed 72 dpi
    however they were rendered."""
    import io as _io

    import plotly.graph_objects as go
    from PIL import Image

    import app

    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, -5], mode="lines"))
    app.apply_plot_style(fig, {**app._default_style(), "show_title": False},
                         "Potential vs RHE / V", "Current (mA/cm2)")
    width, height = app._preset_export_size(8.6, 520, 520)
    data = app._render_export(fig, fmt, width, height, dpi)
    assert data

    if fmt in ("svg", "pdf"):
        return
    image = Image.open(_io.BytesIO(data))
    assert image.info.get("dpi"), f"{fmt} carries no dpi tag"
    assert float(image.info["dpi"][0]) == pytest.approx(dpi, abs=1)
    printed_cm = image.size[0] / dpi * 2.54
    assert printed_cm == pytest.approx(8.6, abs=0.05)
