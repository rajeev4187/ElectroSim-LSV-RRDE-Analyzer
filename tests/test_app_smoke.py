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
    assert fig.layout.font.family == app._font_stack("Arial")
    assert fig.to_json()  # serialisable, i.e. exportable


def test_export_formats_are_declared_consistently():
    import app

    for label, (ext, mime) in app._EXPORT_FORMATS.items():
        assert ext and mime and "/" in mime, label
    assert 300 in app._EXPORT_DPI_CHOICES


def test_screen_canvas_prints_at_its_stated_width_at_every_dpi():
    """Journals specify a width in centimetres. Kaleido sizes figures in CSS
    pixels (1/96 in) and then multiplies by scale = dpi/96, so the physical
    width follows from the CSS size alone and dpi only decides how many pixels
    fill it. Converting cm to pixels *at dpi* double-counts the resolution --
    a 150 dpi single-column figure came out 13.4 cm instead of the 8.6 cm
    asked for."""
    import app

    for width_cm in app._JOURNAL_WIDTHS_CM.values():
        if width_cm is None:  # "As shown on screen"
            continue
        css_w = app._cm_to_css_px(width_cm)
        for dpi in app._EXPORT_DPI_CHOICES:
            scale = max(1.0, dpi / app._CSS_PX_PER_INCH)
            printed_cm = (css_w * scale) / dpi * 2.54
            assert printed_cm == pytest.approx(width_cm, abs=0.05), (
                f"{width_cm} cm printed {printed_cm:.2f} cm at {dpi} dpi"
            )

    # The canvas every chart is drawn on is one of those journal widths.
    assert app._SCREEN_CANVAS_W == app._cm_to_css_px(app._SCREEN_CANVAS_W_CM)


def test_browser_discovery_collects_more_than_one_candidate():
    """Kaleido 1.x drives a real browser and finds one from a fixed path list
    built, on Windows, out of %PROGRAMFILES% and friends -- so a server whose
    environment lacks those reports "requires Google Chrome to be installed"
    on a machine that has Chrome. The app looks browsers up itself and keeps
    every one it finds, because any single browser can also just fail to
    launch."""
    import app

    import os

    paths = app._browser_paths()
    assert isinstance(paths, tuple)
    assert len(set(paths)) == len(paths), "duplicate browsers in the list"
    for path in paths:
        assert os.path.isfile(path), f"{path} does not exist"
    # _browser_path is the one that would be used, for reporting.
    assert app._browser_path() == (paths[0] if paths else None)


def test_render_falls_through_to_the_next_browser():
    """A dead first browser must not end the export: every candidate is tried
    before giving up, and the original error is what surfaces if none work."""
    import plotly.graph_objects as go

    import app

    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    boom = RuntimeError("Kaleido requires Google Chrome to be installed.")
    calls = []

    def explode(**kwargs):
        raise boom

    def fake_calc(export_fig, opts, kopts):
        calls.append(kopts["path"])
        if kopts["path"] == "/nonexistent/chrome":
            raise RuntimeError("could not launch")
        return b"PNGBYTES"

    kaleido = pytest.importorskip("kaleido")
    real_calc = kaleido.calc_fig_sync
    real_paths = app._browser_paths
    fig.to_image = explode  # type: ignore[method-assign]
    kaleido.calc_fig_sync = fake_calc
    app._browser_paths = lambda: ("/nonexistent/chrome", "/second/chrome")
    try:
        assert app._kaleido_render(fig, {"format": "png"}) == b"PNGBYTES"
        assert calls == ["/nonexistent/chrome", "/second/chrome"], (
            "should try each browser in turn"
        )

        # Nothing available at all -> the original failure is re-raised, so the
        # message the user sees is kaleido's own rather than a masking one.
        app._browser_paths = lambda: ()
        with pytest.raises(RuntimeError) as excinfo:
            app._kaleido_render(fig, {"format": "png"})
        assert excinfo.value is boom
    finally:
        kaleido.calc_fig_sync = real_calc
        app._browser_paths = real_paths


def test_fonts_name_fallbacks_the_export_renderer_can_actually_find():
    """The live chart is drawn by the user's browser, the TIFF by kaleido's
    headless Chromium on the server -- often a slim Linux container with no
    Arial or Calibri. A bare family name leaves that renderer substituting
    freely, and glyphs outside the substitute's coverage (log₁₀, Ω, −Z″,
    %H₂O₂) come out as tofu boxes in the download while the on-screen chart
    looks perfect."""
    import app

    for face in app._JOURNAL_FONTS:
        stack = app._font_stack(face)
        assert stack.startswith(face), f"{face} is not first in its own stack"
        assert stack.count(",") >= 2, f"{face} has no real fallbacks"
        assert stack.rsplit(",", 1)[-1].strip() in (
            "sans-serif", "serif", "monospace"
        ), f"{face} ends in no generic family"

    # An unknown face still gets fallbacks rather than being passed through bare.
    assert "," in app._font_stack("Some Unshipped Face")
    assert app._font_stack(None).startswith("Arial")


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
    """The whole export path, end to end: a figure must come out at its own
    canvas width in centimetres at every dpi, and every raster format must
    carry a dpi tag.

    Both halves regressed before: the width presets converted cm to pixels *at
    dpi* while kaleido separately multiplies by dpi/96, so a 150 dpi figure
    came out 13.4 cm; and only TIFF went through Pillow, so PNG/JPEG claimed
    72 dpi however they were rendered."""
    import io as _io

    import plotly.graph_objects as go
    from PIL import Image

    import app

    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, -5], mode="lines"))
    app.apply_plot_style(fig, {**app._default_style(), "show_title": False},
                         "Potential vs RHE / V", "Current (mA/cm2)")
    width, height = app._export_size(fig, None, None)
    data = app._render_export(fig, fmt, width, height, dpi)
    assert data

    if fmt in ("svg", "pdf"):
        return
    image = Image.open(_io.BytesIO(data))
    assert image.info.get("dpi"), f"{fmt} carries no dpi tag"
    assert float(image.info["dpi"][0]) == pytest.approx(dpi, abs=1)
    printed_cm = image.size[0] / dpi * 2.54
    assert printed_cm == pytest.approx(app._SCREEN_CANVAS_W_CM, abs=0.05)

    # No alpha channel in any raster export. Kaleido's PNG always carries one
    # and it is always fully opaque here, but a 4-sample TIFF tagged
    # ExtraSamples=2 is what Word, the Windows photo viewer and several
    # manuscript-submission converters render as black or with the colour
    # channels shifted -- the "the TIFF is garbled" report.
    assert image.mode == "RGB", f"{fmt} exported as {image.mode}, not RGB"
    if fmt == "tiff":
        assert image.tag_v2.get(277) == 3, "TIFF has an extra (alpha) sample"
        assert image.tag_v2.get(338) is None, "TIFF declares ExtraSamples"


@pytest.mark.skipif(not _kaleido_available(),
                    reason="kaleido/headless browser not available")
def test_export_defaults_to_the_size_the_figure_is_shown_at():
    """The download must be the figure that was on screen, at the size it was
    displayed. Charts used to stretch to the browser width while the export
    fell back to a square canvas, and since fonts, margins and legend anchors
    are all in absolute pixels that is a different layout, not a rescale: the
    saved figure had clipped axis titles and overlapping tick labels."""
    import plotly.graph_objects as go

    import app

    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, -5], mode="lines"))
    app.apply_plot_style(fig, app._default_style(), "x", "y")

    assert fig.layout.width == app._SCREEN_CANVAS_W
    assert app._export_size(app._export_figure(fig), None, None) == (
        fig.layout.width, fig.layout.height
    )
    # And at that canvas the margins leave room for the plot itself.
    assert app._margin_crowding(fig, fig.layout.width, fig.layout.height) is None
