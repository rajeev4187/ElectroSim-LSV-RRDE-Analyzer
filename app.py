"""ElectroSim-LSV-RRDE-Analyzer: a Streamlit GUI for iR compensation,
Tafel-slope, Koutecky-Levich, and ORR/RRDE analysis.

Run with:
    streamlit run app.py

Workflow
--------
1. Upload an Excel workbook (Sheet 1 = EIS, Sheet 2 = LSV) or two CSV files.
2. **EIS / Ru Analysis** tab: fit the Nyquist arc to extract Ru (and Rct).
3. **LSV iR Correction** tab: apply the ohmic-drop correction with a
   compensation factor selectable from 5 % to 100 %; download the result.
4. **LSV Analysis** tab: independent of the above — upload its own
   polarization-curve file(s) and get the onset potential, overpotential at
   benchmark current densities (e.g. j = 10 mA/cm²), and the Tafel slope of
   the linear (kinetic) region.
5. **K-L Analysis** tab: independent, multi-rotation-rate RDE data — the
   classic Koutecky-Levich fit (1/j vs 1/sqrt(omega)) for the kinetic current
   density and, given the electrolyte's O2 transport parameters, the
   electron-transfer number n.
6. **ORR / RRDE Analysis** tab: independent, one rotation rate (usually 1600
   rpm) for onset/E1/2/Tafel plus (with ring current) electron number and
   peroxide yield directly; several rotation rates for a merged ring/disk
   comparison plot.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from plotly.subplots import make_subplots

from scripts.modules import correction, data_io, eis, orr, sweep, tafel

APP_NAME = "ElectroSim-LSV-RRDE-Analyzer"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡", layout="wide",
)

EIS_SAMPLE_PATH = "sample-data/EIS example.xlsx"
LSV_SAMPLE_PATH = "sample-data/LSV Example.xlsx"
REPO_URL = "https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope"
CITATION_TEXT = (
    f"Kumar, R. (2026). {APP_NAME} "
    "(v1.1.0) [Computer software]. North Carolina Central "
    f"University. {REPO_URL}"
)
CITATION_BIBTEX = (
    "@software{kumar_electrosim_lsv_rrde_2026,\n"
    "  author  = {Kumar, Rajeev},\n"
    f"  title   = {{{APP_NAME}}},\n"
    "  version = {1.1.0},\n"
    "  year    = {2026},\n"
    f"  url     = {{{REPO_URL}}}\n"
    "}"
)


def render_citation() -> None:
    """Show a 'cite this app' block in the sidebar."""
    st.sidebar.divider()
    with st.sidebar.expander("📚 Cite this app"):
        st.markdown(
            "If this tool supports your work, please cite it "
            f"([repository]({REPO_URL})):"
        )
        st.markdown(f"> {CITATION_TEXT}")
        st.code(CITATION_BIBTEX, language="bibtex")
        st.caption(
            "Machine-readable metadata: CITATION.cff. A versioned release can "
            "be archived on Zenodo to obtain a citable DOI."
        )


def require_access() -> bool:
    """Optional password gate.

    Open by default. If an ``app_password`` secret is configured (via
    ``.streamlit/secrets.toml`` locally or the app's Secrets on Streamlit
    Cloud), visitors must enter it before using the app. Uses a constant-time
    comparison to avoid timing side-channels and never echoes the password.
    """
    try:
        expected = st.secrets["app_password"]
    except Exception:
        return True  # no password set -> public access

    if st.session_state.get("authed"):
        return True

    st.title(f"⚡ {APP_NAME}")
    st.text_input("Password", type="password", key="_pw")
    if st.button("Enter"):
        if hmac.compare_digest(str(st.session_state.get("_pw", "")),
                               str(expected)):
            st.session_state["authed"] = True
            del st.session_state["_pw"]
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


def _figure_signature(fig) -> str:
    """Short fingerprint of a figure's current content.

    Used to detect that a cached raster export no longer matches the figure on
    screen (e.g. the user moved a fit-range slider after preparing the
    export), so a stale image is never handed out as a download.
    """
    try:
        payload = fig.to_json()
    except Exception:
        payload = repr(fig)
    return hashlib.sha1(payload.encode("utf-8", "replace")).hexdigest()


# First entry of the colorway in Streamlit's own Plotly template. That
# template paints with placeholder colours (#000001, #000036, …) which the
# Streamlit *frontend* swaps for the real theme colours while drawing the
# chart in the browser. Anything rendered outside that frontend — a
# server-side TIFF, or a standalone HTML file — takes them literally, which is
# what turned exported figures (most visibly the results table, which sets no
# template of its own) into a near-black block.
_STREAMLIT_TEMPLATE_SENTINEL = "#000001"


def _export_figure(fig):
    """Return a copy of ``fig`` that is safe to render outside Streamlit.

    Figures that already pin a real template (all the plots here use
    ``plotly_white``) are returned untouched; one still carrying Streamlit's
    placeholder-colour template is re-rendered on ``plotly_white``.
    """
    template = getattr(fig.layout, "template", None)
    colorway = getattr(getattr(template, "layout", None), "colorway", None)
    if not colorway or str(colorway[0]).lower() != _STREAMLIT_TEMPLATE_SENTINEL:
        return fig
    clone = go.Figure(fig)
    clone.update_layout(template="plotly_white")
    return clone


def _export_size(fig, width: int | None, height: int | None) -> tuple[int, int]:
    """Pixel size for a static export.

    Square by default (the common preference for journal figures): width
    matches height unless the figure (or caller) pins its own width — e.g.
    a table (sized by column count) or a deliberately multi-panel figure
    (the merged Ring/Disk plot) that already sets both dimensions itself.
    Height defaults to the figure's own layout height when it sets one — a
    table sizes itself by row count, so forcing the same fixed height on
    every figure (as this used to) simply cut the last rows off.
    """
    if height is None:
        height = int(getattr(fig.layout, "height", None) or 520)
    if width is None:
        layout_width = getattr(fig.layout, "width", None)
        width = int(layout_width) if layout_width else height
    return int(width), max(int(height), 200)


# A zip of a data folder holds tens of files, not thousands. The total
# uncompressed size is already bounded, but a zip of a hundred thousand tiny
# members passes that check and then makes the app parse each one -- so bound
# the count too, and say so rather than grinding for minutes.
MAX_ZIP_MEMBERS = 500


# Journal figure widths. Almost every publisher specifies a figure by the
# column width it must fit, in centimetres, and there was previously no way to
# hit one: the size boxes are in pixels, and what they mean physically is not
# obvious (the old 520 px square comes out 13.8 cm wide -- between a single and
# a double column, matching neither).
_JOURNAL_WIDTHS_CM: dict[str, float | None] = {
    "As shown on screen": None,
    "Single column (8.6 cm)": 8.6,
    "1.5 column (12.0 cm)": 12.0,
    "Double column (17.8 cm)": 17.8,
}

# Kaleido's width/height are **CSS pixels**, which are 1/96 inch by definition,
# and it multiplies them by `scale` to get the output raster. Since
# `_render_export` sets scale = dpi/96, the output is width*dpi/96 pixels =
# width/96 inches at that dpi. So the physical size of a figure is fixed by its
# CSS size alone, and dpi only buys resolution within it -- which is exactly
# the separation a journal asks for ("8.6 cm wide, at least 300 dpi").
# Converting cm to pixels *at dpi* here would therefore double-count the
# resolution: a 150 dpi single-column figure came out 13.4 cm instead of 8.6.
_CSS_PX_PER_INCH = 96


def _margin_crowding(fig, out_w: int, out_h: int) -> tuple[str, float] | None:
    """``(axis, fraction)`` when the margins eat too much of the canvas.

    Returns the worse of the two axes when either loses more than 55 % of its
    length to margins, else ``None``.
    """
    margin = getattr(fig.layout, "margin", None)
    if margin is None:
        return None

    def _side(value, default):
        return float(default if value is None else value)

    horizontal = _side(margin.l, 0) + _side(margin.r, 0)
    vertical = _side(margin.t, 0) + _side(margin.b, 0)
    worst = max(
        ("width", horizontal / out_w if out_w else 0.0),
        ("height", vertical / out_h if out_h else 0.0),
        key=lambda pair: pair[1],
    )
    return worst if worst[1] > 0.55 else None


def _preset_export_size(width_cm: float,
                        base_w: int, base_h: int) -> tuple[int, int]:
    """CSS-pixel size for a ``width_cm``-wide figure, keeping the figure's own
    aspect ratio so nothing is stretched. Independent of dpi -- see above."""
    out_w = int(round(width_cm / 2.54 * _CSS_PX_PER_INCH))
    aspect = (base_h / base_w) if base_w else 1.0
    return out_w, int(round(out_w * aspect))


# Raster/vector formats offered for figure export. TIFF is what most journals
# require for submission; the rest cover the common alternatives (SVG/PDF stay
# vector, so they scale without resampling).
_EXPORT_FORMATS: dict[str, tuple[str, str]] = {
    "TIFF (LZW)": ("tiff", "image/tiff"),
    "PNG": ("png", "image/png"),
    "SVG (vector)": ("svg", "image/svg+xml"),
    "PDF (vector)": ("pdf", "application/pdf"),
    "JPEG": ("jpeg", "image/jpeg"),
}
_EXPORT_DPI_CHOICES = [150, 300, 600, 1200]


def _render_export(export_fig, fmt: str, width: int, height: int,
                   dpi: int = 300) -> bytes:
    """Render a figure to ``fmt`` at ``dpi``.

    Kaleido rasterises at a CSS-pixel size scaled by ``scale``; converting the
    requested dpi into that scale factor (relative to the nominal 96 dpi of a
    CSS pixel) is what makes the dpi selector actually change the pixel count
    rather than only the metadata. TIFF is not a Kaleido output format, so it
    is rendered as a high-resolution PNG and converted in-memory by Pillow,
    which is also where the LZW compression is applied.

    Every raster format goes back through Pillow to carry a **dpi tag**.
    Kaleido writes none, so a 600-dpi PNG used to arrive claiming the
    default 72 dpi -- which is the number the submission system and the
    typesetter read, and the reason an otherwise correct figure comes back
    flagged as too low-resolution.
    """
    if fmt in ("svg", "pdf"):
        # Vector: dpi is meaningless. Kaleido sizes these in CSS pixels, the
        # same unit as the raster paths (not in points, despite the sizes
        # looking point-like at typical figure dimensions).
        return export_fig.to_image(format=fmt, width=width, height=height)

    scale = max(1.0, float(dpi) / 96.0)
    png_bytes = export_fig.to_image(format="png", width=width, height=height,
                                    scale=scale)
    image = Image.open(io.BytesIO(png_bytes))
    buffer = io.BytesIO()
    if fmt == "tiff":
        image.save(buffer, format="TIFF", dpi=(dpi, dpi), compression="tiff_lzw")
    elif fmt in ("jpeg", "jpg"):
        # JPEG has no alpha channel; a transparent PNG would otherwise raise.
        image.convert("RGB").save(buffer, format="JPEG", dpi=(dpi, dpi),
                                  quality=95, subsampling=0)
    else:
        image.save(buffer, format="PNG", dpi=(dpi, dpi))
    return buffer.getvalue()


def _render_tiff(export_fig, width: int, height: int, dpi: int = 300) -> bytes:
    """Back-compatible shim for the TIFF path (see :func:`_render_export`)."""
    return _render_export(export_fig, "tiff", width, height, dpi)


def figure_downloads(fig, stem: str, key: str, what: str = "figure",
                     width: int | None = None, height: int | None = None,
                     data: pd.DataFrame | None = None) -> None:
    """Render the download controls for one figure: a chosen image format,
    interactive HTML, and (optionally) the plotted data as CSV.

    Image rendering is server-side (kaleido + Pillow), which launches a
    headless browser per call -- too slow/fragile to run on *every* script
    rerun (it would fire on every unrelated widget interaction, e.g. dragging
    a slider). It only happens when the user clicks "Prepare"; the bytes are
    cached in session_state together with a signature of the figure, so the
    download button persists across reruns but is withdrawn as soon as the
    figure itself changes.

    The HTML and CSV exports need no external renderer and are therefore
    always available -- they are the fallback when kaleido/Chrome is
    unavailable on the host (e.g. a slim cloud container). Both are built
    **only when the user opens the download expander**: serialising a
    multi-thousand-point figure to a self-contained HTML page on every rerun
    of every figure was costing far more than the charts themselves.
    """
    state_key = f"_export_{key}"
    with st.expander(f"⬇️ Download {what}"):
        export_fig = _export_figure(fig)
        cols = st.columns(3 if data is not None else 2)

        with cols[0]:
            fmt_label = st.selectbox(
                "Image format", list(_EXPORT_FORMATS), index=0,
                key=f"_fmt_{key}",
                help="TIFF at 300 dpi is the usual journal submission "
                     "requirement; SVG/PDF stay vector and rescale cleanly.",
            )
            fmt, mime = _EXPORT_FORMATS[fmt_label]
            dpi = 300
            if fmt not in ("svg", "pdf"):
                dpi = st.selectbox(
                    "Resolution (dpi)", _EXPORT_DPI_CHOICES,
                    index=_EXPORT_DPI_CHOICES.index(300), key=f"_dpi_{key}",
                    help="300 dpi is the common minimum for halftone figures; "
                         "600+ for line art in high-quality journals.",
                )
            base_w, base_h = _export_size(export_fig, width, height)
            # A caller that pins its own width has a layout reason for it (a
            # table sized by column count, a two-panel figure): re-flowing
            # those to a column width clips their contents, so they default
            # to their own size and the presets stay available but unchosen.
            preset = st.selectbox(
                "Figure width", list(_JOURNAL_WIDTHS_CM),
                index=0 if width is not None else 1,
                key=f"_wpreset_{key}",
                help="Journals specify figures by the column width they must "
                     "fit. The pixel size below follows from that width and "
                     "the dpi; override it if you need an exact size.",
            )
            width_cm = _JOURNAL_WIDTHS_CM[preset]
            if width_cm is not None:
                base_w, base_h = _preset_export_size(width_cm, base_w, base_h)
            # The preset is part of the widget key, not just its default:
            # Streamlit keeps a number_input's stored value across reruns, so
            # a changed default alone would never reach the widget and picking
            # a preset would appear to do nothing.
            sc1, sc2 = st.columns(2)
            out_w = sc1.number_input(
                "Width (px)", min_value=200, max_value=4000,
                value=int(base_w), step=20, key=f"_w_{key}_{preset}",
            )
            out_h = sc2.number_input(
                "Height (px)", min_value=200, max_value=4000,
                value=int(base_h), step=20, key=f"_h_{key}_{preset}",
            )
            scale = max(1.0, float(dpi) / _CSS_PX_PER_INCH)
            st.caption(
                f"↳ Figure size "
                f"{out_w / _CSS_PX_PER_INCH * 2.54:.1f} × "
                f"{out_h / _CSS_PX_PER_INCH * 2.54:.1f} cm; exported at "
                f"{dpi} dpi that is "
                f"{int(round(out_w * scale))} × {int(round(out_h * scale))} "
                "pixels."
            )
            # Margins are in absolute pixels (they have to be: they hold text
            # of a fixed point size). On a narrow canvas they can therefore
            # take most of the figure, leaving a strip of plot inside a wide
            # white border. That is a font/width mismatch, not a bug, but it
            # is invisible until the file is opened -- so say it here.
            crowding = _margin_crowding(export_fig, int(out_w), int(out_h))
            if crowding is not None:
                axis, used = crowding
                st.warning(
                    f"At this size the margins take {used:.0%} of the "
                    f"figure's {axis}, leaving little room for the plot "
                    "itself. The axis text is sized for a larger canvas: "
                    "lower **Font size** in the plot-appearance panel, or "
                    "choose a wider column."
                )
            # The prepared bytes depend on the render settings as much as on
            # the figure, so they are part of the cache identity. Keying on
            # the figure alone served stale bytes -- with the *old* format's
            # extension and MIME type -- to anyone who changed format, dpi or
            # size and downloaded without pressing Prepare again.
            settings = (fmt, int(dpi), int(out_w), int(out_h))
            if st.button(f"🖼️ Prepare {what}", key=f"_img_prep_{key}",
                         width="stretch"):
                try:
                    st.session_state[state_key] = {
                        "sig": _figure_signature(export_fig),
                        "settings": settings,
                        "bytes": _render_export(export_fig, fmt, int(out_w),
                                                int(out_h), int(dpi)),
                        "ext": fmt, "mime": mime, "error": None,
                    }
                except Exception as exc:  # kaleido missing / no Chrome
                    st.session_state[state_key] = {
                        "sig": _figure_signature(export_fig),
                        "settings": settings, "bytes": None,
                        "ext": fmt, "mime": mime, "error": str(exc),
                    }
            cached = st.session_state.get(state_key) or {}
            if cached.get("bytes") or cached.get("error"):
                # Only now is the signature worth computing -- it serialises
                # the whole figure, so doing it unconditionally on every rerun
                # was pure overhead for figures nobody exports.
                sig = _figure_signature(export_fig)
                stale_settings = cached.get("settings") != settings
                fresh = (cached.get("bytes") and cached.get("sig") == sig
                         and not stale_settings)
                if fresh:
                    st.download_button(
                        f"⬇️ {what} ({cached['ext'].upper()})",
                        data=cached["bytes"],
                        file_name=f"{stem}.{cached['ext']}",
                        mime=cached["mime"], key=f"_img_dl_{key}",
                        width="stretch",
                    )
                elif cached.get("bytes"):
                    st.caption(
                        "↻ Export settings changed — press Prepare again."
                        if stale_settings else
                        "↻ Figure changed — press Prepare again."
                    )
                elif cached.get("error"):
                    st.caption(
                        f"Image export unavailable ({cached['error']}). Use "
                        "the HTML download beside this button, or the 📷 icon "
                        "on the chart."
                    )

        with cols[1]:
            try:
                html = export_fig.to_html(include_plotlyjs="cdn", full_html=True)
                st.download_button(
                    f"⬇️ {what} (HTML)", data=html.encode("utf-8"),
                    file_name=f"{stem}.html", mime="text/html",
                    key=f"_html_dl_{key}", width="stretch",
                    help="Interactive page — opens in any browser (needs "
                         "internet the first time, it loads plotly.js from a "
                         "CDN) and can be saved as an image from there. "
                         "Always available, even when the server-side image "
                         "renderer is not.",
                )
            except Exception as exc:
                st.caption(f"HTML export unavailable ({exc}).")

        if data is not None:
            with cols[2]:
                st.download_button(
                    "⬇️ Plotted data (CSV)",
                    data=data.to_csv(index=False).encode("utf-8"),
                    file_name=f"{stem}_data.csv", mime="text/csv",
                    key=f"_csv_dl_{key}", width="stretch",
                    help="Exactly the series drawn above, for replotting in "
                         "Origin/Excel.",
                )


def _padded_frame(columns: dict[str, list]) -> pd.DataFrame:
    """Build a DataFrame from unequal-length columns (each plotted series has
    its own point count), padding the short ones with blanks so the CSV keeps
    one column pair per series — the layout Origin/Excel expect."""
    n = max((len(v) for v in columns.values()), default=0)
    return pd.DataFrame({
        k: list(v) + [""] * (n - len(v)) for k, v in columns.items()
    })


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
# Parsing is cached on the file's *content*, not on the widget: Streamlit
# re-runs this whole script on every interaction (each slider drag, each
# checkbox), and without a cache every rerun re-parsed every uploaded
# workbook from scratch. Hashing the bytes means an unchanged upload is
# parsed exactly once no matter how much the user fiddles with the controls,
# while genuinely new content still invalidates immediately.
_CACHE_TTL_S = 3600


@st.cache_data(ttl=_CACHE_TTL_S, show_spinner=False, max_entries=64)
def _cached_sheets(payload: bytes) -> list[str]:
    return data_io.list_sheets(io.BytesIO(payload))


@st.cache_data(ttl=_CACHE_TTL_S, show_spinner=False, max_entries=64)
def _cached_eis_datasets(payload: bytes, sheet):
    return data_io.load_eis_datasets(io.BytesIO(payload), sheet=sheet)


@st.cache_data(ttl=_CACHE_TTL_S, show_spinner=False, max_entries=64)
def _cached_lsv_datasets(payload: bytes, sheet):
    return data_io.load_lsv_datasets(io.BytesIO(payload), sheet=sheet)


@st.cache_data(ttl=_CACHE_TTL_S, show_spinner=False, max_entries=64)
def _cached_table(payload: bytes, sheet) -> pd.DataFrame:
    return data_io.read_table(io.BytesIO(payload), sheet=sheet)


@st.cache_data(ttl=_CACHE_TTL_S, show_spinner=False, max_entries=8)
def _cached_sample_bytes(path: str) -> bytes:
    """Read a bundled sample file once per session rather than per rerun."""
    with open(path, "rb") as fh:
        return fh.read()


def _clear_reset_control():
    """Sidebar button to wipe all uploaded data / session state.

    Shared by the EIS/Ru and LSV/iR tabs (and everything else keyed off
    ``uploader_nonce``) since it resets every uploader in the app at once.
    """
    st.sidebar.header("Data controls")
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0
    if st.sidebar.button("🗑️ Clear / reset files",
                         help="Remove uploaded files and start over."):
        st.session_state.uploader_nonce += 1
        for k in list(st.session_state.keys()):
            if k not in ("uploader_nonce", "authed"):
                del st.session_state[k]
        st.rerun()


def eis_data_loader():
    """Render the EIS/Ru tab's own data-source controls.

    Independent of the LSV/iR tab's loader (each tab uploads its own file,
    matching the LSV Analysis and RRDE tabs) but still *linked* to it: the
    sidebar "Sample" selector below pairs EIS dataset *i* with LSV dataset
    *i* by position once both are loaded. Returns ``(eis_list, name)``.
    """
    st.markdown(
        "**Data source** — EIS spectrum (Z', Z'' pairs), used to extract Ru. "
        "Paired with the LSV iR Correction tab's data by sample position."
    )
    nonce = st.session_state.get("uploader_nonce", 0)
    source = st.radio(
        "Choose input", ["Upload Excel workbook", "Upload CSV file",
                         "Use bundled sample"],
        horizontal=True, key="eis_source",
        help="Z', Z'' pairs. Several datasets may sit side-by-side as "
             "repeated column pairs.",
    )
    try:
        if source in ("Use bundled sample", "Upload Excel workbook"):
            if source == "Use bundled sample":
                payload, name = _cached_sample_bytes(EIS_SAMPLE_PATH), EIS_SAMPLE_PATH
            else:
                up = st.file_uploader(
                    "Excel (.xlsx)", type=["xlsx", "xls"], key=f"eis_xlsx_{nonce}"
                )
                if up is None:
                    st.info("⬆️ Upload an Excel workbook to begin.")
                    return None, None
                payload, name = up.getvalue(), up.name
            sheets = _cached_sheets(payload)
            sheet = st.selectbox("EIS sheet", sheets, index=0, key="eis_sheet")
            return _cached_eis_datasets(payload, sheet), name

        up = st.file_uploader(
            "EIS CSV (Z', Z'')", type=["csv", "txt"], key=f"eis_csv_{nonce}"
        )
        if up is None:
            st.info("⬆️ Upload a CSV file to begin.")
            return None, None
        return _cached_eis_datasets(up.getvalue(), None), up.name

    except Exception as exc:  # surface loader errors instead of a stack trace
        st.error(f"Could not load EIS data: {exc}")
        return None, None


def lsv_data_loader():
    """Render the LSV iR Correction tab's own data-source controls.

    Mirrors :func:`eis_data_loader` — see its docstring for the pairing
    ("linked") behaviour. Returns ``(lsv_list, name)``.
    """
    st.markdown(
        "**Data source** — LSV curve (Potential, Current pairs), corrected "
        "using the Ru fitted in the EIS/Ru Analysis tab. Paired with that "
        "tab's data by sample position."
    )
    nonce = st.session_state.get("uploader_nonce", 0)
    source = st.radio(
        "Choose input", ["Upload Excel workbook", "Upload CSV file",
                         "Use bundled sample"],
        horizontal=True, key="lsv_source",
        help="Potential, Current pairs. Several datasets may sit side-by-side "
             "as repeated column pairs.",
    )
    try:
        if source in ("Use bundled sample", "Upload Excel workbook"):
            if source == "Use bundled sample":
                payload, name = _cached_sample_bytes(LSV_SAMPLE_PATH), LSV_SAMPLE_PATH
            else:
                up = st.file_uploader(
                    "Excel (.xlsx)", type=["xlsx", "xls"], key=f"lsv_xlsx_{nonce}"
                )
                if up is None:
                    st.info("⬆️ Upload an Excel workbook to begin.")
                    return None, None
                payload, name = up.getvalue(), up.name
            sheets = _cached_sheets(payload)
            sheet = st.selectbox(
                "LSV sheet", sheets, index=min(1, len(sheets) - 1), key="lsv_sheet"
            )
            return _cached_lsv_datasets(payload, sheet), name

        up = st.file_uploader(
            "LSV CSV (Potential, Current)", type=["csv", "txt"],
            key=f"lsv_csv_{nonce}",
        )
        if up is None:
            st.info("⬆️ Upload a CSV file to begin.")
            return None, None
        return _cached_lsv_datasets(up.getvalue(), None), up.name

    except Exception as exc:  # surface loader errors instead of a stack trace
        st.error(f"Could not load LSV data: {exc}")
        return None, None


# --------------------------------------------------------------------------- #
# Units & electrode area                                                      #
# --------------------------------------------------------------------------- #
_ABS_CURRENT_UNITS = ["mA", "A", "µA", "nA"]
_DENSITY_CURRENT_UNITS = ["mA/cm²", "A/cm²", "µA/cm²", "nA/cm²"]


def _detect_units(eis_label: str, lsv_label: str):
    """Guess (current_unit, ru_unit) from the column-header labels.

    The sample workbook's headers carry units (e.g. ``Current (mA/cm2)``,
    ``Z' (Ohm)``); we read them so the sidebar dropdowns start on the right
    choice. Returns ``None`` for either guess when nothing recognisable is
    found, leaving that control on its default.
    """
    lsv = (lsv_label or "").lower()
    eis = (eis_label or "").lower()
    # Look only at the current half of the LSV label (after "current").
    cur = lsv.split("current", 1)[1] if "current" in lsv else lsv
    density = "cm2" in cur or "cm²" in cur

    if "ma" in cur:
        prefix = "mA"
    elif "µa" in cur or "ua" in cur:
        prefix = "µA"
    elif "na" in cur:
        prefix = "nA"
    elif "a" in cur:
        prefix = "A"
    else:
        prefix = None

    current_unit = None
    if prefix is not None:
        current_unit = f"{prefix}/cm²" if density else prefix

    ru_unit = None
    if "ohm" in eis or "Ω".lower() in eis or "ω" in eis:
        ru_unit = correction.RU_OHM_CM2 if ("cm2" in eis or "cm²" in eis) \
            else correction.RU_OHM
    return current_unit, ru_unit


def sidebar_units(current_default: str | None = None,
                  ru_default: str | None = None):
    """Render the unit / electrode-area controls.

    Returns ``(current_unit, ru_unit, area_cm2)``. The electrode area is only
    collected (and returned non-None) when the LSV current and the EIS
    resistance disagree on area-normalisation; otherwise the units are already
    consistent and area is bypassed.
    """
    st.sidebar.header("Units & electrode area")
    st.sidebar.caption(
        "The ohmic drop I·Ru must come out in volts. Tell the app how the LSV "
        "current and the EIS resistance are reported — an electrode area is "
        "only needed to reconcile a per-area quantity with a non-per-area one."
    )
    if current_default or ru_default:
        st.sidebar.caption(
            "ℹ️ Units pre-filled from the column headers — adjust if needed."
        )

    det_density = bool(current_default) and "cm²" in current_default
    # Section A vs Section B: absolute current vs current density.
    quantity = st.sidebar.radio(
        "LSV current is reported as",
        ["Absolute current (e.g. mA)", "Current density (e.g. mA/cm²)"],
        index=1 if det_density else 0,
        help="Absolute current pairs with Ru in Ω; current density pairs with "
             "Ru in Ω·cm².",
    )
    is_density = quantity.startswith("Current density")
    if is_density:
        opts = _DENSITY_CURRENT_UNITS
    else:
        opts = _ABS_CURRENT_UNITS
    cur_idx = opts.index(current_default) if current_default in opts else 0
    current_unit = st.sidebar.selectbox(
        "Current-density unit" if is_density else "Current unit",
        opts, index=cur_idx,
    )

    ru_opts = list(correction.RU_UNITS)
    ru_idx = ru_opts.index(ru_default) if ru_default in ru_opts else 0
    ru_unit = st.sidebar.selectbox(
        "EIS resistance (Ru) unit", ru_opts, index=ru_idx,
        help="Ω for a raw resistance; Ω·cm² for an area-specific resistance.",
    )

    area_cm2 = None
    if correction.needs_area(current_unit, ru_unit):
        area_cm2 = st.sidebar.number_input(
            "Electrode area (cm²)",
            min_value=1e-4, value=0.04, step=0.01, format="%.4f",
            help="Geometric (or active) electrode area used to convert between "
                 "Ω and Ω·cm². Enter YOUR electrode's area — the default "
                 "(0.04 cm²) is just a common value.",
        )
        if is_density:  # current mA/cm² + Ru in Ω
            st.sidebar.caption(
                f"↪ Ru (Ω) × {area_cm2:g} cm² → Ω·cm², then "
                f"{current_unit} → A/cm² (÷1000 for mA) ⇒ iR drop in V."
            )
        else:  # absolute current + Ru in Ω·cm²
            st.sidebar.caption(
                f"↪ Ru (Ω·cm²) ÷ {area_cm2:g} cm² → Ω, then "
                f"{current_unit} → A (÷1000 for mA) ⇒ iR drop in V."
            )
    else:
        st.sidebar.caption(
            f"✓ {current_unit} and Ru in {ru_unit} are already consistent — "
            "no area needed (current scaled to base unit, e.g. mA ÷1000)."
        )
    return current_unit, ru_unit, area_cm2


def render_manual_ru_section(ru_unit: str = "Ω",
                             area_cm2: float | None = None) -> float | None:
    """Let the user type a Ru value straight from the instrument.

    Many potentiostats' own software already fits Ru (e.g. from a built-in
    PEIS routine) — this skips re-fitting the Nyquist arc above and feeds
    that value into the **LSV iR Correction** tab instead. Returns the value
    reconciled into ``ru_unit`` (the same convention :func:`render_eis_tab`
    returns), or ``None`` if the user hasn't opted to use one.
    """
    with st.expander("📥 Enter Ru directly (value from instrument)"):
        st.caption(
            "If your potentiostat already reports Ru — from a built-in EIS "
            "routine, for instance — skip fitting the Nyquist arc above and "
            "type the value here. It links straight into the **LSV iR "
            "Correction** tab, just like the fitted value."
        )
        c1, c2, c3 = st.columns([2, 1, 2])
        with c1:
            manual_val = st.number_input(
                "Ru value", min_value=0.0, value=0.0, step=0.1, format="%.3f",
                key="manual_instrument_ru_val",
            )
        with c2:
            unit_opts = list(correction.RU_UNITS)
            unit_idx = unit_opts.index(ru_unit) if ru_unit in unit_opts else 0
            manual_unit = st.selectbox(
                "Unit", unit_opts, index=unit_idx, key="manual_instrument_ru_unit",
                help="Ω for a raw resistance; Ω·cm² for an area-specific "
                     "resistance. Converted automatically if it differs from "
                     "the EIS resistance unit set in the sidebar.",
            )
        with c3:
            st.markdown("&nbsp;")  # align checkbox with the inputs above
            use_manual = st.checkbox(
                "Use this value for iR correction",
                key="manual_instrument_ru_use",
                help="Overrides any circle-fit or in-tab manual Ru above with "
                     "this instrument value on the LSV iR Correction tab.",
            )
        if not use_manual or manual_val <= 0:
            return None
        try:
            return correction.convert_ru_unit(manual_val, manual_unit,
                                              ru_unit, area_cm2)
        except ValueError as exc:
            st.error(f"Could not convert Ru unit: {exc}")
            return None


# --------------------------------------------------------------------------- #
# EIS / Ru analysis tab                                                       #
# --------------------------------------------------------------------------- #
def render_eis_tab(eis_d, eis_list, sel, ru_unit: str = "Ω",
                   current_unit: str = "mA",
                   area_cm2: float | None = None) -> float | None:
    """Render the EIS analysis tab and return the raw Ru (file's unit).

    The impedance is *displayed* in the unit that pairs with the LSV current
    to give volts: **Ω·cm²** when the current is a density (the raw Ω data is
    multiplied by the electrode area), otherwise **Ω**. The *returned* Ru is
    the raw value in the file's own unit (``ru_unit``), so the LSV tab can show
    both the raw Ru and the area-reconciled "Ru effective".
    """
    # Unit the analysis is reported in (pairs with the current quantity).
    disp_unit = (correction.RU_OHM_CM2
                 if correction.is_density_unit(current_unit)
                 else correction.RU_OHM)
    # Scale factor converting the file's impedance unit to disp_unit
    # (×area for Ω→Ω·cm², ÷area for Ω·cm²→Ω, ×1 when already consistent).
    scale = correction.reconcile_ru(1.0, ru_unit, current_unit, area_cm2)
    zr = eis_d.z_real * scale
    zi = eis_d.z_imag * scale

    st.subheader("EIS — uncompensated resistance (Ru) from the Nyquist arc")
    st.caption(
        f"Sample {sel + 1} · columns: {eis_d.label or 'Z′, Z″'}. The "
        "high-frequency real-axis intercept of the kinetic semicircle is Ru; "
        "adjust the arc range to exclude the low-frequency diffusion tail."
    )
    if scale != 1.0:
        st.caption(
            f"ℹ️ Impedance reported in **{disp_unit}**: raw Z (Ω) × electrode "
            f"area {area_cm2:g} cm² (current is a density)."
        )
    style = plot_style_controls("eis", show_markers=True)
    font_size = style["font_size"]

    n = len(eis_d)
    auto_start, auto_stop = eis.auto_arc_range(zr, zi)

    left, right = st.columns([1, 2])
    with left:
        method = st.radio(
            "Ru method",
            ["Circle fit (recommended)", "Manual value"],
            help="Circle fit extrapolates the arc to the real axis.",
        )
        st.markdown("**Arc points used for the fit**")
        arc = st.slider(
            "Index range (high-frequency first)",
            min_value=0,
            max_value=n,
            value=(int(auto_start), int(auto_stop)),
            help="Points outside this range (the rising diffusion tail) are ignored.",
        )
        start, stop = arc

        ru_result = None
        manual_ru_val = None
        if method.startswith("Circle"):
            try:
                ru_result = eis.fit_ru_circle(zr, zi, start=start, stop=stop)
            except Exception as exc:
                st.error(f"Circle fit failed: {exc}")
            if ru_result is not None and ru_result.arc_coverage_deg is not None:
                st.caption(
                    f"↳ Fitted arc covers "
                    f"{ru_result.arc_coverage_deg:.0f}° of the circle. Ru is "
                    "the *extrapolated* crossing of that circle with the real "
                    "axis, not a measured point."
                )
                if ru_result.is_extrapolated:
                    st.warning(
                        f"Only {ru_result.arc_coverage_deg:.0f}° of arc is "
                        "being fitted. Below about a quarter turn the "
                        "extrapolated Ru is sensitive to small curvature "
                        "errors — widen the fit range, or measure to a "
                        "higher frequency, before quoting it."
                    )
        else:
            quick = eis.fit_ru_circle(zr, zi, start=start, stop=stop)
            manual_ru_val = st.number_input(
                f"Ru ({disp_unit})", value=float(round(quick.ru, 3)),
                step=0.1, format="%.3f"
            )

    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=zr,
                y=np.abs(zi),
                mode="markers",
                name="EIS data",
                marker={"size": 7, "color": "#1f77b4"},
            )
        )
        # Highlight the points selected for the fit.
        fig.add_trace(
            go.Scatter(
                x=zr[start:stop],
                y=np.abs(zi)[start:stop],
                mode="markers",
                name="Fitted arc",
                marker={"size": 10, "color": "#ff7f0e",
                        "symbol": "circle-open", "line": {"width": 2}},
            )
        )
        if ru_result is not None and ru_result.center is not None:
            cx, cy = eis.circle_path(ru_result.center, ru_result.radius)
            fig.add_trace(
                go.Scatter(x=cx, y=cy, mode="lines", name="Fitted circle",
                           line={"color": "#2ca02c", "dash": "dash"})
            )
        ru_for_marker = ru_result.ru if ru_result else manual_ru_val
        if ru_for_marker is not None:
            fig.add_trace(
                go.Scatter(
                    x=[ru_for_marker], y=[0], mode="markers",
                    name="Ru", marker={"size": 13, "color": "red", "symbol": "x"},
                )
            )
            # A real annotation (not scatter text) so the edit config's
            # annotationPosition lets the user drag it off the data if it
            # starts out overlapping nearby points.
            fig.add_annotation(
                x=ru_for_marker, y=0, text=f"Ru={ru_for_marker:.2f} {disp_unit}",
                showarrow=False, yshift=-18,
                font={"color": "red"},
            )
        _journal_axes_style(fig, f"Z′ / {disp_unit}", f"−Z″ / {disp_unit}",
                            font_size, style=style)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal aspect -> true circle
        st.plotly_chart(fig, width="stretch", config=_PLOTLY_EDIT_CONFIG)
        nyquist_data = _padded_frame({
            f"Z′ ({disp_unit})": zr,
            f"−Z″ ({disp_unit})": np.abs(zi),
        })
        figure_downloads(
            fig, f"nyquist_sample{sel + 1}", key="png_eis",
            what="Nyquist plot", data=nyquist_data,
        )

    # Metrics row. When the impedance was area-scaled, also surface the
    # original (raw, un-normalised) Ru in the file's unit for reference.
    show_raw = scale != 1.0
    chosen_ru = None
    if ru_result is not None:
        cols = st.columns(5 if show_raw else 4)
        i = 0
        if show_raw:
            cols[i].metric(f"Ru ({ru_unit}, original)",
                           f"{ru_result.ru / scale:.3f}")
            i += 1
        cols[i].metric(f"Ru ({disp_unit})", f"{ru_result.ru:.3f}")
        i += 1
        cols[i].metric(f"Rct ({disp_unit})",
                       f"{ru_result.rct:.3f}" if ru_result.rct else "—")
        i += 1
        cols[i].metric(f"Ru + Rct ({disp_unit})",
                       f"{ru_result.r_low:.3f}" if ru_result.r_low else "—")
        i += 1
        cols[i].metric(f"Fit RMSE ({disp_unit})",
                       f"{ru_result.rmse:.3f}" if ru_result.rmse else "—")
        chosen_ru = ru_result.ru / scale  # raw, in the file's unit
    elif manual_ru_val is not None:
        if show_raw:
            c1, c2 = st.columns(2)
            c1.metric(f"Ru ({ru_unit}, original)",
                      f"{manual_ru_val / scale:.3f}")
            c2.metric(f"Ru ({disp_unit})", f"{manual_ru_val:.3f}")
        else:
            st.metric(f"Ru ({disp_unit})", f"{manual_ru_val:.3f}")
        chosen_ru = float(manual_ru_val) / scale  # raw, in the file's unit

    # Batch view: circle-fit Ru for every loaded EIS sample.
    if len(eis_list) > 1:
        with st.expander(f"Ru for all {len(eis_list)} samples (batch fit)"):
            rows = []
            for i, d in enumerate(eis_list):
                try:
                    rr = eis.fit_ru_circle(d.z_real * scale, d.z_imag * scale)
                    row = {"Sample": f"Sample {i + 1}", "Columns": d.label}
                    if show_raw:  # original Ru in the file's unit
                        row[f"Ru ({ru_unit}, original)"] = round(rr.ru / scale, 3)
                    row[f"Ru ({disp_unit})"] = round(rr.ru, 3)
                    row[f"Rct ({disp_unit})"] = round(rr.rct, 3) if rr.rct else None
                    row[f"RMSE ({disp_unit})"] = round(rr.rmse, 3) if rr.rmse else None
                    rows.append(row)
                except Exception as exc:
                    row = {"Sample": f"Sample {i + 1}", "Columns": d.label}
                    if show_raw:
                        row[f"Ru ({ru_unit}, original)"] = None
                    row[f"Ru ({disp_unit})"] = None
                    row[f"Rct ({disp_unit})"] = None
                    row[f"RMSE ({disp_unit})"] = f"fit failed: {exc}"
                    rows.append(row)
            batch_df = pd.DataFrame(rows)
            st.dataframe(batch_df, width="stretch", hide_index=True)
            st.download_button(
                "⬇️ Download batch Ru table (CSV)",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="eis_ru_batch.csv", mime="text/csv",
                key="dl_eis_batch",
            )
            _journal_table_figure(
                batch_df, font_size, "eis_ru_batch_table", key="png_eis_table",
                style=style,
            )

    return chosen_ru


# --------------------------------------------------------------------------- #
# LSV iR-correction tab                                                        #
# --------------------------------------------------------------------------- #
_FACTOR_CHOICES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85, 90, 95, 100]
_PALETTE = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
            "#e377c2", "#17becf", "#bcbd22"]


def _ascii_unit(unit: str) -> str:
    """Header-safe ASCII rendering of a unit (µ→u, ²→2, /→per, drop ·)."""
    return (unit.replace("µ", "u").replace("²", "2")
                .replace("·", "").replace("/", "per").replace("Ω", "ohm"))


def _build_export_csv(lsv_d, results, ru, current_unit, ru_unit="Ω") -> str:
    """Assemble an Origin-friendly CSV of Potential-vs-Current pairs.

    Layout (no comment lines, so OriginLab / Excel import it directly):

        Potential_raw_V, Current_<unit>,
        Ecorr_<f>pct_Ru<ru><ru_unit>_V, Current_<unit>,  (one pair per factor)

    Each curve is a consecutive (X = Potential, Y = Current) pair — set the
    potential column as X in Origin and plot the adjacent current as Y. Ru
    (with its unit) and the compensation % are encoded in the corrected-
    potential column names, so the provenance travels with the data. Raw pair
    comes first, then factors in ascending order; the file can also be
    re-loaded by this app.
    """
    cur_header = f"Current_{_ascii_unit(current_unit)}"
    ru_tag = _ascii_unit(ru_unit)
    headers = ["Potential_raw_V", cur_header]
    columns = [lsv_d.potential, lsv_d.current]
    for r in results:
        p = int(r.factor_percent)
        headers += [f"Ecorr_{p}pct_Ru{ru:.2f}{ru_tag}_V", cur_header]
        columns += [r.potential_corrected, lsv_d.current]
    frame = pd.DataFrame(np.column_stack(columns))
    frame.columns = headers  # allows the repeated Current header
    return frame.to_csv(index=False)


def render_lsv_tab(lsv_d, ru: float | None, current_unit: str = "mA",
                   ru_unit: str = "Ω", area_cm2: float | None = None,
                   sample_label: str = "Sample 1"):
    st.subheader("LSV — ohmic-drop (iR) correction")
    if ru is None:
        st.warning("Determine Ru on the **EIS / Ru Analysis** tab first.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.metric(f"Using Ru ({ru_unit})", f"{ru:.3f}")
        if correction.needs_area(current_unit, ru_unit):
            ru_eff = correction.reconcile_ru(ru, ru_unit, current_unit, area_cm2)
            eff_unit = ("Ω·cm²" if correction.is_density_unit(current_unit)
                        else "Ω")
            st.caption(
                f"Current is **{current_unit}** — using area {area_cm2:g} cm² "
                f"→ effective Ru = **{ru_eff:.3f} {eff_unit}**. "
                "(Set units in the sidebar · Units & electrode area.)"
            )
        else:
            st.caption(
                f"Current is **{current_unit}**, Ru in **{ru_unit}** — units "
                "consistent. (Set units in the sidebar · Units & electrode area.)"
            )
        factors = st.multiselect(
            "Compensation factors (%) — compare several",
            options=_FACTOR_CHOICES,
            default=[85],
            help="Each selected factor is corrected, plotted, and exported "
                 "(range 5–100 %; 85 % is the recommended safe default).",
        )
        custom = st.number_input(
            "…or add a custom factor (%)",
            min_value=correction.MIN_FACTOR_PERCENT,
            max_value=correction.MAX_FACTOR_PERCENT,
            value=85, step=1,
        )
        if st.checkbox("Include custom factor", value=False):
            factors = sorted(set(factors) | {int(custom)})
        if not factors:
            st.info("Select at least one compensation factor.")
            return
        style = plot_style_controls("lsvir", show_markers=False)
        font_size = style["font_size"]
        st.caption(
            "E_corrected = E_measured − (factor) · I · Ru. 100 % is full "
            "correction; partial compensation (≤ 85 %) guards against "
            "over-correction/oscillation when Ru is uncertain."
        )

    factors = sorted(set(factors))
    try:
        results = [
            correction.apply_ir_correction(
                lsv_d.potential, lsv_d.current, ru,
                factor_percent=f, current_unit=current_unit,
                ru_unit=ru_unit, area_cm2=area_cm2,
            )
            for f in factors
        ]
    except ValueError as exc:  # e.g. missing electrode area
        st.error(f"Cannot apply correction: {exc}")
        return

    # Data extents across raw + all corrected curves (defaults for axis range).
    all_pot = np.concatenate(
        [lsv_d.potential] + [r.potential_corrected for r in results]
    )
    x_lo, x_hi = float(np.min(all_pot)), float(np.max(all_pot))
    y_lo, y_hi = float(np.min(lsv_d.current)), float(np.max(lsv_d.current))

    with right:
        view = st.radio(
            "Comparison view",
            ["Overlay (same axes)", "Side-by-side"],
            horizontal=True,
            help="Both show the LSV with vs without iR compensation.",
        )
        # Pad the full data extent (raw + every corrected curve) so traces
        # never touch the plot border — Plotly's own autorange sometimes
        # snaps the window tight to the data, clipping the last point.
        x_pad = (x_hi - x_lo) * 0.04 or 0.01
        y_pad = (y_hi - y_lo) * 0.08 or 0.01
        x_range = [x_lo - x_pad, x_hi + x_pad]
        y_range = [y_lo - y_pad, y_hi + y_pad]
        with st.expander("🔧 Axis range (applies to plot & TIFF export)"):
            if st.checkbox("Set axis limits manually"):
                cx1, cx2 = st.columns(2)
                xmin = cx1.number_input("X min (V)", value=round(x_lo, 3),
                                        step=0.05, format="%.3f")
                xmax = cx2.number_input("X max (V)", value=round(x_hi, 3),
                                        step=0.05, format="%.3f")
                cy1, cy2 = st.columns(2)
                ymin = cy1.number_input(f"Y min ({current_unit})",
                                        value=round(y_lo, 3), step=0.5,
                                        format="%.3f")
                ymax = cy2.number_input(f"Y max ({current_unit})",
                                        value=round(y_hi, 3), step=0.5,
                                        format="%.3f")
                if xmax > xmin:
                    x_range = [xmin, xmax]
                if ymax > ymin:
                    y_range = [ymin, ymax]
                else:
                    st.caption("Max must exceed min; using auto range.")
        raw_hover = "Raw · %{x:.3f} V, %{y:.3f} " + current_unit + "<extra></extra>"
        if view == "Overlay (same axes)":
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=lsv_d.potential, y=lsv_d.current, mode="lines",
                           name="Without iR comp (raw)",
                           line={"color": "#1f77b4", "width": 2.5},
                           hovertemplate=raw_hover)
            )
            for i, r in enumerate(results):
                pct = int(r.factor_percent)
                fig.add_trace(
                    go.Scatter(
                        x=r.potential_corrected, y=lsv_d.current, mode="lines",
                        name=f"With iR comp {pct}%",
                        line={"color": _PALETTE[i % len(_PALETTE)], "width": 2.5},
                        hovertemplate=(f"iR {pct}% · " + "%{x:.3f} V, %{y:.3f} "
                                       + current_unit + "<extra></extra>"),
                    )
                )
            fig.update_layout(
                title=f"LSV — {sample_label} — with vs without iR compensation",
                xaxis_title="Potential / V",
                yaxis_title=f"Current / {current_unit}",
            )
        else:
            fig = make_subplots(
                rows=1, cols=2, shared_yaxes=True,
                subplot_titles=("Without iR compensation",
                                "With iR compensation"),
                horizontal_spacing=0.04,
            )
            fig.add_trace(
                go.Scatter(x=lsv_d.potential, y=lsv_d.current, mode="lines",
                           name="Without iR comp (raw)",
                           line={"color": "#1f77b4", "width": 2.5},
                           hovertemplate=raw_hover),
                row=1, col=1,
            )
            for i, r in enumerate(results):
                pct = int(r.factor_percent)
                fig.add_trace(
                    go.Scatter(
                        x=r.potential_corrected, y=lsv_d.current, mode="lines",
                        name=f"With iR comp {pct}%",
                        line={"color": _PALETTE[i % len(_PALETTE)], "width": 2.5},
                        hovertemplate=(f"iR {pct}% · " + "%{x:.3f} V, %{y:.3f} "
                                       + current_unit + "<extra></extra>"),
                    ),
                    row=1, col=2,
                )
            fig.update_xaxes(title_text="Potential / V", row=1, col=1)
            fig.update_xaxes(title_text="Potential / V", row=1, col=2)
            fig.update_yaxes(
                title_text=f"Current / {current_unit}", row=1, col=1
            )
            fig.update_layout(
                title=f"LSV — {sample_label} — with vs without iR compensation"
            )

        # Padded default range (or the user's manual override) — applies to
        # every x/y axis in the figure, including both side-by-side panels.
        fig.update_xaxes(range=x_range)
        fig.update_yaxes(range=y_range)

        # Legend at the bottom so it never overlaps the title / subplot titles.
        # Journal style (Arial, box-border axes) applied globally, which
        # works for both the single-axes overlay and the two-panel
        # side-by-side view (update_xaxes/update_yaxes with no row/col
        # target every axis in the figure).
        fig.update_layout(
            template="plotly_white",
            height=470,
            font={"family": "Arial", "size": font_size},
            title={"y": 0.97, "yanchor": "top"},
            legend={"orientation": "h", "yanchor": "top", "y": -0.18,
                    "xanchor": "center", "x": 0.5,
                    "font": {"family": "Arial",
                             "size": max(11, round(font_size * 0.55))},
                    "bgcolor": "rgba(255,255,255,0.7)"},
            margin=_journal_margin(font_size, legend_below=True),
            hovermode="x unified",
            hoverlabel={"font": {"family": "Arial", "size": 12}},
        )
        fig.update_xaxes(showspikes=True, spikemode="across",
                         spikethickness=1, spikedash="dot",
                         spikecolor="#888", **_BOX_AXIS_STYLE)
        fig.update_yaxes(**_BOX_AXIS_STYLE)
        st.plotly_chart(fig, width="stretch", config=_PLOTLY_EDIT_CONFIG)
        fac_png = "-".join(str(int(r.factor_percent)) for r in results)
        slug = re.sub(r"\W+", "_", sample_label).strip("_").lower()
        plotted = {
            "Potential raw (V)": lsv_d.potential,
            f"Current raw ({current_unit})": lsv_d.current,
        }
        for r in results:
            pct = int(r.factor_percent)
            plotted[f"Potential iR-corrected {pct}% (V)"] = r.potential_corrected
            plotted[f"Current iR-corrected {pct}% ({current_unit})"] = lsv_d.current
        figure_downloads(
            fig, f"lsv_iR_comparison_{slug}_f{fac_png}pct", key="png_lsv",
            what="Comparison plot", data=_padded_frame(plotted),
        )

    # Over-compensation assessment per factor (fold-back detection).
    assessments = [
        correction.assess_correction(lsv_d.potential, r.potential_corrected)
        for r in results
    ]

    def _status(a):
        if a.over_compensated:
            return "⚠ over-corrected (folds back)"
        if a.reverted_fraction > a.raw_reverted_fraction + 0.005:
            return "borderline"
        return "✓ good"

    # Per-factor summary table. Show the raw Ru in its own unit, and — when an
    # area reconciliation applies — the area-normalised "Ru effective".
    n = len(results)
    ru_col = f"Ru ({ru_unit})"
    irdrop_col = "Max |iR drop| (mV)"
    cols = {
        "Compensation %": [int(r.factor_percent) for r in results],
        ru_col: [round(ru, 4)] * n,
    }
    # Columns that should always render with fixed decimals (so values that
    # happen to be whole numbers, e.g. 220.0, still show as 220.000).
    numeric_fmt = {ru_col: "%.4f", irdrop_col: "%.3f", "Fold-back %": "%.1f"}
    if correction.needs_area(current_unit, ru_unit):
        ru_eff = results[0].ru_effective
        eff_unit = ("Ω·cm²" if correction.is_density_unit(current_unit) else "Ω")
        eff_col = f"Ru effective ({eff_unit})"
        cols[eff_col] = [round(ru_eff, 4)] * n
        numeric_fmt[eff_col] = "%.4f"
    cols[irdrop_col] = [
        round(float(np.max(np.abs(r.ir_drop))) * 1000, 4) for r in results
    ]
    cols["Fold-back %"] = [round(a.reverted_fraction * 100, 3) for a in assessments]
    cols["Status"] = [_status(a) for a in assessments]
    summary = pd.DataFrame(cols)
    st.markdown("**Results summary**")
    st.dataframe(
        summary, width="stretch", hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format=fmt)
            for c, fmt in numeric_fmt.items()
        },
    )
    st.download_button(
        "⬇️ Download results table (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"ir_results_summary_{slug}_Ru{ru:.1f}.csv",
        mime="text/csv",
        key="dl_summary",
    )
    display = summary.copy()
    for c, fmt in numeric_fmt.items():
        display[c] = [fmt % v for v in summary[c]]
    _journal_table_figure(  # CSV of this table is the results button above
        display, font_size, f"ir_results_table_{slug}_Ru{ru:.1f}",
        key="png_lsv_table", style=style,
    )

    # Warn if any selected factor over-compensates, and recommend a safe one.
    if any(a.over_compensated for a in assessments):
        bad = [int(r.factor_percent)
               for r, a in zip(results, assessments) if a.over_compensated]
        rec = correction.recommend_factor(
            lsv_d.potential, lsv_d.current, ru, current_unit,
            ru_unit=ru_unit, area_cm2=area_cm2,
        )
        st.warning(
            f"⚠ Over-compensation at {', '.join(f'{b}%' for b in bad)}: the "
            "iR-corrected LSV folds back on itself (the potential reverses "
            "instead of advancing), which is unphysical. Highest safe factor "
            f"for this sample ≈ **{rec}%**."
        )

    with st.expander("ℹ️ How to interpret — what is a *good* compensation?"):
        st.markdown(
            "- **Goal.** iR compensation removes the ohmic drop `I·Ru` so the "
            "plotted potential reflects the true electrode potential. A good "
            "correction makes features (onset, peaks, Tafel region) **sharper "
            "and shifts the curve to lower overpotential** without distorting "
            "its shape.\n"
            "- **Good (✓).** The corrected sweep stays **monotonic** — it keeps "
            "advancing in the sweep direction. Fold-back ≈ 0 %.\n"
            "- **Over-corrected (⚠).** The curve **folds back**: at high "
            "current `f·I·Ru` exceeds the real potential step, so the corrected "
            "potential reverses and revisits earlier values. This is the "
            "classic sign of too much compensation (and, in feedback hardware, "
            "of oscillation).\n"
            "- **Rule of thumb.** Increase the factor until just before "
            "fold-back appears. **100 %** is full correction; **85 %** is the "
            "recommended safe default. If fold-back shows up well below 100 %, "
            "your `Ru` may be over-estimated — re-check the EIS arc fit."
        )

    csv_text = _build_export_csv(lsv_d, results, ru, current_unit, ru_unit)
    st.caption(
        "Export layout (Origin-friendly): consecutive Potential–Current pairs — "
        "raw first, then one **corrected potential vs current** pair per factor. "
        "Current is unchanged by iR compensation (only potential shifts)."
    )
    with st.expander("Preview export (first lines)"):
        st.code("\n".join(csv_text.splitlines()[:14]), language="text")
    fac_tag = "-".join(str(int(r.factor_percent)) for r in results)
    st.download_button(
        "⬇️ Download corrected LSV (CSV — includes Ru & compensation %)",
        data=csv_text.encode("utf-8"),
        file_name=f"lsv_iR_corrected_Ru{ru:.1f}_f{fac_tag}pct.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Tafel slope analysis tab (independent data source)                          #
# --------------------------------------------------------------------------- #
# Reactions a sample can be tagged with. Those in tafel.REACTION_REFERENCES
# also get a mechanistic benchmark in the results table; the rest are plain
# labels (the benchmark column then reads "—"), which is deliberate — there
# is no comparably canonical slope table for them.
_TAFEL_REACTIONS = [
    "HER", "HOR", "OER", "ORR",
    "CO₂RR", "N₂RR", "NO₃RR", "CORR",
    "MOR", "EOR", "UOR",
    "Other / unspecified",
]
# Publication-style export settings shared by every tab: a selectable Arial
# font size and a closed box border (mirrored axis lines) with no interior
# gridlines.
# Font families that are safe for journal submission: each is either a
# standard PostScript/PDF base font or metric-compatible with one, so the
# figure renders identically on a typesetter that lacks the exact face.
_JOURNAL_FONTS = [
    "Arial", "Helvetica", "Times New Roman", "Calibri", "Georgia",
    "DejaVu Sans", "Courier New",
]

# Named colour sets. "Colourblind-safe" is Okabe-Ito, which stays
# distinguishable under all three common forms of colour-vision deficiency and
# is the set most journals recommend when they recommend one at all.
# "Grayscale" exists because some journals still charge for colour figures.
_PALETTES: dict[str, list[str]] = {
    "Default (Plotly)": ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
                         "#8c564b", "#e377c2", "#17becf", "#bcbd22"],
    "Colourblind-safe (Okabe-Ito)": ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
                                     "#E69F00", "#56B4E9", "#F0E442", "#000000"],
    "Viridis": ["#440154", "#414487", "#2a788e", "#22a884",
                "#7ad151", "#fde725", "#31688e", "#35b779"],
    "Grayscale (print)": ["#000000", "#4d4d4d", "#7f7f7f", "#a6a6a6",
                          "#1a1a1a", "#666666", "#999999", "#cccccc"],
    "Nature-style": ["#E64B35", "#4DBBD5", "#00A087", "#3C5488",
                     "#F39B7F", "#8491B4", "#91D1C2", "#DC0000"],
}

_LINE_DASHES = ["solid", "dot", "dash", "longdash", "dashdot"]
_MARKER_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross",
                   "x", "star", "hexagon"]
_LEGEND_POSITIONS = ["top-left", "top-right", "bottom-left", "bottom-right",
                     "outside-right", "below", "hidden"]


def plot_style_controls(key_prefix: str, *, default_palette: str = "Default (Plotly)",
                        show_markers: bool = True) -> dict:
    """Render the shared "Plot appearance" panel and return a style dict.

    Every tab's figures read their fonts, colours, line/marker geometry, grid
    and legend placement from this one dict, so a change here applies to what
    is on screen *and* to whatever is exported from it -- the export is a
    render of the same figure object, not a separately styled copy.

    Returned keys are consumed by :func:`apply_plot_style` and by the
    per-trace colour pickers each tab renders from ``palette``.
    """
    style: dict = {}
    with st.expander("🎨 Plot appearance — fonts, colours, lines, legend"):
        c1, c2, c3, c4 = st.columns(4)
        style["font_family"] = c1.selectbox(
            "Font", _JOURNAL_FONTS, index=0, key=f"{key_prefix}_font_family",
            help="Kept to faces that are safe for journal submission.",
        )
        style["font_size"] = c2.number_input(
            "Axis font size (pt)", min_value=8, max_value=48, value=28, step=1,
            key=f"{key_prefix}_font_size",
            help="Applies to axis titles and tick labels. Legend and "
                 "annotations scale from this.",
        )
        style["palette_name"] = c3.selectbox(
            "Colour palette", list(_PALETTES),
            index=list(_PALETTES).index(default_palette),
            key=f"{key_prefix}_palette",
            help="Sets the default colour of each series; individual series "
                 "can still be overridden below.",
        )
        style["legend_position"] = c4.selectbox(
            "Legend", _LEGEND_POSITIONS, index=0,
            key=f"{key_prefix}_legend_pos",
        )

        d1, d2, d3, d4 = st.columns(4)
        style["line_width"] = d1.slider(
            "Line width", 0.5, 6.0, 2.5, 0.5, key=f"{key_prefix}_line_width",
        )
        style["marker_size"] = d2.slider(
            "Marker size", 0, 20, 8 if show_markers else 0, 1,
            key=f"{key_prefix}_marker_size",
            help="0 hides markers and draws lines only.",
        )
        style["marker_symbol"] = d3.selectbox(
            "Marker symbol", _MARKER_SYMBOLS, index=0,
            key=f"{key_prefix}_marker_symbol",
        )
        style["fit_dash"] = d4.selectbox(
            "Fit-line style", _LINE_DASHES, index=1, key=f"{key_prefix}_fit_dash",
            help="Dash pattern for fitted//derived lines, so they read as "
                 "distinct from the measured data.",
        )

        e1, e2, e3, e4 = st.columns(4)
        style["show_grid"] = e1.checkbox(
            "Gridlines", value=False, key=f"{key_prefix}_grid",
            help="Off by default — most journals prefer a clean plot frame.",
        )
        style["mirror_axes"] = e2.checkbox(
            "Box frame", value=True, key=f"{key_prefix}_mirror",
            help="Closed box (all four axis lines), the usual journal style.",
        )
        style["n_ticks"] = e3.slider(
            "Approx. tick count", 3, 12, 5, 1, key=f"{key_prefix}_nticks",
        )
        style["show_title"] = e4.checkbox(
            "Show plot title", value=True, key=f"{key_prefix}_show_title",
            help="Off for figures destined for a caption instead.",
        )

    style["palette"] = _PALETTES[style["palette_name"]]
    return style


def _default_style() -> dict:
    """Style dict matching the app's previous hard-coded look, for any code
    path that has not been given one."""
    return {
        "font_family": "Arial", "font_size": 28,
        "palette_name": "Default (Plotly)", "palette": _PALETTES["Default (Plotly)"],
        "legend_position": "top-left", "line_width": 2.5, "marker_size": 8,
        "marker_symbol": "circle", "fit_dash": "dot", "show_grid": False,
        "mirror_axes": True, "n_ticks": 5, "show_title": True,
    }


def _axis_style(style: dict) -> dict:
    """Axis keyword arguments implied by ``style`` (replaces the old fixed
    ``_BOX_AXIS_STYLE`` so the frame, grid and tick density follow the user's
    choices)."""
    return {
        "showgrid": bool(style.get("show_grid")),
        "gridcolor": "#d9d9d9",
        "zeroline": False,
        "showline": True, "linewidth": 2, "linecolor": "black",
        "mirror": bool(style.get("mirror_axes", True)),
        # Plotly's default ticks are thin and grey, which all but vanish once
        # a figure is downsampled into a journal column. Match the axis line.
        "ticks": "outside", "tickwidth": 2, "ticklen": 7, "tickcolor": "black",
        "nticks": int(style.get("n_ticks", 5)),
    }


_LEGEND_ANCHORS = {
    "top-left": {"x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"},
    "top-right": {"x": 0.98, "y": 0.98, "xanchor": "right", "yanchor": "top"},
    "bottom-left": {"x": 0.02, "y": 0.02, "xanchor": "left", "yanchor": "bottom"},
    "bottom-right": {"x": 0.98, "y": 0.02, "xanchor": "right", "yanchor": "bottom"},
    "outside-right": {"x": 1.02, "y": 1.0, "xanchor": "left", "yanchor": "top"},
    "below": {"x": 0.5, "y": -0.18, "xanchor": "center", "yanchor": "top",
              "orientation": "h"},
}


def apply_plot_style(fig, style: dict, xtitle: str, ytitle: str,
                     height: int = 460, yrange: list | None = None,
                     title: str | None = None) -> None:
    """Apply a :func:`plot_style_controls` dict to ``fig`` in place.

    The single place figure styling happens, so on-screen and exported
    figures cannot drift apart.
    """
    style = {**_default_style(), **(style or {})}
    family, size = style["font_family"], int(style["font_size"])
    axis_font = {"family": family, "size": size}
    small_font = {"family": family, "size": max(9, round(size * 0.55))}
    position = style.get("legend_position", "top-left")

    layout = {
        "template": "plotly_white",
        "height": height,
        "font": axis_font,
        # 10 px is not enough room for a tick label, let alone an axis title:
        # every exported figure came out with its y-axis numbers and both axis
        # titles clipped at the edge. One derivation for every figure, so the
        # styled tabs and the hand-built ones cannot drift (see
        # :func:`_journal_margin`).
        "margin": {**_journal_margin(
            size, legend_below=(position == "below"),
            titled=bool(title and style.get("show_title"))),
            **({"r": 170} if position == "outside-right" else {})},
    }
    if position == "hidden":
        layout["showlegend"] = False
    else:
        layout["legend"] = dict(**_LEGEND_ANCHORS[position], font=small_font,
                                bgcolor="rgba(255,255,255,0.7)",
                                tracegroupgap=18)
    if title and style.get("show_title"):
        layout["title"] = {"text": title, "font": axis_font}
    fig.update_layout(**layout)

    axis_kwargs = _axis_style(style)
    # Axis titles read one step up from the tick labels in most journal house
    # styles; using one size for both made the titles disappear into the numbers.
    title_font = {"family": family, "size": size + 8}
    fig.update_xaxes(title={"text": xtitle, "font": title_font},
                     tickfont=axis_font, **axis_kwargs)
    fig.update_yaxes(title={"text": ytitle, "font": title_font},
                     tickfont=axis_font, range=yrange, **axis_kwargs)


def trace_color_pickers(labels: list[str], style: dict, key_prefix: str) -> dict:
    """Per-series colour overrides, seeded from the chosen palette.

    Returns ``{label: "#rrggbb"}``. Rendered as a row of colour pickers so a
    user preparing a figure for a specific journal (or matching an existing
    figure in the same paper) can set each series exactly, rather than being
    limited to whichever palette happens to be selected.
    """
    palette = style.get("palette") or _PALETTES["Default (Plotly)"]
    colors: dict[str, str] = {}
    if not labels:
        return colors
    with st.expander(f"🎨 Series colours ({len(labels)})"):
        cols = st.columns(min(4, len(labels)))
        for i, label in enumerate(labels):
            default = palette[i % len(palette)]
            # Keyed on the palette name too, so switching palette re-seeds the
            # pickers instead of leaving them stuck on the previous set.
            widget_key = f"{key_prefix}_color_{i}_{style.get('palette_name', '')}"
            colors[label] = cols[i % len(cols)].color_picker(
                label if len(label) <= 18 else label[:17] + "…",
                value=default, key=widget_key,
            )
    return colors


_JOURNAL_FONT_SIZES = [28, 36]


def _journal_margin(font_size: int, *, legend_below: bool = False,
                    titled: bool = True) -> dict:
    """Margins wide enough that tick labels and axis titles survive export.

    Plotly's 10 px default (and the hard-coded 10 px dicts these figures used
    to carry) clips the y-axis numbers and both axis titles the moment the
    figure is rendered to TIFF/PNG at the size a journal wants.

    The sizes are derived from the text each margin has to hold rather than
    copied from the reference app's fixed 110/90 -- those were sized for its
    ~930 px canvas, and a fixed 200 px of horizontal margin swallows 62 % of
    an 8.6 cm single-column figure (325 CSS px), however small the font. Left
    holds a rotated axis title plus a tick label; bottom holds a tick label
    plus the axis title, which renders at ``font_size + 8`` (see
    :func:`apply_plot_style`); right and top only need to keep the last tick
    label and any title off the edge.
    """
    title_size = font_size + 8
    tick_len = 7
    return {
        # rotated title + ~4 characters of tick label + tick + padding
        "l": int(max(45, title_size * 1.2 + font_size * 2.4 + tick_len + 10)),
        # half of the last x tick label, or room for an outside legend
        "r": int(max(25, font_size * 1.5 + 10)),
        "t": int(max(25, title_size * 1.8)) if titled else int(max(15, font_size)),
        # x tick label + axis title + tick + padding (+ a legend strip below)
        "b": int(max(40, font_size * 1.2 + title_size * 1.3 + tick_len + 10)
                 + (font_size * 2.2 if legend_below else 0)),
    }


_BOX_AXIS_STYLE = {
    "showgrid": False, "zeroline": False,
    "showline": True, "linewidth": 2, "linecolor": "black", "mirror": True,
    "ticks": "outside", "tickwidth": 2, "ticklen": 7, "tickcolor": "black",
    "nticks": 5,
}
# Every journal-style plot uses this so its legend and any text annotations
# (slope labels, etc.) can be dragged to a better spot before export.
_PLOTLY_EDIT_CONFIG = {
    "edits": {"legendPosition": True, "annotationPosition": True},
    "displaylogo": False,
}


def _range_controls(
    key_prefix: str, x_label: str, y_label: str | None,
    x_lo: float, x_hi: float, y_lo: float | None = None, y_hi: float | None = None,
    x_step: float = 0.05, y_step: float = 0.5,
) -> tuple[list | None, list | None]:
    """Render an optional axis-range expander shared by every LSV/ORR plot.

    A pure *view* control: it never filters the underlying data or traces,
    only sets the plotted axis range via ``fig.update_xaxes``/
    ``update_yaxes(range=...)`` — so the full dataset is always in the
    figure (and its CSV/full-precision export), and widening the range (or
    unchecking the box) always recovers everything, including outside the
    default scientifically-typical window shown here. Pass ``y_label=None``
    to omit the Y-axis controls (e.g. a plot whose Y-axis is already fixed,
    like %H2O2/n). Returns ``(x_range, y_range)``, each ``None`` unless the
    user opts in (or the axis is omitted)."""
    x_range: list | None = None
    y_range: list | None = None
    with st.expander(f"🔧 {x_label}" + (f" / {y_label}" if y_label else "")
                     + " range (view only — data is never cut)"):
        if st.checkbox("Set axis limits manually", key=f"{key_prefix}_range_toggle"):
            cx1, cx2 = st.columns(2)
            xmin = cx1.number_input(f"{x_label} min", value=round(x_lo, 3),
                                    step=x_step, format="%.3f", key=f"{key_prefix}_xmin")
            xmax = cx2.number_input(f"{x_label} max", value=round(x_hi, 3),
                                    step=x_step, format="%.3f", key=f"{key_prefix}_xmax")
            if xmax > xmin:
                x_range = [xmin, xmax]
            else:
                st.caption(f"{x_label}: max must exceed min; using auto range.")
            if y_label is not None:
                cy1, cy2 = st.columns(2)
                ymin = cy1.number_input(f"{y_label} min", value=round(y_lo, 3),
                                        step=y_step, format="%.3f", key=f"{key_prefix}_ymin")
                ymax = cy2.number_input(f"{y_label} max", value=round(y_hi, 3),
                                        step=y_step, format="%.3f", key=f"{key_prefix}_ymax")
                if ymax > ymin:
                    y_range = [ymin, ymax]
                else:
                    st.caption(f"{y_label}: max must exceed min; using auto range.")
    return x_range, y_range


def _journal_axes_style(fig, xtitle: str, ytitle: str, font_size: int,
                        height: int = 460, yrange: list | None = None,
                        legend_position: str = "top-left",
                        style: dict | None = None) -> None:
    """Apply the shared journal look to ``fig`` in place.

    Thin wrapper over :func:`apply_plot_style`: callers that already hold a
    user style dict pass it through, and callers that only know a font size
    (the historical signature) get the same result as before. Keeping one
    implementation is what stops the on-screen figure and the exported one
    from diverging.
    """
    merged = {**_default_style(), **(style or {})}
    merged["font_size"] = font_size
    if style is None or "legend_position" not in style:
        merged["legend_position"] = legend_position
    apply_plot_style(fig, merged, xtitle, ytitle, height=height, yrange=yrange)


def _journal_table_figure(display_df: pd.DataFrame, font_size: int, stem: str,
                          key: str, what: str = "Results table",
                          style: dict | None = None) -> None:
    """Render ``display_df`` (already formatted to display-ready strings —
    e.g. numbers pre-rounded/pre-formatted by the caller) as an Arial,
    publication-style Plotly table figure, with TIFF/HTML downloads via
    :func:`figure_downloads`. The export canvas is sized from the table's own
    content: columns are widened in proportion to the longest string they
    hold, and the row height leaves room for the lines Plotly wraps text
    onto — otherwise long entries overlap and the last rows fall outside a
    fixed-size export. Pair this with the caller's own CSV download of the
    underlying (full-precision) data.
    """
    style = {**_default_style(), **(style or {})}
    family = style["font_family"]
    cell_font = max(9.0, font_size * 0.4)
    header_font = max(10.0, font_size * 0.45)
    # Explicit NaN/None check before stringifying: a plain ``.astype(str)``
    # on a nullable numeric column leaves those entries as a raw float NaN
    # instead of the string "nan" in some pandas versions, which then breaks
    # the length-based sizing below.
    str_cols = {
        c: display_df[c].apply(lambda v: "—" if pd.isna(v) else str(v))
        for c in display_df.columns
    }
    widths = []
    for c in display_df.columns:
        longest = str_cols[c].str.len().max()
        longest = int(longest) if pd.notna(longest) else 0
        # +2 characters of breathing room so short entries aren't wrapped, and
        # a cap so one long sentence can't squeeze every other column.
        widths.append(min(max(len(str(c)), longest, 6) + 2, 46))
    char_px = cell_font * 0.66
    # Clamped to the same bounds the export size inputs accept, so a very wide
    # table cannot produce a default the width widget then silently rejects.
    table_width = int(min(max(sum(widths) * char_px + 60, 200), 4000))
    # Worst-case wrapped lines in any cell of a row -> uniform row height.
    max_wrap = max(
        (int(np.ceil(len(v) / w)) for col, w in zip(display_df.columns, widths)
         for v in str_cols[col]),
        default=1,
    )
    row_h = int(cell_font * 1.5 * max(1, max_wrap)) + 8
    # Journal table convention: a ruled header, no vertical rules, and faint
    # zebra striping so a wide row stays readable across the page. Colours are
    # neutral greys rather than the chosen palette -- a results table should
    # not compete with the figures for colour.
    n_rows = len(display_df)
    row_fill = ["#ffffff" if i % 2 == 0 else "#f4f4f4" for i in range(n_rows)]
    # Numbers right-align so their decimal points line up down the column --
    # the whole reason a results table is readable at a glance. Text stays
    # left-aligned. A column counts as numeric when every entry that is not
    # the em-dash placeholder parses as a number.

    def _is_numeric_col(col: str) -> bool:
        values = [v for v in str_cols[col] if v != "—"]
        if not values:
            return False
        for v in values:
            try:
                float(v.replace("±", " ").split()[0])
            except (ValueError, IndexError):
                return False
        return True

    aligns = ["right" if _is_numeric_col(c) else "left"
              for c in display_df.columns]
    table_fig = go.Figure(data=[go.Table(
        columnwidth=widths,
        header={"values": [f"<b>{c}</b>" for c in display_df.columns],
                "font": {"family": family, "size": header_font, "color": "#000000"},
                "fill": {"color": "#e2e2e2"},
                "line": {"color": "#000000", "width": 1},
                "align": aligns, "height": int(header_font * 1.6) + 10},
        cells={"values": [str_cols[c] for c in display_df.columns],
               "font": {"family": family, "size": cell_font}, "align": aligns,
               "fill": {"color": [row_fill] * len(display_df.columns)},
               # No interior rules: the zebra striping already separates rows,
               # and the grid of boxes this used to draw is exactly what the
               # comment above says a journal table does not have.
               "line": {"color": "rgba(0,0,0,0)", "width": 0},
               "height": row_h},
    )])
    table_height = int(header_font * 1.6) + 20 + row_h * len(display_df) + 20
    table_fig.update_layout(
        template="plotly_white",  # never export with Streamlit's placeholder colours
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        width=table_width, height=table_height,
    )
    figure_downloads(
        table_fig, stem, key=key, what=what,
        width=table_width, height=table_height,
        data=display_df,
    )


def _rescale_current(raw: np.ndarray, from_unit: str, to_unit: str) -> np.ndarray:
    """Convert a current (or current-density) array from ``from_unit`` to
    ``to_unit``. Same-family only (both absolute or both density) — a
    cross-family change (needs an electrode area) is left unconverted."""
    if from_unit == to_unit:
        return raw
    from_density = correction.is_density_unit(from_unit)
    to_density = correction.is_density_unit(to_unit)
    table = correction.CURRENT_DENSITY_UNITS if from_density else correction.CURRENT_UNITS
    if from_density != to_density or from_unit not in table or to_unit not in table:
        return raw
    return raw * (table[from_unit] / table[to_unit])


def _darken(hex_color: str, factor: float = 0.6) -> str:
    """Darken a ``#rrggbb`` color so a fit line stands out against its own
    sample's (lighter) data color."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return "#{:02x}{:02x}{:02x}".format(
        *(max(0, int(c * factor)) for c in (r, g, b))
    )


_REPLICATE_SUFFIX = re.compile(r"\s*\(\d+\)$")


def _default_replicate_group(label: str) -> str:
    """Guess a replicate-group name from a sample label by stripping a
    trailing ``" (2)"``/``" (3)``/… — the exact suffix ``_dedup_label`` adds
    when the same base file/column/folder name is loaded more than once,
    which is exactly the common case of uploading several repeat scans of
    one sample."""
    stripped = _REPLICATE_SUFFIX.sub("", label).strip()
    return stripped or label


def _dedup_label(label: str, seen: dict[str, int]) -> str:
    """Disambiguate a repeated label as ``"name"``, ``"name (2)"``, … — used
    by every LSV-tab upload path (individual files and ZIP batch upload) so
    labels collide the same way regardless of source, and so
    ``_default_replicate_group`` can strip the suffix back off to
    auto-group repeat runs of one sample."""
    seen[label] = seen.get(label, 0) + 1
    return label if seen[label] == 1 else f"{label} ({seen[label]})"


def _selection_signature(points: list) -> tuple:
    """Fingerprint a Plotly box-select event so it can be applied exactly
    once (Streamlit keeps the same selection in session_state across
    unrelated reruns; without this, it would keep re-overriding the fit
    sliders after the user manually adjusts them)."""
    return tuple(sorted(
        (p.get("curve_number"), p.get("point_index")) for p in points
    ))


def _selection_range_for_sample(points: list, orig_label: str,
                                log_i: np.ndarray) -> tuple[int, int] | None:
    """Map a Tafel-plot box-select event back to a ``(start, stop)`` index
    range for one sample: keep only the selected points whose ``customdata``
    matches this sample, then find the index span in its own log|i| array
    covering that x-range. Returns ``None`` if the selection doesn't touch
    this sample."""
    xs = []
    for p in points:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if cd == orig_label and p.get("x") is not None:
            xs.append(float(p["x"]))
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    idx = np.flatnonzero((log_i >= lo - 1e-9) & (log_i <= hi + 1e-9))
    if len(idx) < 3:
        return None
    return int(idx.min()), int(idx.max()) + 1


def _tafel_extract_zip_samples(
    upload, seen_labels: dict[str, int]
) -> list[tuple[str, data_io.LSVData]]:
    """Batch-load LSV runs from a ZIP of a data folder — for pulling in many
    repeat runs/samples at once instead of picking files one by one. Each
    file becomes one run; files sharing a top-level subfolder are labelled
    ``"name"``, ``"name (2)"``, … via :func:`_dedup_label` — the same
    suffix a manually re-uploaded duplicate gets — so they land in the same
    **Replicate group** by default and error bars across them fall out of
    the existing replicate-statistics machinery with no extra tagging.
    Files at the zip's own root all become runs of one sample, named after
    the zip."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(upload.getvalue()))
    except zipfile.BadZipFile:
        st.error(f"{upload.name}: not a valid .zip file.")
        return []
    total = sum(info.file_size for info in zf.infolist())
    if total > data_io.MAX_UNCOMPRESSED_BYTES:
        st.error(
            f"{upload.name}: zip expands too large; rejected as a possible "
            "decompression bomb."
        )
        return []

    members = [
        m for m in zf.namelist()
        if m.lower().endswith((".csv", ".txt", ".xlsx", ".xls"))
        and "__MACOSX" not in m and not m.rsplit("/", 1)[-1].startswith(".")
    ]
    if not members:
        st.warning(f"{upload.name}: no CSV/Excel files found inside.")
        return []
    if len(members) > MAX_ZIP_MEMBERS:
        st.error(
            f"{upload.name}: contains {len(members)} data files, more than "
            f"the {MAX_ZIP_MEMBERS}-file limit. Split it into smaller zips."
        )
        return []

    default_sample = upload.name.rsplit(".", 1)[0]
    series: list[tuple[str, data_io.LSVData]] = []
    skipped = 0
    for member in sorted(members):
        parts = member.replace("\\", "/").split("/")
        sample_name = parts[0] if len(parts) > 1 else default_sample
        filename = parts[-1]
        try:
            raw_bytes = zf.read(member)
        except Exception:
            skipped += 1
            continue
        try:
            if filename.lower().endswith((".xlsx", ".xls")):
                sheets = data_io.list_sheets(io.BytesIO(raw_bytes))
                ds = data_io.load_lsv_datasets(io.BytesIO(raw_bytes), sheet=sheets[0])
            else:
                ds = data_io.load_lsv_datasets(io.BytesIO(raw_bytes), sheet=None)
        except Exception:
            skipped += 1
            continue
        for d in ds:
            label = f"{sample_name} · {d.label}" if len(ds) > 1 else sample_name
            series.append((_dedup_label(label, seen_labels), d))

    if skipped:
        st.caption(
            f"↪ {skipped} file(s) inside the zip were skipped (unreadable or "
            "not a Potential/Current pair)."
        )
    return series


def _tafel_data_loader() -> list[tuple[str, data_io.LSVData]]:
    """File uploader local to the Tafel tab — independent of the main sidebar
    EIS/LSV loader. Multiple files (different samples/lots/repeat runs) may
    be uploaded at once, either individually or batched via a ZIP of a data
    folder; each becomes one or more labelled series so several samples (and
    repeat scans of the same sample) can be combined into one journal-style
    overlay. Returns a flat list of ``(label, LSVData)``."""
    st.markdown(
        "**Data source** (independent of the EIS/LSV loader above) — upload "
        "one or more files, one per sample/run; they can be combined into a "
        "single overlay plot below, or batch-loaded from a ZIP to pull in "
        "many runs/samples at once (handy for extracting overpotential "
        "error bars across repeat scans and across samples)."
    )

    series: list[tuple[str, data_io.LSVData]] = []
    seen_labels: dict[str, int] = {}

    with st.expander("📦 Batch upload — a ZIP of a whole data folder"):
        st.caption(
            "Zip your data folder (one subfolder per sample, containing its "
            "repeat-run files — extra nesting inside a sample's subfolder "
            "is fine) and upload it here. Files sharing a subfolder are "
            "auto-tagged '(2)', '(3)', … the same way a manually re-"
            "uploaded duplicate is, so they land in the same **Replicate "
            "group** below by default. Files at the zip's own root all "
            "become runs of one sample, named after the zip."
        )
        zip_up = st.file_uploader("ZIP file", type=["zip"], key="tafel_zip")
        if zip_up is not None:
            zip_series = _tafel_extract_zip_samples(zip_up, seen_labels)
            if zip_series:
                st.success(
                    f"Loaded {len(zip_series)} run(s) from {zip_up.name}: "
                    + ", ".join(lbl for lbl, _ in zip_series)
                )
            series.extend(zip_series)

    source = st.radio(
        "Additionally upload individual files",
        ["Excel workbook(s)", "CSV file(s)", "None"],
        horizontal=True,
        key="tafel_source",
        help="Columns: Potential, Current (or current density). Several "
             "datasets may also sit side-by-side within one file as "
             "repeated column pairs.",
    )

    try:
        if source == "Excel workbook(s)":
            ups = st.file_uploader(
                "Excel (.xlsx)", type=["xlsx", "xls"], key="tafel_xlsx",
                accept_multiple_files=True,
            )
            if ups:
                first_sheets = _cached_sheets(ups[0].getvalue())
                sheet = st.selectbox(
                    "Sheet (applied to every uploaded workbook)",
                    first_sheets, index=0, key="tafel_sheet",
                )
                for up in ups:
                    base = up.name.rsplit(".", 1)[0]
                    try:
                        sheets_here = _cached_sheets(up.getvalue())
                        use_sheet = sheet if sheet in sheets_here else sheets_here[0]
                        ds = _cached_lsv_datasets(up.getvalue(), use_sheet)
                    except Exception as exc:
                        st.warning(f"{up.name}: {exc}")
                        continue
                    for d in ds:
                        label = f"{base} · {d.label}" if len(ds) > 1 else base
                        series.append((_dedup_label(label, seen_labels), d))
        elif source == "CSV file(s)":
            ups = st.file_uploader(
                "CSV (Potential, Current)", type=["csv", "txt"], key="tafel_csv",
                accept_multiple_files=True,
            )
            if ups:
                for up in ups:
                    base = up.name.rsplit(".", 1)[0]
                    try:
                        ds = _cached_lsv_datasets(up.getvalue(), None)
                    except Exception as exc:
                        st.warning(f"{up.name}: {exc}")
                        continue
                    for d in ds:
                        label = f"{base} · {d.label}" if len(ds) > 1 else base
                        series.append((_dedup_label(label, seen_labels), d))
    except Exception as exc:
        st.error(f"Could not load data: {exc}")

    if not series:
        st.info("⬆️ Upload one or more files, or a ZIP of a data folder, to begin.")
    return series


# Literature-typical pH for common supporting electrolytes, for the RHE
# conversion below. Real pH depends on activity coefficients (concentrated
# strong acid/base solutions don't follow pH = -log10[conc] exactly) and on
# temperature — these are the values commonly cited as-is in the ORR/HER
# electrocatalysis literature for RHE conversion, not first-principles
# calculations; check against your own electrolyte when precision matters,
# or measure it directly.
_ELECTROLYTE_PH_PRESETS: dict[str, float] = {
    "0.1 M KOH": 13.0,
    "1 M KOH": 14.0,
    "0.1 M NaOH": 13.0,
    "1 M NaOH": 14.0,
    "0.5 M H2SO4": 0.3,
    "1 M H2SO4": 0.1,
    "0.1 M HClO4": 1.0,
}


_RHE_MODES = ["Already vs RHE", "Reference electrode + electrolyte pH",
              "Direct calibration offset"]


def _render_rhe_conversion(key_prefix: str, default_ph: float = 13.0,
                           default_ref_electrode: str | None = None,
                           default_electrolyte: str | None = None,
                           default_mode: str = "Already vs RHE"):
    """Render the "convert to RHE" controls shared by the LSV/K-L/ORR tabs.

    Three ways to get there, picked with a radio button:

    - **Already vs RHE** — no conversion.
    - **Reference electrode + electrolyte pH** — the standard
      ``E(RHE) = E(measured) + E°(ref vs NHE) + 0.0592·pH`` formula; the
      electrolyte dropdown fills in a literature-typical pH (see
      :data:`_ELECTROLYTE_PH_PRESETS`), overridable via "Custom".
    - **Direct calibration offset** — bypasses the formula entirely for a
      reference electrode that was calibrated directly against a reversible
      hydrogen electrode *in the same electrolyte* (common practice for
      Hg/HgO, Hg/Hg2SO4, etc.), since that single measured offset already
      captures the electrolyte's actual pH/activity/junction-potential
      effects more accurately than the nominal formula.

    Returns a callable ``to_rhe(potential_array) -> ndarray``.
    """
    st.markdown("**Reference electrode → RHE conversion**")
    mode_idx = _RHE_MODES.index(default_mode) if default_mode in _RHE_MODES else 0
    mode = st.radio(
        "How should potentials be converted to the RHE scale?",
        _RHE_MODES, index=mode_idx,
        key=f"{key_prefix}_rhe_mode", horizontal=True,
        help="Use 'Direct calibration offset' if you've measured your "
             "reference electrode against an RHE in your own electrolyte "
             "(e.g. Hg/HgO vs a Pt-wire RHE under H2 bubbling) — that "
             "single offset is usually more accurate than the nominal "
             "E° + pH formula.",
    )

    if mode == "Already vs RHE":
        return lambda pot: np.asarray(pot, dtype=float)

    if mode == "Direct calibration offset":
        st.caption(
            "E(RHE) = E(measured) + offset — skips the reference-electrode/"
            "pH formula since a directly measured offset already accounts "
            "for the electrolyte's actual pH, ionic strength, and junction "
            "potential."
        )
        offset = st.number_input(
            "Calibrated offset (V)", value=0.926, step=0.001, format="%.3f",
            key=f"{key_prefix}_rhe_offset",
            help="0.926 V is a commonly cited Hg/HgO-vs-RHE offset in 1 M "
                 "KOH (pH 14) — the same number the formula mode gives for "
                 "'Hg/HgO, 1 M KOH' at pH 14. In 0.1 M KOH (pH 13) the "
                 "formula gives 0.867 V instead, so do not carry the 1 M "
                 "value across electrolytes: replace it with your own "
                 "electrode's measured value.",
        )
        return lambda pot: np.asarray(pot, dtype=float) + offset

    rc1, rc2 = st.columns([2, 1])
    ref_names = list(tafel.REFERENCE_ELECTRODES) + ["Custom"]
    ref_idx = (ref_names.index(default_ref_electrode)
               if default_ref_electrode in ref_names else 0)
    ref_choice = rc1.selectbox(
        "Reference electrode used for the input data", ref_names,
        index=ref_idx, key=f"{key_prefix}_ref_electrode",
    )
    if ref_choice == "Custom":
        e_ref = rc2.number_input(
            "E° vs NHE (V)", value=0.000, step=0.001, format="%.3f",
            key=f"{key_prefix}_ref_custom",
        )
    else:
        e_ref = tafel.REFERENCE_ELECTRODES[ref_choice]
        rc2.metric("E° vs NHE (V)", f"{e_ref:.3f}")

    pc1, pc2 = st.columns([2, 1])
    electrolyte_names = list(_ELECTROLYTE_PH_PRESETS) + ["Custom"]
    default_electrolyte_idx = (
        electrolyte_names.index(default_electrolyte)
        if default_electrolyte in electrolyte_names else 0
    )
    electrolyte_choice = pc1.selectbox(
        "Electrolyte (sets a literature-typical pH — pick Custom to enter "
        "your own or a measured value)",
        electrolyte_names, index=default_electrolyte_idx,
        key=f"{key_prefix}_electrolyte",
    )
    if electrolyte_choice == "Custom":
        ph = pc2.number_input(
            "Electrolyte pH", min_value=0.0, max_value=14.0, value=default_ph,
            step=0.1, key=f"{key_prefix}_ph_custom",
        )
    else:
        ph = _ELECTROLYTE_PH_PRESETS[electrolyte_choice]
        pc2.metric("pH", f"{ph:g}")
    st.caption(
        f"↪ E(RHE) = E(measured) + {e_ref:.3f} V + "
        f"{tafel.NERNST_SLOPE_V_PER_PH:.4f} × {ph:g} = "
        f"E(measured) + {(e_ref + tafel.NERNST_SLOPE_V_PER_PH * ph):.3f} V. "
        "Electrolyte pH values above are literature-typical, not measured "
        "for your specific sample — check them (or use 'Direct calibration "
        "offset' instead) when precision matters."
    )
    return lambda pot: tafel.to_rhe(np.asarray(pot, dtype=float), e_ref, ph)


def render_tafel_tab() -> None:
    st.subheader("LSV analysis")
    st.caption(
        "Onset potential, benchmark overpotentials at fixed current "
        "densities, and the Tafel slope of a polarization curve — fitting "
        "E = a + b·log₁₀|i| over the linear (activation-controlled) region; "
        "**b** is the Tafel slope, reported as its positive magnitude "
        "(mV/dec) per literature convention. This tab has its own file "
        "upload and does not use the EIS/LSV data loaded in the sidebar."
    )

    series = _tafel_data_loader()
    if not series:
        return
    st.success(
        f"Loaded {len(series)} sample(s): " + ", ".join(lbl for lbl, _ in series)
    )

    top1, top2, top3 = st.columns(3)
    _reaction_defaults = ["Auto-detect (recommended)"] + _TAFEL_REACTIONS
    default_reaction_choice = top1.selectbox(
        "Reaction for newly added samples", _reaction_defaults, index=0,
        key="tafel_default_reaction",
        help="**Auto-detect** reads each sample's own curve — the sign of the "
             "faradaic current and where on the RHE scale it flows — and "
             "assigns HER / HOR / OER / ORR accordingly, per sample. That "
             "matters beyond labelling: the reaction sets the equilibrium "
             "potential used for η, the mechanistic benchmark table, and the "
             "overpotential cap on the auto-detected Tafel window. Pick a "
             "fixed reaction here to override the detection for every "
             "sample; each sample can still be changed individually below.",
    )
    auto_reaction = default_reaction_choice.startswith("Auto-detect")
    default_reaction = (_TAFEL_REACTIONS[0] if auto_reaction
                        else default_reaction_choice)
    font_size = top2.selectbox(
        "Axis font size (pt)", _JOURNAL_FONT_SIZES, index=0,
        key="tafel_font_size_quick",
        help="Quick control; the full appearance panel below overrides it.",
    )
    vicinity_pct = top3.number_input(
        "Final-plot vicinity margin (%)", min_value=0, max_value=300,
        value=25, step=5, key="tafel_vicinity_pct",
        help="Once the fit range below looks right, the final Tafel plot "
             "keeps only the points in this window around it (as a % of "
             "the fit-region width) instead of the full sweep — set to a "
             "large value (e.g. 300) to show everything.",
    )

    st.markdown("**Current unit & electrode area**")
    # Auto-detect a sensible default from the first sample's column header
    # (e.g. "Current (mA/cm2)"), same heuristic the EIS/LSV tabs use — the
    # user can still override or opt into density conversion below.
    cur_default, _ = _detect_units("", series[0][1].label) if series else (None, None)
    native_unit = cur_default or "mA"
    cur1, cur2, cur3 = st.columns([1.4, 1, 1])
    convert_density = cur1.checkbox(
        "Convert absolute current to current density (÷ electrode area)",
        value=False, key="tafel_convert_density",
        help="Enable if the uploaded current is absolute (e.g. mA) and you "
             "want the Tafel analysis reported as a current density "
             "(e.g. mA/cm²) instead.",
    )
    if convert_density:
        abs_default = native_unit if native_unit in _ABS_CURRENT_UNITS else _ABS_CURRENT_UNITS[0]
        current_unit = cur2.selectbox(
            "Desired current unit", _ABS_CURRENT_UNITS,
            index=_ABS_CURRENT_UNITS.index(abs_default), key="tafel_current_unit_abs",
            help="Current is auto-detected from the file and rescaled to "
                 "this unit, then divided by the electrode area.",
        )
        area_cm2 = cur3.number_input(
            "Electrode area (cm²)", min_value=1e-4, value=0.04, step=0.01,
            format="%.4f", key="tafel_area_cm2",
        )
        display_unit = f"{current_unit}/cm²"
        st.caption(
            f"↪ Current density = current ({current_unit}) ÷ {area_cm2:g} "
            f"cm² → reported as {display_unit}."
        )
        if correction.is_density_unit(native_unit):
            st.warning(
                f"Detected current is already a density ({native_unit}); "
                "the ÷ area conversion below assumes an absolute current. "
                "Uncheck it to use the density as-is."
            )
    else:
        opts_all = _ABS_CURRENT_UNITS + _DENSITY_CURRENT_UNITS
        cur_idx = opts_all.index(cur_default) if cur_default in opts_all else 0
        current_unit = cur2.selectbox(
            "Desired current unit", opts_all, index=cur_idx, key="tafel_current_unit",
            help="Current is auto-detected from the file; pick a different "
                 "unit of the same kind (e.g. mA → µA, or mA/cm² → A/cm²) "
                 "to rescale the values.",
        )
        area_cm2 = None
        display_unit = current_unit
        if correction.is_density_unit(current_unit) != correction.is_density_unit(native_unit):
            st.warning(
                f"Detected current unit ({native_unit}) and the selected "
                f"unit ({current_unit}) are different kinds (absolute vs. "
                "density) — converting between them needs an electrode "
                "area; enable 'Convert absolute current to current "
                "density' above. Values are shown unconverted for now."
            )
    if cur_default:
        st.caption(
            f"ℹ️ Detected current unit from the column header: **{native_unit}**. "
            "Values are automatically rescaled if you pick a different unit above."
        )

    st.markdown("**Onset potential & benchmark current densities**")
    st.caption(
        "E_onset is where |current| first departs from the flat pre-onset "
        "baseline (same detector the Tafel fit-range auto-start uses). The "
        "benchmark values below read off the potential at one or more fixed "
        "current densities (e.g. **j = 10 mA/cm²**, the standard OER/HER "
        "activity benchmark; **j = 2 mA/cm²** is also common for lower-"
        "current comparisons) — reported as an overpotential η = |E − E_eq| "
        "for HER/HOR/OER/ORR/NO₃RR/N₂RR/CO₂RR (using each sample's own "
        "equilibrium potential), or as the raw potential for reactions "
        "without a single well-defined E_eq."
    )
    target_j_text = st.text_input(
        f"Target current densities ({display_unit}), comma-separated",
        value="10, 2", key="tafel_target_j",
    )
    target_js: list[float] = []
    for tok in target_j_text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            value = float(tok)
        except ValueError:
            st.warning(f"Ignoring unparseable target current density {tok!r}.")
            continue
        # De-duplicate: each benchmark becomes a results column and a bar
        # chart whose labels are derived from the value, so entering
        # "10, 10" would raise a DuplicateWidgetID and take the whole tab
        # down. De-duplicate on the *formatted* value, not the float: "%g"
        # collapses to 6 significant figures, so 1000000 and 1000000.1 are
        # distinct floats that both render "1e+06" and would collide just the
        # same. Order is preserved so the columns stay in the order typed.
        if not any(f"{value:g}" == f"{seen:g}" for seen in target_js):
            target_js.append(value)
        else:
            st.caption(f"↪ Ignoring repeated target current density {value:g}.")
    if not convert_density and not correction.is_density_unit(current_unit):
        st.caption(
            "↪ Current is not a density — these benchmarks compare across "
            "samples most meaningfully when normalized by electrode area "
            "('Convert ... to current density' above)."
        )

    style = plot_style_controls("tafel")
    font_size = style["font_size"]
    _PALETTE_T = style["palette"]

    labels = [lbl for lbl, _ in series]
    chosen = st.multiselect(
        "Samples to combine (journal-style overlay)",
        labels, default=labels[: min(len(labels), 8)], key="tafel_chosen",
    )
    if not chosen:
        st.info("Select at least one sample to plot.")
        return
    chosen_series = [(lbl, d) for lbl, d in series if lbl in chosen]

    to_rhe_fn = _render_rhe_conversion("tafel", default_ph=7.0)

    # A box-select drag on the Tafel plot below (key "tafel_plot_select")
    # lands here as a fresh selection event; apply it to the affected
    # sample(s)' fit-range widgets exactly once (a signature check stops it
    # from re-clobbering a later manual slider drag on an unrelated rerun).
    prior_points = []
    prior_event = st.session_state.get("tafel_plot_select")
    if prior_event is not None:
        try:
            prior_points = prior_event.get("selection", {}).get("points", [])
        except Exception:
            prior_points = []
    sel_sig = _selection_signature(prior_points)
    apply_selection = bool(prior_points) and sel_sig != st.session_state.get(
        "_tafel_last_selection_sig"
    )
    if apply_selection:
        st.session_state["_tafel_last_selection_sig"] = sel_sig

    # Reserved now, filled in further down (after the fit-range sliders
    # below have run and `fits` exists) — this keeps the Original LSV plot
    # visually above the sliders while the sliders themselves render closer
    # to the Tafel plot they actually control.
    lsv_plot_slot = st.container()

    fits = []
    with st.expander(
        "🔧 Per-sample Tafel fit range & line color — auto-detected starting "
        "near the reaction onset; drag to adjust, or box-select the region "
        "directly on the Tafel plot below",
        expanded=True,
    ):
        for i, (lbl, d) in enumerate(chosen_series):
            mask = d.current != 0
            if not mask.any():
                st.warning(f"{lbl}: all currents are zero — skipped.")
                continue
            # Strip any approach/vertex leg before analysis: a raw export
            # often ramps to the sweep's vertex first, and that leg would
            # otherwise be treated as part of the scan (it breaks the
            # baseline the onset detector reads, and puts two currents at
            # the same potential).
            pot_raw = to_rhe_fn(d.potential[mask])
            cur_scaled = _rescale_current(d.current[mask], native_unit, current_unit)
            cur_raw = cur_scaled / area_cm2 if convert_density else cur_scaled
            pot, cur = sweep.clean_sweep(pot_raw, cur_raw)
            trimmed = len(pot_raw) - len(pot)
            log_i = tafel.log_current(cur)
            n = len(pot)

            # Infer the reaction from this sample's own curve, unless the
            # user pinned one at the top of the tab. Done before the
            # selectbox renders so the detection can seed its initial value.
            guess = None
            reaction_key = f"tafel_reaction_{i}"
            if auto_reaction:
                try:
                    guess = tafel.infer_reaction(pot, cur)
                except Exception:
                    guess = None
            seeded = guess.reaction if guess is not None else default_reaction
            if seeded not in _TAFEL_REACTIONS:
                seeded = default_reaction

            # The reaction drives the overpotential cap on the auto window,
            # so read whatever the widget currently holds (a user override
            # from a previous run wins over the fresh guess).
            reaction_for_range = st.session_state.get(reaction_key, seeded)
            e_eq_for_range = tafel.REACTION_E_EQ_V_RHE.get(reaction_for_range)
            a0, a1 = tafel.auto_tafel_range(
                pot, log_i, current=cur, e_eq=e_eq_for_range
            )
            range_key = f"tafel_range_{i}"
            if apply_selection:
                try:
                    sel_range = _selection_range_for_sample(prior_points, lbl, log_i)
                except Exception:
                    sel_range = None
                if sel_range is not None:
                    st.session_state[range_key] = sel_range
            c0, cg, cr, c1, c2 = st.columns([1.0, 1.0, 0.8, 1.7, 0.5])
            display_name = c0.text_input(
                "Legend name", value=f"Sample {i + 1}", key=f"tafel_name_{i}",
                help="Shown in the plot legend; edit if you'd rather show "
                     "the auto-detected file/column name.",
            )
            # Only pre-fill a guess when the label carries the "(2)"/"(3)"
            # suffix _dedup_label adds for a genuine repeat upload (same
            # base file/folder loaded more than once) — otherwise leave it
            # blank so a one-off sample doesn't look like it was grouped.
            default_group = (_default_replicate_group(lbl)
                             if _REPLICATE_SUFFIX.search(lbl) else "")
            replicate_group = cg.text_input(
                "Replicate group", value=default_group, key=f"tafel_group_{i}",
                placeholder="Blank = no repeats",
                help="Give two or more samples the same group name to treat "
                     "them as repeat scans of the same underlying sample — "
                     "their fitted values (Tafel slope, onset, η@j, …) are "
                     "then also reported as a mean ± SD in the **Replicate "
                     "statistics** section below. Pre-filled only when the "
                     "sample was auto-detected as a repeat upload.",
            )
            reaction_help = ("This sample's legend and analysis are grouped "
                             "under this reaction.")
            if guess is not None:
                reaction_help += f" Auto-detected: {guess.reason}."
            sample_reaction = cr.selectbox(
                "Reaction", _TAFEL_REACTIONS,
                index=_TAFEL_REACTIONS.index(seeded),
                key=reaction_key, help=reaction_help,
            )
            if guess is not None and sample_reaction == guess.reaction:
                badge = {"high": "✅", "medium": "🟡", "low": "⚠️"}[guess.confidence]
                cr.caption(f"{badge} auto ({guess.confidence})")
            elif guess is not None:
                cr.caption(f"↪ overriding auto-detect ({guess.reaction})")
            slider_kwargs = ({} if range_key in st.session_state
                             else {"value": (int(a0), int(a1))})
            start, stop = c1.slider(
                "Fit range (index)", 0, n, key=range_key, **slider_kwargs,
                help="Auto-starts near the current onset and extends while "
                     "the potential vs log|i| relationship stays linear; "
                     "drag either handle, or box-select the region on the "
                     "Tafel plot below, to fine-tune.",
            )
            c1.caption(
                f"↪ Potential vs RHE: {pot[start]:.3f} V to "
                f"{pot[min(stop, n - 1)]:.3f} V"
            )
            color = c2.color_picker(
                "Color", value=_PALETTE_T[i % len(_PALETTE_T)],
                key=f"tafel_color_{i}_{style['palette_name']}",
            )
            if trimmed:
                c1.caption(
                    f"↪ {trimmed} point(s) of approach/vertex leg removed "
                    "before analysis."
                )
            fits.append({"label": display_name or lbl, "orig_label": lbl,
                         "reaction": sample_reaction,
                         "reaction_guess": guess,
                         "replicate_group": replicate_group or (display_name or lbl),
                         "potential": pot, "current": cur, "log_i": log_i,
                         "start": start, "stop": stop, "color": color})

    if not fits:
        st.error("No usable samples (all-zero current).")
        return

    # Group samples by reaction (stable) so each reaction's traces sit
    # together in plot/legend order, enabling a split legend per reaction.
    fits.sort(key=lambda f: _TAFEL_REACTIONS.index(f["reaction"]))

    axis_font = {"family": "Arial", "size": font_size}
    small_font = {"family": "Arial", "size": max(12, round(font_size * 0.5))}

    # Original LSV (linear-scale polarization curve), before the log-current
    # Tafel transform — shown for context alongside the derived Tafel plot.
    # Rendered into the placeholder reserved above the fit-range sliders, so
    # it stays visually first even though it needs `fits` (built by those
    # sliders) to know each sample's chosen color/label/reaction.
    with lsv_plot_slot:
        st.markdown("**Original LSV (polarization curve)**")
        d_by_label = {lbl: d for lbl, d in chosen_series}

        # Data extent across all chosen samples — default for the optional
        # manual potential/current range below (view-only; never cuts data
        # — see :func:`_range_controls`). Note this covers HER's typically
        # very negative potentials (well below -0.8 V vs RHE) fine, since
        # the range is read straight off the real data extent with no
        # artificial floor.
        all_pot_full = np.concatenate([to_rhe_fn(d.potential) for d in d_by_label.values()])
        all_cur_full = np.concatenate([
            _rescale_current(d.current, native_unit, current_unit) for d in d_by_label.values()
        ])
        if convert_density:
            all_cur_full = all_cur_full / area_cm2
        lsv_x_lo, lsv_x_hi = float(np.min(all_pot_full)), float(np.max(all_pot_full))
        lsv_y_lo, lsv_y_hi = float(np.min(all_cur_full)), float(np.max(all_cur_full))
        lsv_x_range, lsv_y_range = _range_controls(
            "lsv_orig", "Potential (V)", f"Current ({display_unit})",
            lsv_x_lo, lsv_x_hi, lsv_y_lo, lsv_y_hi,
        )

        lsv_fig = go.Figure()
        lsv_plotted: dict[str, list] = {}
        for meta in fits:
            d = d_by_label.get(meta["orig_label"])
            if d is None:
                continue
            pot_full = to_rhe_fn(d.potential)
            cur_full_scaled = _rescale_current(d.current, native_unit, current_unit)
            cur_full = cur_full_scaled / area_cm2 if convert_density else cur_full_scaled
            lsv_plotted[f"{meta['label']} — E vs RHE (V)"] = list(pot_full)
            lsv_plotted[f"{meta['label']} — current ({display_unit})"] = list(cur_full)
            lsv_fig.add_trace(go.Scatter(
                x=pot_full, y=cur_full, mode="lines", name=meta["label"],
                legendgroup=meta["reaction"],
                line={"color": meta["color"], "width": 3},
            ))
        lsv_fig.update_layout(
            title=({"text": "Original LSV",
                    "font": {"family": style["font_family"], "size": font_size}}
                   if style.get("show_title") else None),
            template="plotly_white", height=460, font=axis_font,
            legend={"x": 0.02, "y": 0.98, "xanchor": "left",
                    "yanchor": "top", "font": small_font,
                    "bgcolor": "rgba(255,255,255,0.7)", "tracegroupgap": 18},
            margin=_journal_margin(font_size,
                                   titled=bool(style.get("show_title"))),
        )
        lsv_fig.update_xaxes(
            title={"text": "Potential vs RHE / V", "font": {"family": "Arial", "size": font_size}},
            tickfont={"family": "Arial", "size": font_size}, **_BOX_AXIS_STYLE,
        )
        lsv_fig.update_yaxes(
            title={"text": f"Current ({display_unit})",
                   "font": {"family": "Arial", "size": font_size}},
            tickfont={"family": "Arial", "size": font_size}, **_BOX_AXIS_STYLE,
        )
        if lsv_x_range is not None:
            lsv_fig.update_xaxes(range=lsv_x_range)
        if lsv_y_range is not None:
            lsv_fig.update_yaxes(range=lsv_y_range)
        st.plotly_chart(lsv_fig, width="stretch", config=_PLOTLY_EDIT_CONFIG)
        figure_downloads(
            lsv_fig, "original_lsv_plot", key="png_lsv_original",
            what="LSV plot", data=_padded_frame(lsv_plotted),
        )

    results = []
    for f in fits:
        try:
            r = tafel.fit_tafel(f["potential"], f["log_i"], f["start"], f["stop"])
        except Exception as exc:
            st.warning(f"{f['label']}: fit failed ({exc})")
            continue
        results.append((f, r))
    if not results:
        return

    st.markdown("**Tafel plot**")
    distinct_reactions = []
    for f, _ in results:
        if f["reaction"] not in distinct_reactions:
            distinct_reactions.append(f["reaction"])
    multi_reaction = len(distinct_reactions) > 1

    # vicinity_pct's window around each sample's fit region — used only as
    # the *default view*, never to drop points from the trace, so nothing
    # keeps the data itself from being there (the CSV export below, and the
    # plot once the range is widened, always cover every point).
    view_x_los, view_x_his, view_y_los, view_y_his = [], [], [], []
    for f, _ in results:
        start, stop = f["start"], f["stop"]
        n_pts = len(f["log_i"])
        margin = round((stop - start) * vicinity_pct / 100.0)
        v0, v1 = max(0, start - margin), min(n_pts, stop + margin)
        if v1 > v0:
            view_x_los.append(float(np.min(f["log_i"][v0:v1])))
            view_x_his.append(float(np.max(f["log_i"][v0:v1])))
            view_y_los.append(float(np.min(f["potential"][v0:v1])))
            view_y_his.append(float(np.max(f["potential"][v0:v1])))
    default_x_lo = min(view_x_los) if view_x_los else 0.0
    default_x_hi = max(view_x_his) if view_x_his else 1.0
    default_y_lo = min(view_y_los) if view_y_los else -0.8
    default_y_hi = max(view_y_his) if view_y_his else 0.8
    st.caption(
        f"Default view: within {vicinity_pct:g}% of each sample's fit-region "
        "width around its selected linear range (adjust the % above, or the "
        "range below — e.g. HER data commonly needs potentials well below "
        "-0.8 V vs RHE). No data is ever dropped from the plot or its "
        "export, only the visible window changes."
    )
    tafel_x_range, tafel_y_range = _range_controls(
        "tafel_plot", "log|i| (X)", "Potential vs RHE / V (Y)",
        default_x_lo, default_x_hi, default_y_lo, default_y_hi,
    )
    if tafel_x_range is None:
        tafel_x_range = [default_x_lo, default_x_hi]
    if tafel_y_range is None:
        tafel_y_range = [default_y_lo, default_y_hi]

    fig = go.Figure()
    tafel_plotted: dict[str, list] = {}
    for f, r in results:
        color = f["color"]
        n_pts = len(f["log_i"])
        grp = f["reaction"]
        # Full per-sample data in the trace — the range controls above set
        # the default/adjustable *view*, not which points are plotted.
        tafel_plotted[f"{f['label']} — log10|i| ({display_unit})"] = list(f["log_i"])
        tafel_plotted[f"{f['label']} — E vs RHE (V)"] = list(f["potential"])
        fig.add_trace(go.Scatter(
            x=f["log_i"], y=f["potential"], mode="lines+markers",
            name=f["label"],
            legendgroup=grp,
            marker={"size": 10, "color": color, "opacity": 0.55},
            line={"color": color, "width": 1},
            customdata=[f["orig_label"]] * n_pts,
        ))
        xs = f["log_i"][f["start"]:f["stop"]]
        xline = np.array([float(np.min(xs)), float(np.max(xs))])
        yline = r.slope_v_per_dec * xline + r.intercept_v
        slope_abs = abs(r.slope_mv_per_dec)
        fit_color = _darken(color)
        tafel_plotted[f"{f['label']} — fit line log10|i|"] = list(xline)
        tafel_plotted[f"{f['label']} — fit line E (V)"] = list(yline)
        fig.add_trace(go.Scatter(
            x=xline, y=yline, mode="lines", showlegend=False, legendgroup=grp,
            name=f"{f['label']} — linear (Tafel) fit, {slope_abs:.0f} mV/dec",
            line={"color": fit_color, "width": 3, "dash": "dot"},
        ))
        xmid, ymid = float(np.mean(xline)), float(np.mean(yline))
        label_text = (
            f"{slope_abs:.0f} mV/dec ({f['reaction']})" if multi_reaction
            else f"{slope_abs:.0f} mV/dec"
        )
        fig.add_annotation(
            x=xmid, y=ymid, text=label_text,
            showarrow=False, yshift=16,
            # Sized off the chosen figure font rather than a fixed 22 pt,
            # which was illegible next to a 36 pt axis and oversized next to
            # a small one.
            font={"family": style["font_family"], "color": fit_color,
                  "size": max(11, round(font_size * 0.62))},
        )

    title_text = (
        f"Tafel plot — {' & '.join(distinct_reactions)}" if multi_reaction
        else "Tafel plot"
    )
    fig.update_layout(
        title=({"text": title_text,
                "font": {"family": style["font_family"], "size": font_size}}
               if style.get("show_title") else None),
        template="plotly_white",
        height=560,
        font=axis_font,  # baseline (inherited by legend/annotations unless overridden)
        legend={"x": 0.02, "y": 0.98, "xanchor": "left",
                "yanchor": "top", "font": small_font,
                "bgcolor": "rgba(255,255,255,0.7)", "tracegroupgap": 18},
        margin=_journal_margin(font_size,
                               titled=bool(style.get("show_title"))),
        dragmode="select",
    )
    # Axis titles/ticks are the "axes" text: always the full 28/36 pt Arial.
    # A sparse tick count (~4-5, via _BOX_AXIS_STYLE) keeps a publication-
    # style plot uncluttered.
    fig.update_xaxes(
        title={"text": f"log₁₀ |Current| ({display_unit})",
               "font": {"family": "Arial", "size": font_size}},
        tickfont={"family": "Arial", "size": font_size}, range=tafel_x_range,
        **_BOX_AXIS_STYLE,
    )
    fig.update_yaxes(
        title={"text": "Potential vs RHE / V", "font": {"family": "Arial", "size": font_size}},
        tickfont={"family": "Arial", "size": font_size}, range=tafel_y_range,
        **_BOX_AXIS_STYLE,
    )
    st.caption(
        "🖱️ Drag a box around a sample's linear region to set its fit range "
        "directly (mouse now defaults to box-select instead of zoom — use "
        "the toolbar's zoom icon or double-click to reset the view). Drag "
        "the legend or a slope label to reposition it before exporting."
    )
    st.plotly_chart(
        fig, width="stretch", key="tafel_plot_select",
        on_select="rerun", selection_mode=["box"],
        config=_PLOTLY_EDIT_CONFIG,
    )
    figure_downloads(
        fig, "tafel_combined_plot", key="png_tafel", what="Tafel plot",
        data=_padded_frame(tafel_plotted),
    )

    rows = []
    fit_warnings: list[str] = []
    for f, r in results:
        slope_abs = abs(r.slope_mv_per_dec)
        ref = tafel.nearest_reference(slope_abs, f["reaction"])
        row = {
            "Sample": f["label"],
            "Replicate group": f["replicate_group"],
            "Reaction": f["reaction"],
            "Tafel slope (mV/dec)": round(slope_abs, 1),
            # A slope with no uncertainty cannot be compared against another
            # lab's number, so the fit's own standard error travels with it.
            "± 95% CI (mV/dec)": (
                round(r.slope_ci95_mv_per_dec, 1)
                if np.isfinite(r.slope_ci95_mv_per_dec) else None
            ),
            "R2": round(r.r_squared, 4),
            "Fit points": f["stop"] - f["start"],
            # Decades of current spanned: the usual reason a Tafel slope is
            # wrong is a window too short to be a Tafel region at all, and
            # R² cannot show that (a few adjacent points on any smooth curve
            # fit a line beautifully).
            "Decades fitted": (
                round(r.decades, 2) if np.isfinite(r.decades) else None
            ),
            "Nearest mechanistic benchmark": (
                f"~{ref[0]:.0f} mV/dec ({ref[1]})" if ref else "—"
            ),
        }
        for issue in r.quality_warnings:
            fit_warnings.append(f"**{f['label']}** — {issue}")
        try:
            row["E_onset (V vs RHE)"] = round(
                tafel.onset_potential(f["potential"], f["current"]), 3
            )
        except ValueError:
            row["E_onset (V vs RHE)"] = None
        e_eq = tafel.REACTION_E_EQ_V_RHE.get(f["reaction"])
        for target_j in target_js:
            e_at_j = tafel.potential_at_current_density(
                f["potential"], f["current"], target_j
            )
            label = (f"η @ j={target_j:g} (V)" if e_eq is not None
                     else f"E @ j={target_j:g} (V vs RHE)")
            row[label] = (
                round(abs(e_at_j - e_eq) if e_eq is not None else e_at_j, 3)
                if e_at_j is not None else None
            )
        rows.append(row)
    summary = pd.DataFrame(rows)

    # A sample's "Replicate group" defaults to its own name (see the
    # per-sample controls above) so genuine repeat uploads auto-group; blank
    # it back out here for groups of exactly one so a one-off sample doesn't
    # display a group name it was never actually grouped under.
    group_counts = summary["Replicate group"].value_counts()
    has_replicates = bool((group_counts > 1).any())
    summary["Replicate group"] = [
        g if group_counts[g] > 1 else "" for g in summary["Replicate group"]
    ]

    st.markdown("**Results summary**")
    st.dataframe(summary, width="stretch", hide_index=True)
    if fit_warnings:
        st.warning(
            "**Fit-quality notes** (the slope is still reported, but treat "
            "these with care):\n\n- " + "\n- ".join(fit_warnings)
        )
    st.download_button(
        "⬇️ Download Tafel fit summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="tafel_fit_summary.csv",
        mime="text/csv",
        key="dl_tafel_summary",
    )

    # Render the numbers explicitly (the CSV above keeps the full-precision
    # floats).
    display = summary.copy()
    display["R2"] = [f"{v:.4f}" for v in summary["R2"]]
    _journal_table_figure(  # CSV of this table is the summary button above
        display, font_size, "tafel_results_table", key="png_tafel_table",
        style=style,
    )

    if has_replicates:
        st.markdown("**Replicate statistics**")
        st.caption(
            "Samples sharing the same **Replicate group** name above (set in "
            "the per-sample expander) are treated as repeat scans of one "
            "underlying sample — each fitted value below is a mean ± SD "
            "across that group's members (SD omitted for a group of one)."
        )
        group_col = "Replicate group"
        exclude = {group_col, "Sample", "Reaction", "Nearest mechanistic benchmark"}
        numeric_cols = [
            c for c in summary.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(summary[c])
        ]
        rep_rows = []
        # Blanked-out singletons all share "" — exclude them so they don't
        # get lumped together as a fake shared group.
        grouped = summary[summary[group_col] != ""]
        for group_name, gdf in grouped.groupby(group_col, sort=False):
            rep_row = {
                "Replicate group": group_name,
                "Reaction": "/".join(dict.fromkeys(gdf["Reaction"])),  # order-preserving unique
                "N replicates": len(gdf),
            }
            for c in numeric_cols:
                vals = gdf[c].dropna().to_numpy(dtype=float)
                if len(vals) == 0:
                    rep_row[c] = None
                elif len(vals) == 1:
                    rep_row[c] = f"{vals[0]:.4g}"
                else:
                    rep_row[c] = f"{np.mean(vals):.4g} ± {np.std(vals, ddof=1):.4g}"
            rep_rows.append(rep_row)
        replicate_summary = pd.DataFrame(rep_rows)
        st.dataframe(replicate_summary, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download replicate statistics (CSV)",
            data=replicate_summary.to_csv(index=False).encode("utf-8"),
            file_name="tafel_replicate_statistics.csv", mime="text/csv",
            key="dl_tafel_replicate_stats",
        )
        _journal_table_figure(
            replicate_summary, font_size, "tafel_replicate_stats_table",
            key="png_tafel_replicate_table", what="Replicate statistics table",
            style=style,
        )

    if len(results) >= 2 and target_js:
        st.markdown("**Benchmark overpotential comparison**")
        st.caption(
            "One bar per sample by default; samples sharing a **Replicate "
            "group** name above are instead merged into one bar (its mean "
            "across those repeat runs, with an error bar of ±1 SD) — at "
            "each benchmark current density configured above (e.g. j = 10 "
            "and 2 " + display_unit + ")."
        )
        # Bar per sample, unless it has a real (non-blank) replicate group —
        # then bar per group. A blank group is unique to its own sample, so
        # one-off samples never get silently merged with unrelated ones.
        bar_key = [
            g if g else s
            for g, s in zip(summary["Replicate group"], summary["Sample"])
        ]
        for bar_i, target_j in enumerate(target_js):
            target_cols = [c for c in summary.columns if f"j={target_j:g}" in c]
            if not target_cols:
                continue
            # Different samples may fall under different label conventions
            # (η vs raw E, depending on whether their reaction has a defined
            # E_eq) — merge onto one column, since exactly one is populated
            # per row.
            values = summary[target_cols[0]].copy()
            for c in target_cols[1:]:
                values = values.combine_first(summary[c])
            bar_df = pd.DataFrame({
                "Sample": bar_key, "value": values,
            }).dropna(subset=["value"])
            if bar_df.empty:
                continue
            grouped = bar_df.groupby("Sample", sort=False)["value"]
            means = grouped.mean()
            stds = grouped.std(ddof=1).fillna(0.0)
            ylabel = target_cols[0] if len(target_cols) == 1 else f"Value @ j={target_j:g}"
            bar_colors = [_PALETTE_T[i % len(_PALETTE_T)] for i in range(len(means))]
            fig_bar = go.Figure(go.Bar(
                x=list(means.index), y=means.to_numpy(),
                error_y={"type": "data", "array": stds.to_numpy(), "visible": True},
                marker={"color": bar_colors},
            ))
            fig_bar.update_layout(
                title=({"text": f"Benchmark @ j = {target_j:g} {display_unit}",
                        "font": {"family": style["font_family"],
                                 "size": font_size}}
                       if style.get("show_title") else None),
                template="plotly_white", height=420,
                font={"family": "Arial", "size": font_size},
                margin=_journal_margin(
                    font_size, titled=bool(style.get("show_title"))),
            )
            fig_bar.update_xaxes(
                tickfont={"family": "Arial", "size": font_size}, **_BOX_AXIS_STYLE,
            )
            fig_bar.update_yaxes(
                title={"text": ylabel, "font": {"family": "Arial", "size": font_size}},
                tickfont={"family": "Arial", "size": font_size}, **_BOX_AXIS_STYLE,
            )
            st.plotly_chart(fig_bar, width="stretch", config=_PLOTLY_EDIT_CONFIG)
            figure_downloads(
                fig_bar, f"overpotential_bar_j{target_j:g}",
                key=f"png_tafel_bar_{bar_i}",
                what=f"Benchmark overpotential bar chart @ j={target_j:g}",
                data=_padded_frame({
                    "Sample": list(means.index),
                    "Mean": list(means.to_numpy()),
                    "SD": list(stds.to_numpy()),
                }),
            )

    st.markdown("**Analysis**")
    for rxn in distinct_reactions:
        entries = [
            (f["label"], abs(r.slope_mv_per_dec), r.r_squared)
            for f, r in results if f["reaction"] == rxn
        ]
        if multi_reaction:
            st.markdown(f"*{rxn}*")
        st.write(tafel.analysis_paragraph(rxn, entries))


# --------------------------------------------------------------------------- #
# Koutecky-Levich (K-L) analysis tab (independent data source)                #
# --------------------------------------------------------------------------- #
# The K-L fit must run in A/cm² (the units the 0.62 Levich prefactor is
# defined for), but the plot is conventionally shown in mA/cm². Since
# 1/j_[cm²/mA] = (1/j_[cm²/A]) / 1000, this divides the plotted/exported
# 1/j only -- never the fit that n is derived from.
_KL_PLOT_J_SCALE = 1000.0


def render_kl_tab() -> None:
    st.subheader("Koutecky–Levich (K-L) analysis")
    st.caption(
        "Classic multi-rotation-rate RDE analysis: at each of several fixed "
        "potentials, 1/j vs 1/√ω (ω = 2π·rpm/60, the angular rotation rate) "
        "is linear — Koutecký & Levich, *Zh. Fiz. Khim.* **1958**, *32*, "
        "1565; see also Bard & Faulkner, *Electrochemical Methods*, 2nd ed., "
        "Wiley, 2001, Ch. 9. The **intercept** gives the kinetic (mass-"
        "transport-free) current density j_k, and the **slope** gives the "
        "Levich constant B — hence the electron-transfer number n, once the "
        "electrolyte's O₂ diffusion coefficient, kinematic viscosity, and "
        "bulk concentration are known. This tab has its own file upload and "
        "does not use data loaded elsewhere."
    )

    samples, unit_hint = _orr_data_loader(
        key_prefix="kl",
        file_help="Disk/working-electrode current file(s) for this sample "
                  "— one file per rotation rate",
    )
    if not samples:
        return

    labels = [lbl for lbl, _ in samples]
    active_label = st.selectbox(
        "Active sample", labels, key="kl_active_sample",
        help="K-L analysis is shown for one sample at a time; switch here "
             "to compare catalysts.",
    )
    df = dict(samples)[active_label]

    style = plot_style_controls("kl", show_markers=True)
    font_size = style["font_size"]
    _PALETTE_K = style["palette"]

    def _style_axes(fig, xtitle, ytitle, legend_position="top-left"):
        _journal_axes_style(fig, xtitle, ytitle, font_size,
                            legend_position=legend_position, style=style)

    st.markdown("**Current unit & electrode area**")
    # The uploaded column may hold either an absolute current (the usual RDE
    # export) or an already-normalised current density. Offering both here
    # matters for more than labelling: an already-per-area column must NOT be
    # divided by the electrode area again, and n comes out wrong by exactly
    # that factor if it is.
    kl_unit_options = _ABS_CURRENT_UNITS + _DENSITY_CURRENT_UNITS
    _kl_unit_default = (
        kl_unit_options.index(unit_hint) if unit_hint in kl_unit_options
        else kl_unit_options.index("A")
    )
    cur1, cur2, cur3 = st.columns([1.3, 1.3, 1])
    current_unit = cur1.selectbox(
        "Current unit as uploaded", kl_unit_options, index=_kl_unit_default,
        key="kl_current_unit",
        help="Auto-detected from the file's column header when recognisable "
             "(RDE exports are often in A) — override if it's wrong. Pick a "
             "per-area unit (e.g. mA/cm²) if the file already holds a "
             "current density, so it isn't divided by the area a second time.",
    )
    uploaded_is_density = correction.is_density_unit(current_unit)
    desired_unit = cur2.selectbox(
        "Desired current unit (for the RDE plot)",
        _DENSITY_CURRENT_UNITS if uploaded_is_density else _ABS_CURRENT_UNITS,
        index=(_DENSITY_CURRENT_UNITS if uploaded_is_density
               else _ABS_CURRENT_UNITS).index(current_unit),
        key="kl_desired_unit",
        help="Display unit for the RDE curves below. The Koutecky-Levich "
             "fit and n always use A/cm² internally regardless of this.",
    )
    area_cm2 = cur3.number_input(
        "Electrode area (cm²)", min_value=1e-4, value=0.196, step=0.001,
        format="%.4f", key="kl_area_cm2",
        disabled=uploaded_is_density,
        help="0.196 cm² is the standard 5 mm-diameter RDE glassy-carbon disk. "
             "Used to convert the uploaded absolute current into the A/cm² "
             "current density the Koutecky-Levich fit and n require. Not "
             "needed (and disabled) when the upload is already a density.",
    )
    display_unit = desired_unit if uploaded_is_density else f"{desired_unit}/cm²"
    if unit_hint:
        st.caption(f"ℹ️ Detected current unit from the column header: **{unit_hint}**.")

    to_rhe_fn = _render_rhe_conversion(
        "kl", default_ph=13.0,
        default_ref_electrode="Hg/HgO, 1 M KOH", default_electrolyte="0.1 M KOH",
        default_mode="Reference electrode + electrolyte pH",
    )

    st.markdown("**Electrolyte O₂ transport parameters** (for n via the Levich constant)")
    st.caption(
        "Approximate literature values (25 °C) — check against your own "
        "electrolyte/temperature when precision matters; see Bard & "
        "Faulkner, *Electrochemical Methods*, 2nd ed., Table 9.3.1, and refs. "
        "therein."
    )
    ec1, ec2 = st.columns([1.4, 2])
    preset_names = list(orr.ELECTROLYTE_PRESETS) + ["Custom"]
    preset_choice = ec1.selectbox(
        "Electrolyte", preset_names, index=0, key="kl_electrolyte_preset",
    )
    if preset_choice == "Custom":
        cc1, cc2, cc3 = ec2.columns(3)
        diff_coeff = cc1.number_input(
            "D(O₂) (cm²/s)", min_value=1e-7, value=1.9e-5, step=1e-6,
            format="%.2e", key="kl_D",
        )
        viscosity = cc2.number_input(
            "Viscosity ν (cm²/s)", min_value=1e-4, value=1.0e-2, step=1e-4,
            format="%.2e", key="kl_nu",
        )
        bulk_c = cc3.number_input(
            "Bulk O₂ conc. C (mol/cm³)", min_value=1e-9, value=1.2e-6,
            step=1e-8, format="%.2e", key="kl_C",
        )
    else:
        diff_coeff, viscosity, bulk_c = orr.ELECTROLYTE_PRESETS[preset_choice]
        ec2.caption(
            f"D = {diff_coeff:.3g} cm²/s · ν = {viscosity:.3g} cm²/s · "
            f"C = {bulk_c:.3g} mol/cm³"
        )

    pot_rhe = to_rhe_fn(df["potential"].to_numpy(dtype=float))
    disk_raw = df["disk_current"].to_numpy(dtype=float)
    # NOTE: cleaning happens per rotation rate below, not here — this frame
    # concatenates every rpm, so a global monotonic-segment search would
    # treat the joins between them as sweep reversals.
    # Display series, in whatever unit the RDE plot is set to. An upload that
    # is already a density is only rescaled between per-area units; an
    # absolute upload is rescaled then divided by the electrode area.
    disk = _rescale_current(disk_raw, current_unit, desired_unit)
    if not uploaded_is_density:
        disk = disk / area_cm2
    # The Koutecky-Levich slope -> n conversion (B = 0.62 F D^(2/3) nu^(-1/6) C)
    # is only valid when j is a true current density in A/cm^2 -- the units
    # the standard/literature F, D, nu, C values are given in. Using whatever
    # unit the RDE-curve plot happens to display (e.g. mA/cm^2) throws n off
    # by orders of magnitude, so the fit/n calculation always builds its own
    # A/cm^2 array regardless of the display settings above. Critically, an
    # already-per-area upload converts straight to A/cm^2 and must NOT be
    # divided by the area again -- doing so scaled n by 1/area (~5x for a
    # standard 0.196 cm^2 tip) on top of any unit-prefix error.
    if uploaded_is_density:
        disk_a_cm2 = _rescale_current(disk_raw, current_unit, "A/cm²")
    else:
        disk_a_cm2 = _rescale_current(disk_raw, current_unit, "A") / area_cm2
    rpm_arr = df["rpm"].to_numpy(dtype=float)
    rpm_values = sorted(set(rpm_arr.tolist()))
    if len(rpm_values) < 3:
        st.error(
            f"{active_label}: only {len(rpm_values)} rotation rate(s) loaded "
            "— Koutecky-Levich needs at least 3 for a meaningful fit."
        )
        return

    st.markdown(f"**RDE curves — {active_label}**")
    rde_x_range, rde_y_range = _range_controls(
        "kl_rde", "Potential (V)", f"Disk current ({display_unit})",
        float(np.min(pot_rhe)), float(np.max(pot_rhe)),
        float(np.min(disk)), float(np.max(disk)),
    )
    fig_rde = go.Figure()
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    curves_a_cm2: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for i, rv in enumerate(rpm_values):
        m = np.isclose(rpm_arr, rv)
        # Strip this rotation rate's approach/vertex leg before sorting: the
        # leg revisits potentials the main sweep also covers, and the
        # np.interp below (which the K-L fit reads its currents from) then
        # sees duplicated x values and silently mixes the two legs.
        p_clean, j_clean, j_a_clean = sweep.clean_sweep(
            pot_rhe[m], disk[m], disk_a_cm2[m]
        )
        order = np.argsort(p_clean)
        p, j = p_clean[order], j_clean[order]
        curves[rv] = (p, j)
        curves_a_cm2[rv] = (p, j_a_clean[order])
        fig_rde.add_trace(go.Scatter(
            x=p, y=j, mode="lines", name=f"{rv:g} rpm",
            line={"color": _PALETTE_K[i % len(_PALETTE_K)],
                  "width": style["line_width"]},
        ))
    _style_axes(fig_rde, "Potential vs RHE / V", f"Disk current ({display_unit})")
    if rde_x_range is not None:
        fig_rde.update_xaxes(range=rde_x_range)
    if rde_y_range is not None:
        fig_rde.update_yaxes(range=rde_y_range)
    st.plotly_chart(fig_rde, width="stretch", config=_PLOTLY_EDIT_CONFIG)
    rde_cols = {}
    for rv, (p, j) in curves.items():
        rde_cols[f"{rv:g}rpm — Potential vs RHE (V)"] = list(p)
        rde_cols[f"{rv:g}rpm — Disk current ({display_unit})"] = list(j)
    figure_downloads(
        fig_rde, f"kl_rde_curves_{active_label}", key="png_kl_rde",
        what="RDE curves", data=_padded_frame(rde_cols),
    )

    lo = max(p.min() for p, _ in curves.values())
    hi = min(p.max() for p, _ in curves.values())
    if lo >= hi:
        st.error(
            f"{active_label}: rotation-rate curves don't share a common "
            "potential range to analyze."
        )
        return

    st.markdown("**Koutecky–Levich plot**")
    st.caption(
        f"All {len(rpm_values)} rotation rates overlap only over "
        f"**{lo:.3f} – {hi:.3f} V vs RHE**, so the analysis window below is "
        "clamped to that; the ORR-active default (0.3–0.8 V) is trimmed "
        "wherever it falls outside. To reach further, extend the sweep range "
        "of the rotation rate that limits it."
    )
    kc1, kc2, kc3 = st.columns([1, 1, 1.6])
    kl_pot_lo = kc1.number_input(
        "Analysis window min (V vs RHE)", value=round(max(lo, 0.3), 3),
        step=0.05, format="%.2f", key="kl_pot_lo",
        help="Defaults to the ORR kinetically-active regime (~0.3-0.8 V vs "
             "RHE) intersected with the potential range all rotation rates "
             "share — outside that window the disk current is either "
             "capacitive-only (near OCP) or fully mass-transport-limited, "
             "neither of which is meaningful for a K-L fit.",
    )
    kl_pot_hi = kc2.number_input(
        "Analysis window max (V vs RHE)", value=round(min(hi, 0.8), 3),
        step=0.05, format="%.2f", key="kl_pot_hi",
    )
    n_points = kc3.number_input(
        "Number of analysis potentials (evenly spaced across the window)",
        min_value=1, max_value=15, value=5, step=1, key="kl_n_points",
    )
    pot_window_lo, pot_window_hi = max(lo, kl_pot_lo), min(hi, kl_pot_hi)
    if pot_window_lo >= pot_window_hi:
        st.error(
            f"{active_label}: the analysis window doesn't overlap the "
            f"potential range all rotation rates share ({lo:.3f}-{hi:.3f} V "
            "vs RHE) — widen it above."
        )
        return
    analysis_pots = np.linspace(pot_window_lo, pot_window_hi, int(n_points))
    st.caption(
        f"↪ n below assumes the uploaded disk current is in **{current_unit}** "
        f"and the electrode area is **{area_cm2:g} cm²** (both set above) — "
        "double-check those first if n still looks off, since n is very "
        "sensitive to both."
    )

    kl_rows = []
    kl_diagnostics: list[dict] = []
    fig_kl = go.Figure()
    kl_data: dict[str, list] = {}
    n_out_of_range = False
    for idx, ap in enumerate(analysis_pots):
        omegas_rpm, j_at_pot = [], []
        for rv in rpm_values:
            # Always the A/cm^2 series -- see the note above disk_a_cm2:
            # the Levich slope -> n conversion is only correct in these units.
            p, j = curves_a_cm2[rv]
            jval = float(np.interp(ap, p, j))
            if jval == 0 or not np.isfinite(jval):
                continue
            omegas_rpm.append(rv)
            j_at_pot.append(jval)
        if len(omegas_rpm) < 3:
            continue
        invj = 1.0 / np.array(j_at_pot)
        try:
            fit = orr.fit_koutecky_levich(omegas_rpm, np.array(j_at_pot))
        except ValueError:
            continue
        n_val = orr.levich_slope_to_n(fit.slope, diff_coeff, viscosity, bulk_c)
        if n_val is not None and not (-0.5 <= n_val <= 4.5):
            n_out_of_range = True
        color = _PALETTE_K[idx % len(_PALETTE_K)]
        # x = omega^-1/2 with omega = 2*pi*rpm/60 in rad/s -- the units the
        # 0.62 Levich prefactor is defined for (the alternative 0.201 form
        # takes rpm directly; 0.62*sqrt(2*pi/60) = 0.201 reconciles them).
        x = 1.0 / np.sqrt(orr.angular_velocity(omegas_rpm))
        # The fit itself stays in A/cm^2 (required by the 0.62 prefactor);
        # only the plotted/exported y is rescaled to the mA/cm^2 convention
        # the ORR literature usually shows. 1/j_[mA/cm2] = (1/j_[A/cm2])/1000,
        # so both the points and the fitted line scale by the same factor and
        # the line still overlays the points exactly.
        y = invj / _KL_PLOT_J_SCALE
        fig_kl.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=f"{ap:.3f} V",
            marker={"color": color, "size": max(4, style["marker_size"]),
                    "symbol": style["marker_symbol"]},
        ))
        xline = np.array([float(np.min(x)), float(np.max(x))])
        yline = (fit.slope * xline + fit.intercept) / _KL_PLOT_J_SCALE
        fig_kl.add_trace(go.Scatter(
            x=xline, y=yline, mode="lines", showlegend=False,
            line={"color": _darken(color), "width": style["line_width"],
                  "dash": style["fit_dash"]},
        ))
        kl_data[f"{ap:.3f}V — omega^-1/2 (rad^-1/2 s^1/2)"] = list(x)
        kl_data[f"{ap:.3f}V — 1/j (cm²/mA)"] = list(y)
        kl_rows.append({
            "Potential (V vs RHE)": round(float(ap), 3),
            "KL slope": float(fit.slope),
            "R²": round(fit.r_squared, 4),
            # A K-L fit through near-zero currents (just past onset) produces
            # an arithmetically valid but physically meaningless n; surfacing
            # the reliability test keeps such rows from being read as results.
            "Reliable?": "yes" if fit.is_reliable else "no — low R²",
            # Reported as its positive magnitude (0-4), matching literature
            # convention -- the sign otherwise just tracks the disk current's
            # own recorded sign (negative for a cathodic/reduction sweep).
            "n — electron transfer no. (Levich)": (
                round(abs(n_val), 2) if n_val is not None else None
            ),
            "j_k (mA/cm²)": (
                round(fit.kinetic_current_density * _KL_PLOT_J_SCALE, 4)
                if fit.kinetic_current_density is not None else None
            ),
            "Rotation rates used (rpm)": ", ".join(f"{rv:g}" for rv in omegas_rpm),
        })
        # Measured |j| beside the ideal 4-electron Levich value at the same
        # rotation rate. Their ratio *is* n/4, so a row that reads e.g.
        # "2.8 vs 5.7" immediately explains an n of ~2 without any further
        # digging: the fit is faithfully reporting the current it was given.
        b_4e_row = (0.62 * 4 * orr.FARADAY_C_PER_MOL * diff_coeff ** (2 / 3)
                    * viscosity ** (-1 / 6) * bulk_c)
        ideal_4e = [
            b_4e_row * np.sqrt(orr.angular_velocity(rv)) * _KL_PLOT_J_SCALE
            for rv in omegas_rpm
        ]
        measured = [abs(v) * _KL_PLOT_J_SCALE for v in j_at_pot]
        kl_diagnostics.append({
            "E (V vs RHE)": round(float(ap), 3),
            "measured |j| (mA/cm²)": ", ".join(f"{v:.3g}" for v in measured),
            "ideal 4e⁻ |j| (mA/cm²)": ", ".join(f"{v:.3g}" for v in ideal_4e),
            "measured / ideal": ", ".join(
                f"{m / i:.2f}" if i else "—" for m, i in zip(measured, ideal_4e)
            ),
            "n (raw, unrounded)": f"{n_val:.4f}" if n_val is not None else "—",
        })

    unreliable = [r for r in kl_rows if r.get("Reliable?", "").startswith("no")]
    if unreliable:
        st.warning(
            f"{len(unreliable)} of {len(kl_rows)} analysis potentials gave a "
            "poorly-conditioned Koutecky-Levich fit (R² < 0.95), marked "
            "**no** in the *Reliable?* column. This is normal for potentials "
            "close to the onset, where the current is near zero and 1/j "
            "amplifies noise without limit — narrow the analysis window to "
            "the mass-transport-limited region and they should disappear. "
            "Do not quote n from those rows."
        )
    if n_out_of_range:
        st.warning(
            "One or more analysis potentials gave n well outside the "
            "physically meaningful 0-4 range for O₂ reduction — double-"
            "check the electrode area and electrolyte O₂ transport "
            "parameters above (n is very sensitive to both). Open the "
            "diagnostics below to see the intermediate values."
        )

    if kl_diagnostics:
        # Expected 4-electron limiting current density at 1600 rpm for the
        # transport parameters in use -- the quickest sanity check there is:
        # if the measured |j| is nowhere near it, the problem is upstream of
        # the fit (wrong units, wrong area, or wrong electrolyte preset).
        b_4e = (0.62 * 4 * orr.FARADAY_C_PER_MOL * diff_coeff ** (2 / 3)
                * viscosity ** (-1 / 6) * bulk_c)
        jlim_4e_ma = b_4e * np.sqrt(orr.angular_velocity(1600.0)) * _KL_PLOT_J_SCALE
        with st.expander("🔬 n diagnostics — intermediate values behind each fit"):
            st.caption(
                f"Inputs in use: current read as **{current_unit}**"
                + ("" if uploaded_is_density else f", area **{area_cm2:g} cm²**")
                + f" · D = {diff_coeff:.3g} cm²/s · ν = {viscosity:.3g} cm²/s "
                f"· C = {bulk_c:.3g} mol/cm³. For a 4-electron ORR under these "
                f"conditions |j_lim| at 1600 rpm should be ≈ **{jlim_4e_ma:.2f} "
                "mA/cm²** — if your measured |j| at 1600 rpm is far from that, "
                "the mismatch is in the data/units or the transport "
                "parameters, not the fit itself."
            )
            st.dataframe(
                pd.DataFrame(kl_diagnostics), width="stretch", hide_index=True,
            )

    if not kl_rows:
        st.warning(
            "Not enough overlapping rotation-rate data to fit any analysis "
            "potential — try fewer/different points, or check the uploaded "
            "files share a common potential range."
        )
        return

    _style_axes(
        fig_kl, "ω<sup>−1/2</sup> (rad<sup>−1/2</sup> s<sup>1/2</sup>)",
        "j<sup>−1</sup> (cm² mA<sup>−1</sup>)",
        legend_position="outside-right",
    )
    st.plotly_chart(fig_kl, width="stretch", config=_PLOTLY_EDIT_CONFIG)
    st.caption(
        "Koutecký–Levich: 1/j = 1/j_k + 1/(B·ω^½), with "
        "B = 0.62·n·F·D^(2/3)·ν^(−1/6)·C and ω = 2π·rpm/60 (rad/s). "
        "Slope = 1/B (gives n), intercept = 1/j_k. Plotted in **cm²/mA** by "
        "the usual convention; the fit behind n runs in A/cm², which the "
        "0.62 prefactor requires."
    )
    figure_downloads(
        fig_kl, f"kl_plot_{active_label}", key="png_kl_plot",
        what="Koutecky–Levich plot", data=_padded_frame(kl_data),
    )

    kl_summary = pd.DataFrame(kl_rows)
    st.markdown("**Results summary**")
    st.dataframe(kl_summary, width="stretch", hide_index=True)
    st.download_button(
        "⬇️ Download K-L summary (CSV)",
        data=kl_summary.to_csv(index=False).encode("utf-8"),
        file_name=f"kl_summary_{active_label}.csv", mime="text/csv",
        key="dl_kl_summary",
    )
    kl_display = kl_summary.copy()
    kl_display["KL slope"] = [f"{v:.3e}" for v in kl_summary["KL slope"]]
    _journal_table_figure(  # CSV of this table is the summary button above
        kl_display, font_size, f"kl_results_table_{active_label}",
        key="png_kl_table", style=style,
    )


# --------------------------------------------------------------------------- #
# ORR / RRDE analysis tab (independent data source)                          #
# --------------------------------------------------------------------------- #
_RPM_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*rpm", re.IGNORECASE)
_ORR_POT_HINTS = ("potential", "voltage", "e ", "e(", "ewe", "e/v", "e vs")


def _guess_rpm_from_filename(name: str) -> float | None:
    """Parse a rotation rate from an instrument-exported filename, e.g.
    ``Disk Current vs Disk Potential (1600 RPM).csv`` -> 1600.0."""
    m = _RPM_PATTERN.search(name)
    return float(m.group(1)) if m else None


def _guess_role_from_filename(name: str) -> str:
    """Disk vs ring, guessed from filenames like the pattern above; ring is
    checked first since a ring file's name usually also contains 'disk' (as
    in '... vs Disk Potential')."""
    return "Ring" if "ring" in name.lower() else "Disk"


def _orr_numeric_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Coerce every column to numeric and return (coerced_df, usable_columns)."""
    coerced = df.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    cols = [c for c in coerced.columns if coerced[c].notna().any()]
    return coerced, cols


def _guess_current_unit_from_cols(cols: list) -> str | None:
    """Guess a current unit (e.g. ``"mA"``, ``"A/cm²"``) from raw column
    headers — the same header-sniffing :func:`_detect_units` does for the
    Tafel tab's column labels — so the K-L/ORR "Current unit as uploaded"
    picker can default to what the file actually says instead of always
    assuming A."""
    lowered = {c: str(c).strip().lower() for c in cols}
    pot_col = next(
        (c for c, name in lowered.items() if any(h in name for h in _ORR_POT_HINTS)),
        None,
    )
    for c in cols:
        if c == pot_col:
            continue
        unit = _detect_units("", str(c))[0]
        if unit:
            return unit
    return None


def _orr_read_file(up, key_prefix: str = "orr",
                   slot: str = "0") -> pd.DataFrame | None:
    """Read one uploaded file's raw table (no column interpretation yet).

    ``slot`` identifies the upload's position (sample index / file index) and
    keys the sheet picker. Filenames cannot: uploading two files that happen
    to share a name — routine in RRDE work, where disk and ring exports come
    out of per-rpm folders with identical names — would build the same widget
    key twice and crash the tab with ``DuplicateWidgetID``.
    """
    try:
        if up.name.lower().endswith((".xlsx", ".xls")):
            sheets = _cached_sheets(up.getvalue())
            sheet = sheets[0]
            if len(sheets) > 1:
                sheet = st.selectbox(
                    f"Sheet for {up.name}", sheets,
                    key=f"{key_prefix}_sheet_{slot}",
                )
            return _cached_table(up.getvalue(), sheet)
        return _cached_table(up.getvalue(), None)
    except Exception as exc:
        st.warning(f"{up.name}: {exc}")
        return None


def _orr_merge_entries(
    entries: list[tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float]],
    sample_name: str,
) -> pd.DataFrame | None:
    """Merge a sample's ``(potential, disk_current_or_None,
    ring_current_or_None, rpm)`` entries into one tidy table: columns
    ``potential``, ``disk_current``, ``ring_current`` (if any rpm has ring
    data), ``rpm``. A compiled-workbook entry already carries both
    electrodes; raw per-electrode entries are paired up by matching rotation
    rate (disk and ring share the instrument's own potential grid, recorded
    simultaneously) — interpolating the ring onto the disk's grid if the two
    don't already match exactly."""
    by_rpm: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for pot, disk, ring, rpm_val in entries:
        slot = by_rpm.setdefault(rpm_val, {})
        if disk is not None:
            slot["disk"] = (pot, disk)
        if ring is not None:
            slot["ring"] = (pot, ring)

    frames = []
    for rpm_val, slot in sorted(by_rpm.items()):
        if "disk" not in slot:
            st.warning(f"{sample_name}: no Disk data at {rpm_val:g} rpm — skipped.")
            continue
        pot_d, cur_d = slot["disk"]
        frame = pd.DataFrame({
            "potential": pot_d, "disk_current": cur_d, "rpm": rpm_val,
        })
        if "ring" in slot:
            pot_r, cur_r = slot["ring"]
            if len(pot_r) == len(pot_d) and np.allclose(pot_r, pot_d, atol=1e-6):
                frame["ring_current"] = cur_r
            elif len(pot_r) >= 2:
                # np.argsort alone is not enough: a ring export that repeats a
                # potential (a vertex point, a quantised axis) leaves duplicate
                # x, where np.interp's result depends on which duplicate came
                # first. ascending_xy collapses them to their mean.
                xr, yr = sweep.ascending_xy(pot_r, cur_r)
                frame["ring_current"] = np.interp(pot_d, xr, yr)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def _orr_table_to_entry(
    coerced: pd.DataFrame, cols: list, filename: str, rpm_val: float,
    role: str | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, float] | None:
    """Turn one already-numeric-coerced table into an ``(potential, disk,
    ring, rpm)`` entry. With 4+ numeric columns, treats it as a compiled
    Potential/Disk/Potential/Ring workbook (``role`` ignored); with exactly 2,
    treats it as one electrode's raw Potential/Current file, using ``role``
    if given or else guessing from ``filename``."""
    lowered = {c: str(c).strip().lower() for c in cols}
    pot_col = next(
        (c for c, name in lowered.items() if any(h in name for h in _ORR_POT_HINTS)),
        cols[0],
    )
    if len(cols) == 3:
        # Potential, Disk, Ring on one shared potential grid. Without this
        # branch a 3-column export fell through to the 2-column path, which
        # takes cols[-1] as "the current" — silently reading the *ring* column
        # as the disk current and discarding the disk entirely.
        others = [c for c in cols if c != pot_col]
        disk_col, ring_col = others[0], others[1]
        # Disambiguate by header where the file says which is which.
        if "ring" in str(disk_col).lower() and "ring" not in str(ring_col).lower():
            disk_col, ring_col = ring_col, disk_col
        pot = coerced[pot_col].to_numpy(dtype=float)
        disk = coerced[disk_col].to_numpy(dtype=float)
        ring = coerced[ring_col].to_numpy(dtype=float)
        mask = np.isfinite(pot) & np.isfinite(disk) & np.isfinite(ring)
        return pot[mask], disk[mask], ring[mask], rpm_val
    if len(cols) >= 4:
        others = [c for c in cols if c != pot_col]
        disk_col, ring_pot_col, ring_col = others[0], others[1], others[2]
        pot = coerced[pot_col].to_numpy(dtype=float)
        disk = coerced[disk_col].to_numpy(dtype=float)
        pot_r = coerced[ring_pot_col].to_numpy(dtype=float)
        ring = coerced[ring_col].to_numpy(dtype=float)
        mask = np.isfinite(pot) & np.isfinite(disk)
        pot, disk = pot[mask], disk[mask]
        mask_r = np.isfinite(pot_r) & np.isfinite(ring)
        pot_r, ring = pot_r[mask_r], ring[mask_r]
        if len(pot_r) == len(pot) and np.allclose(pot_r, pot, atol=1e-6):
            ring_aligned = ring
        elif len(pot_r) >= 2:
            xr, yr = sweep.ascending_xy(pot_r, ring)
            ring_aligned = np.interp(pot, xr, yr)
        else:
            ring_aligned = None
        return pot, disk, ring_aligned, rpm_val

    cur_col = next((c for c in cols if c != pot_col), cols[-1])
    pot = coerced[pot_col].to_numpy(dtype=float)
    cur = coerced[cur_col].to_numpy(dtype=float)
    mask = np.isfinite(pot) & np.isfinite(cur)
    pot, cur = pot[mask], cur[mask]
    resolved_role = role or _guess_role_from_filename(filename)
    return (pot, cur, None, rpm_val) if resolved_role == "Disk" else (pot, None, cur, rpm_val)


def _orr_extract_zip_samples(
    upload, key_prefix: str = "orr",
) -> tuple[list[tuple[str, pd.DataFrame]], str | None]:
    """Batch-load RRDE files from a ZIP of a data folder (or several sample
    folders zipped together) — for pasting in a whole export at once instead
    of picking files one by one. Files are grouped into one sample per
    top-level folder inside the zip (files at the zip's own root all become
    one sample, named after the zip); role (disk/ring) and rotation rate are
    guessed entirely from filenames, the same way real RDE/RRDE software
    names its exports (e.g. ``Disk Current vs Disk Potential (1600
    RPM).csv``) — there is no per-file tagging UI here, since a batch upload
    may contain many files; use the per-sample uploaders above instead if a
    file needs correcting by hand. Also returns a current-unit guess sniffed
    from the first readable file's column headers (see
    :func:`_guess_current_unit_from_cols`), or ``None`` if nothing recognisable
    was found.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(upload.getvalue()))
    except zipfile.BadZipFile:
        st.error(f"{upload.name}: not a valid .zip file.")
        return [], None
    total = sum(info.file_size for info in zf.infolist())
    if total > data_io.MAX_UNCOMPRESSED_BYTES:
        st.error(
            f"{upload.name}: zip expands too large; rejected as a possible "
            "decompression bomb."
        )
        return [], None

    members = [
        m for m in zf.namelist()
        if m.lower().endswith((".csv", ".txt", ".xlsx", ".xls"))
        and "__MACOSX" not in m and not m.rsplit("/", 1)[-1].startswith(".")
    ]
    if not members:
        st.warning(f"{upload.name}: no CSV/Excel files found inside.")
        return [], None
    if len(members) > MAX_ZIP_MEMBERS:
        st.error(
            f"{upload.name}: contains {len(members)} data files, more than "
            f"the {MAX_ZIP_MEMBERS}-file limit. Split it into smaller zips."
        )
        return [], None

    default_sample = upload.name.rsplit(".", 1)[0]
    grouped: dict[str, list] = {}
    skipped = 0
    unit_hint = None
    for member in members:
        parts = member.replace("\\", "/").split("/")
        sample_name = parts[0] if len(parts) > 1 else default_sample
        filename = parts[-1]
        try:
            raw_bytes = zf.read(member)
        except Exception:
            skipped += 1
            continue
        try:
            if filename.lower().endswith((".xlsx", ".xls")):
                sheets = data_io.list_sheets(io.BytesIO(raw_bytes))
                df = data_io.read_table(io.BytesIO(raw_bytes), sheet=sheets[0])
            else:
                df = data_io.read_table(io.BytesIO(raw_bytes), sheet=None)
        except Exception:
            skipped += 1
            continue
        coerced, cols = _orr_numeric_columns(df)
        if len(cols) < 2:
            skipped += 1
            continue
        if unit_hint is None:
            unit_hint = _guess_current_unit_from_cols(cols)
        rpm_val = _guess_rpm_from_filename(filename) or 1600.0
        entry = _orr_table_to_entry(coerced, cols, filename, rpm_val)
        if entry is None:
            skipped += 1
            continue
        grouped.setdefault(sample_name, []).append(entry)

    if skipped:
        st.caption(
            f"↪ {skipped} file(s) inside the zip were skipped (unreadable or "
            "not enough numeric columns)."
        )

    samples = []
    for sample_name, entries in grouped.items():
        df = _orr_merge_entries(entries, sample_name)
        if df is not None:
            samples.append((sample_name, df))
    return samples, unit_hint


def _orr_data_loader(
    key_prefix: str = "orr",
    file_help: str = "Disk/ring current file(s) for this sample",
) -> tuple[list[tuple[str, pd.DataFrame]], str | None]:
    """File uploader local to the calling tab — independent of the other
    tabs. Two ways to load data, usable together:

    - **Per-sample uploaders**, one file (or several) at a time, with
      per-file role/rpm tagging you can correct by hand.
    - **A ZIP of a whole data folder**, for batch-loading many samples/
      rotation rates at once (see :func:`_orr_extract_zip_samples`) — role
      and rpm are guessed purely from filenames, with no per-file UI.

    Each upload is one of two layouts, auto-detected by column count:

    - **Raw per-electrode files** (2 numeric columns: Potential, Current) —
      the layout most RDE/RRDE instrument software exports as one small file
      per rotation rate *and* electrode, e.g. ``Disk Current vs Disk
      Potential (1600 RPM).csv`` / ``Ring Current vs ...``. Rotation rate and
      disk/ring role are guessed from the filename and can be corrected.
    - **Compiled workbooks** (4+ numeric columns: Potential, Disk current,
      Potential, Ring current, already paired for one rotation rate) — only
      a rotation-rate tag is needed.

    Returns a list of ``(sample_name, dataframe)``, each dataframe having
    columns ``potential``, ``disk_current``, ``ring_current`` (if present),
    ``rpm``, plus a current-unit guess sniffed from the first file's column
    headers (see :func:`_guess_current_unit_from_cols`), or ``None``.
    """
    st.markdown(
        "**Data source** (independent of the other tabs) — for each sample, "
        "upload its current files: either raw per-electrode files (one file "
        "per rotation rate and electrode, just Potential + Current — the "
        "layout most RDE/RRDE software exports) *or* a compiled workbook "
        "with Potential/Disk-current/Potential/Ring-current already paired "
        "for one rotation rate. Rotation rate and disk/ring role are guessed "
        "from the filename and can be corrected below."
    )

    samples: list[tuple[str, pd.DataFrame]] = []
    unit_hint: str | None = None
    # Sample labels are the selector values *and* part of the per-sample
    # widget keys, and callers index back into the list with dict(samples).
    # Two zip subfolders, or a manual sample typed with the same name as a
    # zip one, would otherwise silently collapse into one entry and collide
    # on `orr_rrde_rpms_{label}`. Disambiguate exactly as the LSV tab does.
    seen_labels: dict[str, int] = {}

    with st.expander("📦 Batch upload — a ZIP of a whole data folder"):
        st.caption(
            "For many samples/rotation rates at once: zip your data folder "
            "(one subfolder per sample, containing its Disk/Ring files — "
            "extra nesting inside a sample's subfolder is fine) and upload "
            "it here. Role and rotation rate are read from filenames "
            "automatically, same as the per-sample uploaders below, but "
            "with no per-file correction UI — use those below instead for "
            "any file that needs fixing by hand."
        )
        zip_up = st.file_uploader(
            "ZIP file", type=["zip"], key=f"{key_prefix}_zip",
        )
        if zip_up is not None:
            zip_samples, zip_unit_hint = _orr_extract_zip_samples(zip_up, key_prefix=key_prefix)
            if zip_samples:
                st.success(
                    f"Loaded {len(zip_samples)} sample(s) from {zip_up.name}: "
                    + ", ".join(lbl for lbl, _ in zip_samples)
                )
            samples.extend(
                (_dedup_label(lbl, seen_labels), d) for lbl, d in zip_samples
            )
            if unit_hint is None:
                unit_hint = zip_unit_hint

    n_samples = st.number_input(
        "Number of additional samples to load individually", min_value=0,
        max_value=8, value=(0 if samples else 1), step=1,
        key=f"{key_prefix}_n_samples",
    )

    for i in range(int(n_samples)):
        with st.expander(f"Sample {i + 1} — files", expanded=(i == 0 and not samples)):
            sample_name = st.text_input(
                "Sample name", value=f"Sample {i + 1}", key=f"{key_prefix}_sample_name_{i}"
            )
            ups = st.file_uploader(
                file_help,
                type=["csv", "txt", "xlsx", "xls"], accept_multiple_files=True,
                key=f"{key_prefix}_files_{i}",
            )
            if not ups:
                st.caption("No files uploaded for this sample yet.")
                continue

            entries = []  # (potential, disk_current, ring_current_or_None, rpm)
            for j, up in enumerate(ups):
                raw = _orr_read_file(up, key_prefix=key_prefix, slot=f"{i}_{j}")
                if raw is None:
                    continue
                coerced, cols = _orr_numeric_columns(raw)
                if len(cols) < 2:
                    st.warning(f"{up.name}: fewer than 2 numeric columns, skipped.")
                    continue
                if unit_hint is None:
                    unit_hint = _guess_current_unit_from_cols(cols)
                rpm_guess = _guess_rpm_from_filename(up.name) or 1600.0

                if len(cols) >= 4:
                    fc1, fc2 = st.columns([3, 1])
                    fc1.caption(f"{up.name} — compiled (Potential/Disk/Potential/Ring)")
                    rpm_val = fc2.number_input(
                        "rpm", min_value=1.0, value=float(rpm_guess), step=100.0,
                        key=f"{key_prefix}_rpm_{i}_{j}", label_visibility="collapsed",
                    )
                    entry = _orr_table_to_entry(coerced, cols, up.name, rpm_val)
                else:
                    fc1, fc2, fc3 = st.columns([2.2, 1, 1])
                    fc1.caption(up.name)
                    role_default = _guess_role_from_filename(up.name)
                    role = fc2.selectbox(
                        "Role", ["Disk", "Ring"],
                        index=0 if role_default == "Disk" else 1,
                        key=f"{key_prefix}_role_{i}_{j}", label_visibility="collapsed",
                    )
                    rpm_val = fc3.number_input(
                        "rpm", min_value=1.0, value=float(rpm_guess), step=100.0,
                        key=f"{key_prefix}_rpm_{i}_{j}", label_visibility="collapsed",
                    )
                    entry = _orr_table_to_entry(coerced, cols, up.name, rpm_val, role=role)
                if entry is not None:
                    entries.append(entry)
            if not entries:
                continue
            df = _orr_merge_entries(entries, sample_name)
            if df is not None:
                samples.append((_dedup_label(sample_name, seen_labels), df))
    return samples, unit_hint


def render_orr_tab() -> None:
    st.subheader("ORR / RRDE analysis")
    st.caption(
        "At one rotation rate (conventionally 1600 rpm): onset potential, "
        "half-wave potential E½ (the steepest point of the disk curve), and "
        "— after removing the mass-transport contribution — the Tafel "
        "slope; plus, when ring current is available, the electron-transfer "
        "number **n** and peroxide yield **%H₂O₂**, with no multi-rotation-"
        "rate fit required. When a sample has several rotation rates, its "
        "disk/ring response is also compared across them. This tab has its "
        "own file upload and does not use data loaded elsewhere."
    )

    samples, unit_hint = _orr_data_loader()
    if not samples:
        return

    labels = [lbl for lbl, _ in samples]
    chosen = st.multiselect("Samples to compare", labels, default=labels, key="orr_chosen")
    if not chosen:
        st.info("Select at least one sample to plot.")
        return
    chosen_samples = {lbl: df for lbl, df in samples if lbl in chosen}

    style = plot_style_controls("orr", show_markers=True)
    font_size = style["font_size"]
    _PALETTE_O = style["palette"]

    st.markdown("**Current unit, electrode area & collection efficiency**")
    cur1, cur2, cur3, cur4, cur5 = st.columns(5)
    convert_density = cur1.checkbox(
        "Convert to current density (÷ area)", value=True,
        key="orr_convert_density",
        help="RRDE current is usually reported as an absolute current (A, "
             "mA, µA); enable to normalize by the electrode's geometric "
             "area for a comparable current density.",
    )
    _orr_unit_default = (
        _ABS_CURRENT_UNITS.index(unit_hint) if unit_hint in _ABS_CURRENT_UNITS
        else _ABS_CURRENT_UNITS.index("A")
    )
    current_unit = cur2.selectbox(
        "Current unit as uploaded", _ABS_CURRENT_UNITS, index=_orr_unit_default,
        key="orr_current_unit",
        help="Auto-detected from the file's column header when recognisable "
             "(RRDE exports are often in A) — override if it's wrong.",
    )
    desired_unit = cur3.selectbox(
        "Desired current unit", _ABS_CURRENT_UNITS,
        index=_ABS_CURRENT_UNITS.index("mA"), key="orr_desired_unit",
        help="Values are rescaled from 'as uploaded' to this unit before "
             "any density conversion below. Defaults to mA (i.e. mA/cm² "
             "once density-converted below), the most common RRDE unit.",
    )
    area_cm2 = cur4.number_input(
        "Electrode area (cm²)", min_value=1e-4, value=0.19625, step=0.001,
        format="%.5f", key="orr_area_cm2",
        help="0.19625 cm² is the standard 5 mm-diameter RRDE glassy-carbon disk.",
    ) if convert_density else None
    display_unit = f"{desired_unit}/cm²" if convert_density else desired_unit
    # The ring is a separate electrode with its own (larger) area, which the
    # app never asks for -- the disk area above is the only one entered. So a
    # "density-converted" ring current is the ring current divided by the
    # *disk* area, which is not the ring's current density and must not be
    # labelled as though it were. n and %H2O2 are unaffected: both are ratios
    # of Ir/N to Id, so the shared divisor cancels out of them exactly.
    ring_unit = (f"{desired_unit}/cm²(disk)" if convert_density
                 else desired_unit)
    collection_efficiency = cur5.number_input(
        "Ring collection efficiency N", min_value=0.01, max_value=1.0,
        value=0.222, step=0.01, format="%.3f", key="orr_collection_efficiency",
        help="From the RRDE electrode's own calibration (e.g. a "
             "ferri/ferrocyanide test); 0.222 is this app's default. Only "
             "used where ring current is available.",
    )
    if unit_hint:
        st.caption(f"ℹ️ Detected current unit from the column header: **{unit_hint}**.")

    to_rhe_fn = _render_rhe_conversion(
        "orr", default_ph=13.0,
        default_ref_electrode="Hg/HgO, 1 M KOH", default_electrolyte="0.1 M KOH",
        default_mode="Reference electrode + electrolyte pH",
    )

    # Prepare each sample's full (all-rotation-rate) data once: RHE
    # potential, disk/ring current density, and its own set of available
    # rpm values -- reused by every plot below.
    prepared = {}
    all_rpms: set[float] = set()
    for lbl, df in chosen_samples.items():
        pot_rhe = to_rhe_fn(df["potential"].to_numpy(dtype=float))
        disk = _rescale_current(
            df["disk_current"].to_numpy(dtype=float), current_unit, desired_unit
        )
        disk = disk / area_cm2 if convert_density else disk
        has_ring = "ring_current" in df.columns
        ring = None
        if has_ring:
            ring = _rescale_current(
                df["ring_current"].to_numpy(dtype=float), current_unit, desired_unit
            )
            ring = ring / area_cm2 if convert_density else ring
        rpm_arr = df["rpm"].to_numpy(dtype=float)
        rpm_values = sorted(set(rpm_arr.tolist()))
        all_rpms.update(rpm_values)
        prepared[lbl] = {
            "potential": pot_rhe, "disk": disk, "ring": ring, "has_ring": has_ring,
            "rpm": rpm_arr, "rpm_values": rpm_values,
        }

    if not all_rpms:
        st.error("No rotation-rate data found.")
        return
    rpm_options = sorted(all_rpms)
    default_rpm = (
        1600.0 if 1600.0 in rpm_options
        else min(rpm_options, key=lambda v: abs(v - 1600.0))
    )
    primary_rpm = st.selectbox(
        "Primary rotation rate — used for the disk curve, Tafel fit, n and "
        "%H₂O₂ below",
        rpm_options, index=rpm_options.index(default_rpm),
        format_func=lambda v: f"{v:g} rpm", key="orr_primary_rpm",
    )

    # For each sample, slice out the rotation rate nearest the chosen
    # primary rpm (a sample need not have that exact value).
    slices = {}
    for lbl in chosen:
        p = prepared[lbl]
        sample_rpm = min(p["rpm_values"], key=lambda v: abs(v - primary_rpm))
        idx = np.flatnonzero(np.isclose(p["rpm"], sample_rpm))
        if len(idx) < 5:
            st.warning(f"{lbl}: fewer than 5 points at {sample_rpm:g} rpm — skipped.")
            continue
        if abs(sample_rpm - primary_rpm) > 1e-6:
            st.caption(
                f"↪ {lbl} has no {primary_rpm:g} rpm data — using its nearest "
                f"available rotation rate, {sample_rpm:g} rpm, instead."
            )
        # Clean this rotation rate's slice (drop any approach/vertex leg)
        # before onset/E1/2/Tafel read anything off it.
        _pot_c, _disk_c, _ring_c = sweep.clean_sweep(
            p["potential"][idx], p["disk"][idx],
            p["ring"][idx] if p["has_ring"] else None,
        )
        slices[lbl] = {
            "rpm": sample_rpm, "potential": _pot_c, "disk": _disk_c,
            "ring": (_ring_c if p["has_ring"] else None),
            "has_ring": p["has_ring"],
        }
    if not slices:
        return

    palette_for = {lbl: _PALETTE_O[i % len(_PALETTE_O)]
                   for i, lbl in enumerate(chosen)}
    palette_for.update(trace_color_pickers(list(chosen), style, "orr_series"))

    def _style_axes(fig, xtitle, ytitle, yrange=None):
        _journal_axes_style(fig, xtitle, ytitle, font_size, yrange=yrange,
                            style=style)

    def _zero_anchored_range(values: np.ndarray) -> list:
        """Default Y range anchored at 0, extending only in the direction
        the data actually goes (e.g. disk current runs negative/cathodic,
        ring current runs positive/anodic) — matches the standard RRDE
        convention instead of Plotly's own auto-range, which pads both
        ends around the data's min/max and generally won't include 0."""
        vmax, vmin = float(np.max(values)), float(np.min(values))
        if vmax == 0 and vmin == 0:
            # An all-zero trace would otherwise give [0, 0], which Plotly
            # renders as a zero-height axis with no ticks at all.
            return [-1.0, 1.0]
        if abs(vmax) >= abs(vmin):
            return [0.0, vmax * 1.05 if vmax > 0 else vmax * 0.95]
        return [vmin * 1.05 if vmin < 0 else vmin * 0.95, 0.0]

    # ---- Disk (& ring) polarization curve, all chosen samples overlaid ---
    has_any_ring = any(s["has_ring"] for s in slices.values())
    title = "Disk & ring polarization curves" if has_any_ring else "Disk polarization curve"
    st.markdown(f"**{title} @ ~{primary_rpm:g} rpm**")
    all_pot_disk = np.concatenate([s["potential"] for s in slices.values()])
    all_disk_cur = np.concatenate([s["disk"] for s in slices.values()])
    disk_data: dict[str, list] = {}
    ring_y_range = None
    if has_any_ring:
        # X (potential) range applies to both panels — they share one axis.
        # Disk/ring current each keep their own independent Y range control
        # below, since the two are usually very different magnitudes.
        disk_x_range, _ = _range_controls(
            "orr_disk", "Potential (V)", None,
            float(np.min(all_pot_disk)), float(np.max(all_pot_disk)),
        )
        all_ring_cur = np.concatenate([s["ring"] for s in slices.values() if s["has_ring"]])
        _, ring_y_range = _range_controls(
            "orr_ring_y", "Potential (V)", f"Ring current ({ring_unit})",
            float(np.min(all_pot_disk)), float(np.max(all_pot_disk)),
            float(np.min(all_ring_cur)), float(np.max(all_ring_cur)),
        )
        if convert_density:
            st.caption(
                "↳ Ring current is normalised by the **disk** area "
                "(the only area this app asks for), so it is not the "
                "ring's own current density — read it as a relative "
                "trace. n and %H₂O₂ are unaffected: the shared area "
                "cancels out of both ratios."
            )
        _, disk_y_range = _range_controls(
            "orr_disk_y", "Potential (V)", f"Disk current ({display_unit})",
            float(np.min(all_pot_disk)), float(np.max(all_pot_disk)),
            float(np.min(all_disk_cur)), float(np.max(all_disk_cur)),
        )
        # Ring on top, disk below — separate sub-plots (not one shared-axis
        # overlay), each with its own Y scale, merged into one figure on a
        # shared X axis (explicitly matched, not just visually adjacent) so
        # panning/zooming one keeps both in sync. Tight spacing so the two
        # panels read as one merged figure (the seam-line thickness itself
        # is fixed below by disabling just one side's mirror, not by
        # widening the gap).
        fig_disk = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.025,
            subplot_titles=("Ring", "Disk"),
        )
        for lbl, s in slices.items():
            color = palette_for[lbl]
            if s["has_ring"]:
                fig_disk.add_trace(go.Scatter(
                    x=s["potential"], y=s["ring"], mode="lines", name=lbl,
                    legendgroup=lbl, line={"color": color, "width": 3},
                ), row=1, col=1)
                disk_data[f"{lbl} — Ring current ({ring_unit})"] = list(s["ring"])
            fig_disk.add_trace(go.Scatter(
                x=s["potential"], y=s["disk"], mode="lines", name=lbl,
                legendgroup=lbl, showlegend=not s["has_ring"],
                line={"color": color, "width": 3},
            ), row=2, col=1)
            disk_data[f"{lbl} — Potential vs RHE (V)"] = list(s["potential"])
            disk_data[f"{lbl} — Disk current ({display_unit})"] = list(s["disk"])
        axis_font_disk = {"family": "Arial", "size": font_size}
        # Room for the tick numbers (which can run wide at the 36 pt export
        # size, e.g. "-10.000") plus the rotated title beyond them — a fixed
        # margin sized for the worst case, since automargin was what caused
        # the right border to clip when the legend was dragged inside the
        # plot (see note below).
        left_margin = 160 + (font_size - 28) * 4
        fig_disk.update_layout(
            template="plotly_white", height=680, width=930, font=axis_font_disk,
            margin={"l": left_margin, "r": 40},
        )
        fig_disk.update_xaxes(
            title={"text": "Potential vs RHE / V", "font": axis_font_disk},
            tickfont=axis_font_disk, row=2, col=1,
        )
        # One common Y-axis title spanning both rows (no separate "Ring"/
        # "Disk" tag on the axis itself — that distinction is already in
        # the subplot titles and legend) — added as a single rotated
        # annotation rather than per-row titles, which also keeps both
        # rows' plot areas the same width (a real per-row title can throw
        # off automargin sizing between rows and misalign the shared X axis).
        # The fixed left margin above is sized generously for the 28/36 pt
        # export font instead of using automargin — automargin recomputes
        # the plot domain on every client-side relayout (e.g. dragging the
        # legend inside the plot), which was clipping the right-hand
        # mirrored border when it fired.
        fig_disk.add_annotation(
            x=0, y=0.5, xref="paper", yref="paper",
            xanchor="center", yanchor="middle", textangle=-90,
            text=f"Current density ({display_unit})", showarrow=False,
            font=axis_font_disk, xshift=-(left_margin - 30),
        )
        fig_disk.update_yaxes(tickfont=axis_font_disk, row=1, col=1)
        fig_disk.update_yaxes(tickfont=axis_font_disk, row=2, col=1)
        fig_disk.update_xaxes(**_BOX_AXIS_STYLE)
        fig_disk.update_yaxes(**_BOX_AXIS_STYLE)
        # An x-axis line is always drawn on its own (bottom) side, and
        # _BOX_AXIS_STYLE's mirror=True adds the opposite (top) side too —
        # so at the seam there are naturally two lines: row 1's own bottom
        # line, and row 2's *mirrored* top line sitting right against it,
        # which read as one over-thick line. Turning off just row 2's
        # mirror leaves exactly one normal-thickness line at the seam
        # (row 1's), while row 1's own mirror still closes the box on the
        # outer top edge and row 2's own bottom line closes it on the
        # outer bottom edge.
        fig_disk.update_xaxes(mirror=False, row=2, col=1)
        # Explicitly force both rows' X axes to the same range together
        # (belt-and-braces on top of shared_xaxes) so they're never just
        # visually similar but a genuinely single, merged X axis.
        fig_disk.update_xaxes(matches="x")
        if disk_x_range is not None:
            fig_disk.update_xaxes(range=disk_x_range)
        final_ring_range = (
            ring_y_range if ring_y_range is not None else _zero_anchored_range(all_ring_cur)
        )
        final_disk_range = (
            disk_y_range if disk_y_range is not None else _zero_anchored_range(all_disk_cur)
        )
        fig_disk.update_yaxes(range=final_ring_range, row=1, col=1)
        fig_disk.update_yaxes(range=final_disk_range, row=2, col=1)
        # Both ranges are zero-anchored, so "0" sits right at the seam on
        # both axes and would otherwise be printed twice, one on top of the
        # other. Keep it only on the disk (bottom) axis — the one right
        # next to the shared X-axis labels — and drop it from ring's ticks.
        ring_span = abs(final_ring_range[1] - final_ring_range[0])
        ring_ticks = [
            t for t in np.linspace(final_ring_range[0], final_ring_range[1], 5)
            if not np.isclose(t, 0, atol=max(ring_span, 1e-9) * 1e-6)
        ]
        fig_disk.update_yaxes(tickvals=ring_ticks, tickformat=".2~f", row=1, col=1)
    else:
        disk_x_range, disk_y_range = _range_controls(
            "orr_disk", "Potential (V)", f"Disk current ({display_unit})",
            float(np.min(all_pot_disk)), float(np.max(all_pot_disk)),
            float(np.min(all_disk_cur)), float(np.max(all_disk_cur)),
        )
        fig_disk = go.Figure()
        for lbl, s in slices.items():
            fig_disk.add_trace(go.Scatter(
                x=s["potential"], y=s["disk"], mode="lines", name=lbl,
                line={"color": palette_for[lbl], "width": 3},
            ))
            disk_data[f"{lbl} — Potential vs RHE (V)"] = list(s["potential"])
            disk_data[f"{lbl} — Disk current ({display_unit})"] = list(s["disk"])
        _style_axes(fig_disk, "Potential vs RHE / V", f"Disk current ({display_unit})")
        if disk_x_range is not None:
            fig_disk.update_xaxes(range=disk_x_range)
        fig_disk.update_yaxes(
            range=disk_y_range if disk_y_range is not None else _zero_anchored_range(all_disk_cur)
        )
    # The merged ring/disk figure sets its own (narrower, journal-style)
    # width above — stretching it to the full container width was what
    # left too little room for the rotated axis title against the tick
    # numbers at larger export font sizes.
    st.plotly_chart(
        fig_disk, width=("content" if has_any_ring else "stretch"), config=_PLOTLY_EDIT_CONFIG
    )
    figure_downloads(
        fig_disk, f"orr_disk_curve_{int(primary_rpm)}rpm", key="png_orr_disk",
        what=title, data=_padded_frame(disk_data),
    )

    # ---- Per-sample onset / E1/2 / Tafel ----------------------------------
    st.markdown("**Onset, half-wave potential & Tafel slope**")
    st.caption(
        "Each sample gets its own Tafel fit-range slider — auto-started "
        "near the kinetic (low-overpotential) region of its mass-transport-"
        "corrected current."
    )
    # The literature-standard read-off is the half-limiting-current one, so
    # that is the default here. "Steepest" (max |dI/dE|) was the only method
    # the GUI exposed even though the analysis module implements all three;
    # on the bundled sample the two disagree by 65 mV (0.517 V vs 0.583 V),
    # which is a large discrepancy to have had no control over.
    _EHW_METHODS = {
        "j_lim/2 interpolation (literature standard)": "interpolated",
        "Steepest point, max |dI/dE|": "steepest",
        "Inflection via d²I/dE² zero-crossing": "second_derivative",
    }
    ehw0, ehw1, ehw2 = st.columns([1.6, 1, 1])
    ehw_method = _EHW_METHODS[ehw0.selectbox(
        "E½ method", list(_EHW_METHODS), index=0, key="orr_ehw_method",
        help="**j_lim/2** interpolates where the disk current crosses the "
             "midpoint between its pre-onset baseline and its plateau — the "
             "convention most published ORR papers use, and directly "
             "comparable to their numbers, but it needs a well-formed "
             "plateau. **Steepest** takes the inflection point instead and "
             "does not care whether the plateau is flat. **d²I/dE²** finds "
             "that same inflection as a zero-crossing (sub-grid precision, "
             "but more sensitive to noise).",
    )]
    ehw_lo = ehw1.number_input(
        "E½ search window min (V vs RHE)", value=0.4, step=0.05, format="%.2f",
        key="orr_ehw_lo",
        help="E1/2 is picked as the steepest point of the disk curve "
             "(max |dI/dE|) within this window — scientifically typical "
             "for ORR (~0.4-0.8 V vs RHE, non-precious-metal to good Pt-"
             "group catalysts) — rather than anywhere past onset, which can "
             "otherwise lock onto a sharp artifact right at onset itself. "
             "Widen if your catalyst's E1/2 genuinely falls outside this "
             "range.",
    )
    ehw_hi = ehw2.number_input(
        "E½ search window max (V vs RHE)", value=0.8, step=0.05, format="%.2f",
        key="orr_ehw_hi",
    )
    results_rows = []
    tafel_slider_specs = []  # (lbl, range_key, default_range, n_pts) — sliders
    # actually render further down, right above the Tafel plot they control
    # (see below); here we only need each one's *current* value (from
    # session_state, same as what the slider will show once it renders), so
    # the table below and this loop's Tafel fit stay in sync with it.
    for lbl, s in slices.items():
        try:
            onset_res = orr.onset_and_half_wave(
                s["potential"], s["disk"], half_wave_search_range=(ehw_lo, ehw_hi),
                method=ehw_method,
            )
        except ValueError as exc:
            st.warning(f"{lbl}: could not locate onset/E½ ({exc}).")
            continue
        # E1/2 is the steepest point *within the search window*. When the
        # steepest point of the real wave lies outside it -- a poor catalyst
        # whose wave sits below 0.4 V, say -- the search returns the window's
        # own edge, which looks like a plausible number and is not one.
        e_half = float(onset_res.half_wave_potential)
        if min(abs(e_half - ehw_lo), abs(e_half - ehw_hi)) <= 0.005:
            st.warning(
                f"{lbl}: E½ = {e_half:.3f} V landed on the edge of the "
                f"{ehw_lo:g}–{ehw_hi:g} V search window, so the wave's real "
                "steepest point is probably outside it. Widen the window "
                "above before quoting this value."
            )
        row = {
            "Sample": lbl, "Rotation rate (rpm)": s["rpm"],
            "E_onset (V vs RHE)": round(onset_res.onset_potential, 3),
            "E_half-wave (V vs RHE)": round(e_half, 3),
            f"j_limiting ({display_unit})": round(onset_res.limiting_current, 4),
        }

        jk = orr.mass_transport_corrected_current(s["disk"], onset_res.limiting_current)
        valid = np.isfinite(jk) & (jk != 0)
        tafel_result = None
        if valid.sum() >= 5:
            pot_tafel = s["potential"][valid]
            log_jk = tafel.log_current(jk[valid])
            a0, a1 = tafel.auto_tafel_range(
                pot_tafel, log_jk, current=jk[valid],
                e_eq=tafel.REACTION_E_EQ_V_RHE.get("ORR"),
            )
            range_key = f"orr_tafel_range_{lbl}"
            default_range = (int(a0), int(a1))
            start, stop = st.session_state.get(range_key, default_range)
            tafel_slider_specs.append((lbl, range_key, default_range, len(pot_tafel)))
            try:
                tafel_result = tafel.fit_tafel(pot_tafel, log_jk, start, stop)
            except ValueError as exc:
                st.warning(f"{lbl}: Tafel fit failed ({exc}).")
            if tafel_result is not None:
                row["Tafel slope (mV/dec)"] = round(abs(tafel_result.slope_mv_per_dec), 1)
                row["Tafel R²"] = round(tafel_result.r_squared, 4)

        if s["has_ring"]:
            n_arr = orr.electron_number(s["disk"], s["ring"], collection_efficiency)
            pct_arr = orr.peroxide_percent(s["disk"], s["ring"], collection_efficiency)
            row["n @ E½"] = round(float(sweep.interp_at(
                onset_res.half_wave_potential, s["potential"], n_arr
            )), 2)
            row["%H₂O₂ @ E½"] = round(float(sweep.interp_at(
                onset_res.half_wave_potential, s["potential"], pct_arr
            )), 1)

        results_rows.append(row)
        s["onset"] = onset_res
        s["tafel"] = tafel_result
        s["jk"] = jk
        s["jk_valid"] = valid
        s["deriv_pot"], s["deriv"] = orr.half_wave_derivative(s["potential"], s["disk"])

    onset_samples = {lbl: s for lbl, s in slices.items() if s.get("onset") is not None}
    if onset_samples:
        st.markdown("**E½ verification (dI/dE)**")
        st.caption(
            "The dashed vertical line is the E½ reported above (the "
            "steepest point, i.e. max |dI/dE|, within the search window) — "
            "it should sit at this curve's peak. Potential axis matches "
            "whatever range you set on the disk & ring plot above."
        )
        fig_deriv = go.Figure()
        deriv_data: dict[str, list] = {}
        for lbl, s in onset_samples.items():
            color = palette_for[lbl]
            fig_deriv.add_trace(go.Scatter(
                x=s["deriv_pot"], y=s["deriv"], mode="lines", name=lbl,
                line={"color": color, "width": 2.5},
            ))
            e_half = s["onset"].half_wave_potential
            # np.interp needs ascending x — deriv_pot may be descending
            # (e.g. a cathodic ORR sweep), which would otherwise silently
            # return a near-constant edge value regardless of e_half.
            order = np.argsort(s["deriv_pot"])
            deriv_at_half = float(np.interp(e_half, s["deriv_pot"][order], s["deriv"][order]))
            fig_deriv.add_trace(go.Scatter(
                x=[e_half], y=[deriv_at_half], mode="markers", showlegend=False,
                marker={"size": 11, "color": _darken(color), "symbol": "circle"},
            ))
            fig_deriv.add_vline(
                x=e_half, line={"color": _darken(color), "width": 1.5, "dash": "dash"},
            )
            deriv_data[f"{lbl} — Potential vs RHE (V)"] = list(s["deriv_pot"])
            deriv_data[f"{lbl} — dI/dE"] = list(s["deriv"])
        _style_axes(fig_deriv, "Potential vs RHE / V", f"dI/dE ({display_unit}/V)")
        if disk_x_range is not None:
            fig_deriv.update_xaxes(range=disk_x_range)
        # Y range from the 1st-99th percentile (within the visible X window)
        # rather than the strict min/max — a lone noise spike (differentiation
        # amplifies noise a lot) can otherwise dominate the auto-range and
        # squash the actual inflection peak down to where it's hard to see.
        deriv_in_view = []
        for s in onset_samples.values():
            mask = (
                (s["deriv_pot"] >= disk_x_range[0]) & (s["deriv_pot"] <= disk_x_range[1])
                if disk_x_range is not None else np.ones_like(s["deriv_pot"], dtype=bool)
            )
            if mask.any():
                deriv_in_view.append(s["deriv"][mask])
        if deriv_in_view:
            y_lo, y_hi = np.percentile(np.concatenate(deriv_in_view), [1, 99])
            pad = (y_hi - y_lo) * 0.15 or abs(y_hi) * 0.15 or 1.0
            fig_deriv.update_yaxes(range=[y_lo - pad, y_hi + pad])
        st.plotly_chart(fig_deriv, width="stretch", config=_PLOTLY_EDIT_CONFIG)
        figure_downloads(
            fig_deriv, f"orr_derivative_{int(primary_rpm)}rpm", key="png_orr_deriv",
            what="dI/dE derivative plot", data=_padded_frame(deriv_data),
        )

    if results_rows:
        summary_df = pd.DataFrame(results_rows)
        st.dataframe(summary_df, width="stretch", hide_index=True)
        st.download_button(
            "⬇️ Download ORR summary (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"orr_summary_{int(primary_rpm)}rpm.csv", mime="text/csv",
            key="dl_orr_summary",
        )
        _journal_table_figure(  # CSV of this table is the summary button above
            summary_df, font_size, f"orr_results_table_{int(primary_rpm)}rpm",
            key="png_orr_table", style=style,
        )

    if tafel_slider_specs:
        st.markdown(
            "**Tafel fit range** (per sample — feeds the mass-transport-"
            "corrected Tafel plot right below)"
        )
        for lbl, range_key, default_range, n_pts_tafel in tafel_slider_specs:
            slider_kwargs = ({} if range_key in st.session_state
                             else {"value": default_range})
            st.slider(
                f"{lbl} — Tafel fit range (index)", 0, n_pts_tafel,
                key=range_key, **slider_kwargs,
            )

    # ---- Tafel plot, all chosen samples overlaid --------------------------
    tafel_samples = {lbl: s for lbl, s in slices.items() if s.get("tafel") is not None}
    if tafel_samples:
        st.markdown("**Tafel plot (mass-transport corrected)**")
        all_logjk = np.concatenate([
            tafel.log_current(s["jk"][s["jk_valid"]]) for s in tafel_samples.values()
        ])
        all_pot_tafel = np.concatenate([
            s["potential"][s["jk_valid"]] for s in tafel_samples.values()
        ])
        orr_tafel_x_range, orr_tafel_y_range = _range_controls(
            "orr_tafel_plot", "log|j_k| (X)", "Potential vs RHE / V (Y)",
            float(np.min(all_logjk)), float(np.max(all_logjk)),
            float(np.min(all_pot_tafel)), float(np.max(all_pot_tafel)),
        )
        fig_tafel = go.Figure()
        for lbl, s in tafel_samples.items():
            color = palette_for[lbl]
            valid = s["jk_valid"]
            pot_tafel = s["potential"][valid]
            log_jk = tafel.log_current(s["jk"][valid])
            r = s["tafel"]
            start, stop = r.fit_slice
            fig_tafel.add_trace(go.Scatter(
                x=log_jk, y=pot_tafel, mode="markers", name=lbl,
                marker={"size": 8, "color": color, "opacity": 0.5},
            ))
            xs = log_jk[start:stop]
            xline = np.array([float(np.min(xs)), float(np.max(xs))])
            yline = r.slope_v_per_dec * xline + r.intercept_v
            slope_abs = abs(r.slope_mv_per_dec)
            fit_color = _darken(color)
            fig_tafel.add_trace(go.Scatter(
                x=xline, y=yline, mode="lines", showlegend=False,
                name=f"{lbl} fit, {slope_abs:.0f} mV/dec",
                line={"color": fit_color, "width": 3, "dash": "dot"},
            ))
            xmid, ymid = float(np.mean(xline)), float(np.mean(yline))
            fig_tafel.add_annotation(
                x=xmid, y=ymid, text=f"{slope_abs:.0f} mV/dec", showarrow=False,
                yshift=14,
                font={"family": "Arial", "color": fit_color, "size": round(font_size * 0.6)},
            )
        _style_axes(fig_tafel, f"log₁₀ |j_k| ({display_unit})", "Potential vs RHE / V")
        if orr_tafel_x_range is not None:
            fig_tafel.update_xaxes(range=orr_tafel_x_range)
        if orr_tafel_y_range is not None:
            fig_tafel.update_yaxes(range=orr_tafel_y_range)
        st.plotly_chart(fig_tafel, width="stretch", config=_PLOTLY_EDIT_CONFIG)
        figure_downloads(
            fig_tafel, f"orr_tafel_{int(primary_rpm)}rpm", key="png_orr_tafel",
            what="Tafel plot",
            data=_padded_frame({
                **{f"{lbl} — log10|jk|": list(tafel.log_current(s["jk"][s["jk_valid"]]))
                   for lbl, s in tafel_samples.items()},
                **{f"{lbl} — Potential vs RHE (V)": list(s["potential"][s["jk_valid"]])
                   for lbl, s in tafel_samples.items()},
            }),
        )

    # ---- n & %H2O2 vs potential, all chosen samples overlaid --------------
    ring_samples = {lbl: s for lbl, s in slices.items() if s["has_ring"]}
    if ring_samples:
        st.markdown(f"**Peroxide yield & electron number @ ~{primary_rpm:g} rpm**")

        # All data is always computed/plotted in full below — the range
        # controls only set the plotted view (fig.update_xaxes(range=...)),
        # never filter points, so widening/narrowing (or the CSV export,
        # which always holds the full curve) recovers everything.
        all_pot_ring = np.concatenate([s["potential"] for s in ring_samples.values()])
        pot_lo, pot_hi = float(np.min(all_pot_ring)), float(np.max(all_pot_ring))
        onset_vals = [
            s["onset"].onset_potential for s in ring_samples.values() if s.get("onset")
        ]
        # Default view excludes potentials past each sample's own ORR onset
        # (the flat, near-OCP region where disk current is ~0 and n/%H2O2
        # are numerically noisy rather than physically meaningful) and is
        # capped at 1 V vs RHE (O2/H2O equilibrium is 1.23 V; ring/disk
        # response above ~1 V is background/capacitive, not O2 reduction) —
        # both are just the starting view, freely widened below.
        default_hi = min([*onset_vals, 1.0, pot_hi]) if onset_vals else min(1.0, pot_hi)
        wc1, wc2 = st.columns(2)
        ho2n_pmin = wc1.number_input(
            "Potential view min (V vs RHE)", value=round(pot_lo, 3),
            step=0.02, format="%.3f", key="orr_ho2n_pmin",
        )
        ho2n_pmax = wc2.number_input(
            "Potential view max (V vs RHE)", value=round(default_hi, 3),
            step=0.02, format="%.3f", key="orr_ho2n_pmax",
        )
        st.caption(
            "↪ View range only — the underlying data (and its CSV export) "
            "always keeps every point; this just sets what's visible. "
            "Defaulted to exclude potentials past each sample's own onset — "
            "widen or narrow to taste."
        )

        fig_ho2, fig_n = go.Figure(), go.Figure()
        ho2_data, n_data = {}, {}
        for lbl, s in ring_samples.items():
            color = palette_for[lbl]
            n_arr = orr.electron_number(s["disk"], s["ring"], collection_efficiency)
            pct_arr = orr.peroxide_percent(s["disk"], s["ring"], collection_efficiency)
            fig_ho2.add_trace(go.Scatter(
                x=s["potential"], y=pct_arr, mode="lines", name=lbl,
                line={"color": color, "width": 3},
            ))
            fig_n.add_trace(go.Scatter(
                x=s["potential"], y=n_arr, mode="lines", name=lbl,
                line={"color": color, "width": 3},
            ))
            ho2_data[f"{lbl} — Potential vs RHE (V)"] = list(s["potential"])
            ho2_data[f"{lbl} — %H2O2"] = list(pct_arr)
            n_data[f"{lbl} — Potential vs RHE (V)"] = list(s["potential"])
            n_data[f"{lbl} — n"] = list(n_arr)

        _style_axes(fig_ho2, "Potential vs RHE / V", "%H₂O₂", yrange=[0, 100])
        _style_axes(fig_n, "Potential vs RHE / V", "n", yrange=[0, 4])
        fig_ho2.update_yaxes(tickvals=[0, 25, 50, 75, 100])
        if ho2n_pmax > ho2n_pmin:
            fig_ho2.update_xaxes(range=[ho2n_pmin, ho2n_pmax])
            fig_n.update_xaxes(range=[ho2n_pmin, ho2n_pmax])

        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(fig_ho2, width="stretch", config=_PLOTLY_EDIT_CONFIG)
            figure_downloads(
                fig_ho2, f"orr_ho2_{int(primary_rpm)}rpm", key="png_orr_ho2",
                what="%H₂O₂ plot", data=_padded_frame(ho2_data),
            )
        with pc2:
            st.plotly_chart(fig_n, width="stretch", config=_PLOTLY_EDIT_CONFIG)
            figure_downloads(
                fig_n, f"orr_n_{int(primary_rpm)}rpm", key="png_orr_n",
                what="n plot", data=_padded_frame(n_data),
            )

    # ---- Rotation-rate comparison, one sample, ring/disk stacked subplots -
    multi_rpm_labels = [lbl for lbl in chosen if len(prepared[lbl]["rpm_values"]) > 1]
    if multi_rpm_labels:
        st.markdown("**Rotation-rate comparison (single sample)**")
        rrde_label = st.selectbox(
            "Sample", multi_rpm_labels, key="orr_rrde_sample",
            help="Every rotation rate this sample has, ring and disk current "
                 "as two stacked sub-plots sharing one potential axis — the "
                 "same layout as the multi-sample disk & ring plot above.",
        )
        p = prepared[rrde_label]
        rpm_pick = st.multiselect(
            "Rotation rates to show", p["rpm_values"], default=p["rpm_values"],
            format_func=lambda v: f"{v:g} rpm", key=f"orr_rrde_rpms_{rrde_label}",
            help="Deselect a rotation rate to drop it from this overlay "
                 "(and its TIFF/CSV export) without affecting the rest of "
                 "the ORR analysis above.",
        )
        if not rpm_pick:
            st.info("Select at least one rotation rate to plot.")
            rpm_pick = p["rpm_values"]
        rpm_mask = np.isin(p["rpm"], rpm_pick)
        pot_sel = p["potential"][rpm_mask]
        rrde_x_range, _ = _range_controls(
            "orr_rrde", "Potential (V)", None,
            float(np.min(pot_sel)), float(np.max(pot_sel)),
        )
        axis_font_rrde = {"family": "Arial", "size": font_size}
        left_margin = 160 + (font_size - 28) * 4
        if p["has_ring"]:
            ring_sel, disk_sel = p["ring"][rpm_mask], p["disk"][rpm_mask]
            _, ring_y_range = _range_controls(
                "orr_rrde_ring_y", "Potential (V)", f"Ring current ({ring_unit})",
                float(np.min(pot_sel)), float(np.max(pot_sel)),
                float(np.min(ring_sel)), float(np.max(ring_sel)),
            )
            if convert_density:
                st.caption(
                    "↳ Ring current is normalised by the **disk** area "
                    "(the only area this app asks for), so it is not the "
                    "ring's own current density — read it as a relative "
                    "trace. n and %H₂O₂ are unaffected: the shared area "
                    "cancels out of both ratios."
                )
            _, disk_y_range = _range_controls(
                "orr_rrde_disk_y", "Potential (V)", f"Disk current ({display_unit})",
                float(np.min(pot_sel)), float(np.max(pot_sel)),
                float(np.min(disk_sel)), float(np.max(disk_sel)),
            )
            # Ring on top, disk below — same merged-stacked-subplot recipe as
            # the multi-sample disk & ring plot above (see its comments for
            # why: shared/matched X axis, single rotated Y title spanning
            # both rows, one-sided mirror at the seam, de-duplicated "0" tick).
            fig_rrde = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.025,
                subplot_titles=("Ring", "Disk"),
            )
            for i, rv in enumerate(rpm_pick):
                m = np.isclose(p["rpm"], rv)
                color = _PALETTE_O[i % len(_PALETTE_O)]
                fig_rrde.add_trace(go.Scatter(
                    x=p["potential"][m], y=p["ring"][m], mode="lines",
                    name=f"{rv:g} rpm", legendgroup=f"{rv:g}",
                    line={"color": color, "width": 2.5},
                ), row=1, col=1)
                fig_rrde.add_trace(go.Scatter(
                    x=p["potential"][m], y=p["disk"][m], mode="lines",
                    name=f"{rv:g} rpm", legendgroup=f"{rv:g}", showlegend=False,
                    line={"color": color, "width": 2.5},
                ), row=2, col=1)
            fig_rrde.update_layout(
                template="plotly_white", height=680, width=930,
                font=axis_font_rrde, margin={"l": left_margin, "r": 40},
            )
            fig_rrde.update_xaxes(
                title={"text": "Potential vs RHE / V", "font": axis_font_rrde},
                tickfont=axis_font_rrde, row=2, col=1,
            )
            fig_rrde.add_annotation(
                x=0, y=0.5, xref="paper", yref="paper",
                xanchor="center", yanchor="middle", textangle=-90,
                text=f"Current ({display_unit})", showarrow=False,
                font=axis_font_rrde, xshift=-(left_margin - 30),
            )
            fig_rrde.update_yaxes(tickfont=axis_font_rrde, row=1, col=1)
            fig_rrde.update_yaxes(tickfont=axis_font_rrde, row=2, col=1)
            fig_rrde.update_xaxes(**_BOX_AXIS_STYLE)
            fig_rrde.update_yaxes(**_BOX_AXIS_STYLE)
            fig_rrde.update_xaxes(mirror=False, row=2, col=1)
            fig_rrde.update_xaxes(matches="x")
            if rrde_x_range is not None:
                fig_rrde.update_xaxes(range=rrde_x_range)
            final_ring_range = (
                ring_y_range if ring_y_range is not None
                else _zero_anchored_range(ring_sel)
            )
            final_disk_range = (
                disk_y_range if disk_y_range is not None
                else _zero_anchored_range(disk_sel)
            )
            fig_rrde.update_yaxes(range=final_ring_range, row=1, col=1)
            fig_rrde.update_yaxes(range=final_disk_range, row=2, col=1)
            ring_span = abs(final_ring_range[1] - final_ring_range[0])
            ring_ticks = [
                t for t in np.linspace(final_ring_range[0], final_ring_range[1], 5)
                if not np.isclose(t, 0, atol=max(ring_span, 1e-9) * 1e-6)
            ]
            fig_rrde.update_yaxes(tickvals=ring_ticks, tickformat=".2~f", row=1, col=1)
        else:
            disk_sel = p["disk"][rpm_mask]
            _, disk_y_range = _range_controls(
                "orr_rrde_disk_y", "Potential (V)", f"Disk current ({display_unit})",
                float(np.min(pot_sel)), float(np.max(pot_sel)),
                float(np.min(disk_sel)), float(np.max(disk_sel)),
            )
            fig_rrde = go.Figure()
            for i, rv in enumerate(rpm_pick):
                m = np.isclose(p["rpm"], rv)
                fig_rrde.add_trace(go.Scatter(
                    x=p["potential"][m], y=p["disk"][m], mode="lines",
                    name=f"{rv:g} rpm",
                    line={"color": _PALETTE_O[i % len(_PALETTE_O)],
                          "width": style["line_width"]},
                ))
            _style_axes(fig_rrde, "Potential vs RHE / V", f"Disk current ({display_unit})")
            if rrde_x_range is not None:
                fig_rrde.update_xaxes(range=rrde_x_range)
            fig_rrde.update_yaxes(
                range=disk_y_range if disk_y_range is not None
                else _zero_anchored_range(disk_sel)
            )
        st.plotly_chart(
            fig_rrde, width=("content" if p["has_ring"] else "stretch"), config=_PLOTLY_EDIT_CONFIG
        )
        rrde_cols: dict[str, list] = {}
        for rv in rpm_pick:
            m = np.isclose(p["rpm"], rv)
            rrde_cols[f"{rv:g}rpm — Potential vs RHE (V)"] = list(p["potential"][m])
            rrde_cols[f"{rv:g}rpm — Disk current ({display_unit})"] = list(p["disk"][m])
            if p["has_ring"]:
                rrde_cols[f"{rv:g}rpm — Ring current ({ring_unit})"] = list(p["ring"][m])
        figure_downloads(
            fig_rrde, f"orr_rrde_multirpm_{rrde_label}", key="png_orr_rrde",
            what="RRDE multi-rpm plot", data=_padded_frame(rrde_cols),
        )


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    if not require_access():
        st.stop()

    st.title(f"⚡ {APP_NAME}")
    st.caption(
        "EIS fitting → Ru extraction → LSV ohmic-drop correction, plus "
        "independent Tafel-slope analysis"
    )

    _clear_reset_control()

    tab_eis, tab_lsv, tab_tafel, tab_kl, tab_orr = st.tabs(
        ["📈 EIS / Ru Analysis", "🔬 LSV iR Correction", "📐 LSV Analysis",
         "📉 K-L Analysis", "⚛️ ORR / RRDE Analysis"]
    )

    with tab_eis:
        eis_list, eis_name = eis_data_loader()
    with tab_lsv:
        lsv_list, lsv_name = lsv_data_loader()

    eis_d = lsv_d = None
    sel, n_pairs = 0, 0
    current_unit, ru_unit, area_cm2 = "mA", "Ω", None
    if eis_list or lsv_list:
        # Link EIS dataset i with LSV dataset i by position — each tab has
        # its own uploader, but they still share one "active sample" so the
        # Ru fitted from EIS pair i is the one applied to LSV pair i. The
        # selector itself lives on the EIS/Ru tab; the LSV tab shows a
        # read-only caption pointing back to it (a Streamlit widget can't be
        # rendered twice with the same identity to keep two copies in sync).
        n_eis, n_lsv = len(eis_list or []), len(lsv_list or [])
        n_pairs = min(n_eis, n_lsv) if (eis_list and lsv_list) else max(n_eis, n_lsv)
        with tab_eis:
            st.divider()
            st.markdown(
                "**Sample** — linked with the LSV iR Correction tab by position"
            )
            if eis_list and lsv_list and n_eis != n_lsv:
                st.warning(
                    f"EIS has {n_eis} dataset(s) but LSV has {n_lsv}. Pairing "
                    f"the first {n_pairs} by position."
                )
            sel = st.selectbox(
                "Active sample (EIS ↔ LSV pair)",
                list(range(n_pairs)),
                format_func=lambda i: f"Sample {i + 1}",
                help="Also selects which LSV dataset is corrected on the "
                     "LSV iR Correction tab.",
            )
        eis_d = eis_list[sel] if eis_list else None
        lsv_d = lsv_list[sel] if lsv_list else None

        cur_default, ru_default = _detect_units(
            eis_d.label if eis_d is not None else "",
            lsv_d.label if lsv_d is not None else "",
        )
        current_unit, ru_unit, area_cm2 = sidebar_units(cur_default, ru_default)

    ru = None
    with tab_eis:
        if eis_d is not None:
            # Returns the raw Ru in the EIS file's unit; the LSV tab reconciles it
            # with the electrode area to report both Ru and Ru effective.
            ru = render_eis_tab(eis_d, eis_list, sel, ru_unit, current_unit, area_cm2)
        else:
            st.info("⬆️ Upload EIS data above to begin, or enter Ru directly below.")
        manual_ru = render_manual_ru_section(ru_unit, area_cm2)
        if manual_ru is not None:
            ru = manual_ru
    with tab_lsv:
        if lsv_d is not None:
            if eis_d is not None:
                st.caption(
                    f"Linked to **Sample {sel + 1}** of {n_pairs} (change it "
                    f"on the EIS/Ru Analysis tab) · EIS {len(eis_d)} pts · "
                    f"LSV {len(lsv_d)} pts."
                )
            else:
                st.caption(
                    f"Sample {sel + 1} of {n_pairs} selected on the EIS/Ru "
                    "Analysis tab — load EIS data there to compute Ru."
                )
            render_lsv_tab(lsv_d, ru, current_unit, ru_unit, area_cm2,
                           sample_label=f"Sample {sel + 1}")
        else:
            st.info("⬆️ Upload LSV data above to begin.")

    with tab_tafel:
        render_tafel_tab()

    with tab_kl:
        render_kl_tab()

    with tab_orr:
        render_orr_tab()

    render_citation()
    st.divider()
    st.caption(
        f"{APP_NAME} v1.1.0 · please "
        f"cite if used ([repository]({REPO_URL})). See **Cite this app** in "
        "the sidebar."
    )


if __name__ == "__main__":
    main()
