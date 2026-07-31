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
from plotly.subplots import make_subplots
from PIL import Image
import streamlit as st

from scripts.modules import correction, data_io, eis, orr, tafel

APP_NAME = "ElectroSim-LSV-RRDE-Analyzer"

st.set_page_config(
    page_title=APP_NAME,
    page_icon="⚡", layout="wide",
)

SAMPLE_PATH = "sample-data/Book1-original data.xlsx"
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

    Used to detect that a cached TIFF no longer matches the figure on screen
    (e.g. the user moved a fit-range slider after preparing the export), so a
    stale image is never handed out as a download.
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

    Defaults to the figure's own layout height when it sets one — a table
    sizes itself by row count, so forcing the same fixed height on every
    figure (as this used to) simply cut the last rows off.
    """
    if width is None:
        width = 1100
    if height is None:
        height = int(getattr(fig.layout, "height", None) or 520)
    return int(width), max(int(height), 200)


def _render_tiff(export_fig, width: int, height: int) -> bytes:
    """Render a figure to a 300 dpi, LZW-compressed TIFF — the raster format
    most journals require for figure submission. Kaleido itself only
    produces PNG/JPEG/WebP/SVG/PDF, so this renders a high-resolution PNG
    first (scale=4, i.e. 4x the CSS pixel size) and converts it in-memory."""
    png_bytes = export_fig.to_image(format="png", width=width, height=height,
                                    scale=4)
    image = Image.open(io.BytesIO(png_bytes))
    buffer = io.BytesIO()
    image.save(buffer, format="TIFF", dpi=(300, 300), compression="tiff_lzw")
    return buffer.getvalue()


def figure_downloads(fig, stem: str, key: str, what: str = "figure",
                     width: int | None = None, height: int | None = None,
                     data: "pd.DataFrame | None" = None) -> None:
    """Render the download controls for one figure: TIFF, interactive HTML
    and (optionally) the plotted data as CSV.

    TIFF rendering is server-side (kaleido + Pillow), which launches a
    headless browser per call — too slow/fragile to run on *every* script
    rerun (it would fire on every unrelated widget interaction, e.g.
    dragging a slider). It only happens when the user clicks "Prepare"; the
    bytes are cached in session_state together with a signature of the
    figure, so the download button persists across reruns but is withdrawn
    as soon as the figure itself changes. The HTML and CSV exports need no
    external renderer and are therefore always available — they are the
    fallback when kaleido/Chrome is unavailable on the host (e.g. a slim
    cloud container).
    """
    state_key = f"_export_{key}"
    export_fig = _export_figure(fig)
    sig = _figure_signature(export_fig)
    cols = st.columns(3 if data is not None else 2)

    with cols[0]:
        if st.button(f"🖼️ Prepare {what} (TIFF)", key=f"_tiff_prep_{key}",
                     use_container_width=True):
            w, h = _export_size(export_fig, width, height)
            try:
                st.session_state[state_key] = {
                    "sig": sig, "bytes": _render_tiff(export_fig, w, h),
                    "error": None,
                }
            except Exception as exc:  # kaleido missing / no Chrome / render error
                st.session_state[state_key] = {
                    "sig": sig, "bytes": None, "error": str(exc),
                }
        cached = st.session_state.get(state_key) or {}
        if cached.get("bytes") and cached.get("sig") == sig:
            st.download_button(
                f"⬇️ {what} (TIFF)", data=cached["bytes"],
                file_name=f"{stem}.tiff", mime="image/tiff", key=f"_tiff_dl_{key}",
                use_container_width=True,
            )
        elif cached.get("bytes"):
            st.caption("↻ Figure changed — press Prepare again.")
        elif cached.get("error"):
            st.caption(
                f"TIFF export unavailable ({cached['error']}). Use the HTML "
                "download beside this button, or the 📷 icon on the chart."
            )

    with cols[1]:
        try:
            html = export_fig.to_html(include_plotlyjs="cdn", full_html=True)
            st.download_button(
                f"⬇️ {what} (HTML)", data=html.encode("utf-8"),
                file_name=f"{stem}.html", mime="text/html",
                key=f"_html_dl_{key}", use_container_width=True,
                help="Interactive page — opens in any browser (needs internet "
                     "the first time, it loads plotly.js from a CDN) and can "
                     "be saved as an image from there. Always available, even "
                     "when the server-side TIFF renderer is not.",
            )
        except Exception as exc:
            st.caption(f"HTML export unavailable ({exc}).")

    if data is not None:
        with cols[2]:
            st.download_button(
                "⬇️ Plotted data (CSV)",
                data=data.to_csv(index=False).encode("utf-8"),
                file_name=f"{stem}_data.csv", mime="text/csv",
                key=f"_csv_dl_{key}", use_container_width=True,
                help="Exactly the series drawn above, for replotting in "
                     "Origin/Excel.",
            )


def _padded_frame(columns: "dict[str, list]") -> pd.DataFrame:
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
def sidebar_data_loader():
    """Render the data-source controls.

    Returns ``(eis_list, lsv_list, label)`` where each list holds one or more
    datasets parsed from repeated column pairs in the sheet/file.
    """
    st.sidebar.header("1 · Data source")

    # Clear / reset: bump a nonce so the file_uploader widgets are recreated
    # empty, and wipe loaded state. Uploads live only in this session's memory.
    if "uploader_nonce" not in st.session_state:
        st.session_state.uploader_nonce = 0
    if st.sidebar.button("🗑️ Clear / reset files",
                         help="Remove uploaded files and start over."):
        st.session_state.uploader_nonce += 1
        for k in list(st.session_state.keys()):
            if k not in ("uploader_nonce", "authed"):
                del st.session_state[k]
        st.rerun()
    nonce = st.session_state.uploader_nonce

    # Upload is the default; the bundled sample is opt-in (not preloaded).
    source = st.sidebar.radio(
        "Choose input",
        ["Upload Excel workbook", "Upload two CSV files", "Use bundled sample"],
        help="Excel: EIS sheet = Z', Z'' pairs; LSV sheet = Potential, Current "
             "pairs. Several datasets may sit side-by-side as repeated pairs.",
    )

    try:
        if source in ("Use bundled sample", "Upload Excel workbook"):
            if source == "Use bundled sample":
                src = SAMPLE_PATH
                name = "sample-data/Book1.xlsx"
            else:
                up = st.sidebar.file_uploader(
                    "Excel (.xlsx)", type=["xlsx", "xls"],
                    key=f"xlsx_{nonce}",
                )
                if up is None:
                    st.info("⬅️ Upload an Excel workbook to begin.")
                    return None, None, None
                src = io.BytesIO(up.read())
                name = up.name
            sheets = data_io.list_sheets(src)
            if hasattr(src, "seek"):
                src.seek(0)
            eis_sheet = st.sidebar.selectbox("EIS sheet", sheets, index=0)
            lsv_sheet = st.sidebar.selectbox(
                "LSV sheet", sheets, index=min(1, len(sheets) - 1)
            )
            if hasattr(src, "seek"):
                src.seek(0)
            eis_list = data_io.load_eis_datasets(src, sheet=eis_sheet)
            if hasattr(src, "seek"):
                src.seek(0)
            lsv_list = data_io.load_lsv_datasets(src, sheet=lsv_sheet)
            return eis_list, lsv_list, name

        # Two CSV files
        eis_up = st.sidebar.file_uploader(
            "EIS CSV (Z', Z'')", type=["csv", "txt"], key=f"eis_csv_{nonce}"
        )
        lsv_up = st.sidebar.file_uploader(
            "LSV CSV (Potential, Current)", type=["csv", "txt"],
            key=f"lsv_csv_{nonce}",
        )
        if eis_up is None or lsv_up is None:
            st.info("⬅️ Upload both CSV files to begin.")
            return None, None, None
        eis_list = data_io.load_eis_datasets(
            io.BytesIO(eis_up.read()), sheet=None
        )
        lsv_list = data_io.load_lsv_datasets(
            io.BytesIO(lsv_up.read()), sheet=None
        )
        return eis_list, lsv_list, f"{eis_up.name} + {lsv_up.name}"

    except Exception as exc:  # surface loader errors instead of a stack trace
        st.sidebar.error(f"Could not load data: {exc}")
        return None, None, None


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
    st.sidebar.header("3 · Units & electrode area")
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
    font_size = st.selectbox(
        "Figure/table export font size (pt)", _JOURNAL_FONT_SIZES, index=0,
        key="eis_font_size",
        help="Font is fixed to Arial for publication-style export.",
    )

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
                marker=dict(size=7, color="#1f77b4"),
            )
        )
        # Highlight the points selected for the fit.
        fig.add_trace(
            go.Scatter(
                x=zr[start:stop],
                y=np.abs(zi)[start:stop],
                mode="markers",
                name="Fitted arc",
                marker=dict(size=10, color="#ff7f0e", symbol="circle-open", line=dict(width=2)),
            )
        )
        if ru_result is not None and ru_result.center is not None:
            cx, cy = eis.circle_path(ru_result.center, ru_result.radius)
            fig.add_trace(
                go.Scatter(x=cx, y=cy, mode="lines", name="Fitted circle",
                           line=dict(color="#2ca02c", dash="dash"))
            )
        ru_for_marker = ru_result.ru if ru_result else manual_ru_val
        if ru_for_marker is not None:
            fig.add_trace(
                go.Scatter(
                    x=[ru_for_marker], y=[0], mode="markers+text",
                    name="Ru", text=[f"Ru={ru_for_marker:.2f} {disp_unit}"],
                    textposition="top center",
                    marker=dict(size=13, color="red", symbol="x"),
                )
            )
        _journal_axes_style(fig, f"Z′ / {disp_unit}", f"−Z″ / {disp_unit}", font_size)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal aspect -> true circle
        st.plotly_chart(fig, use_container_width=True)
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
            st.dataframe(batch_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Download batch Ru table (CSV)",
                data=batch_df.to_csv(index=False).encode("utf-8"),
                file_name="eis_ru_batch.csv", mime="text/csv",
                key="dl_eis_batch",
            )
            _journal_table_figure(
                batch_df, font_size, "eis_ru_batch_table", key="png_eis_table",
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
                   ru_unit: str = "Ω", area_cm2: float | None = None):
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
                "(Set units in the sidebar · section 3.)"
            )
        else:
            st.caption(
                f"Current is **{current_unit}**, Ru in **{ru_unit}** — units "
                "consistent. (Set units in the sidebar · section 3.)"
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
        font_size = st.selectbox(
            "Figure/table export font size (pt)", _JOURNAL_FONT_SIZES, index=0,
            key="lsv_font_size",
            help="Font is fixed to Arial for publication-style export.",
        )
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
        x_range = y_range = None
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
        if view == "Overlay (same axes)":
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=lsv_d.potential, y=lsv_d.current, mode="lines",
                           name="Without iR comp (raw)",
                           line=dict(color="#1f77b4", width=3))
            )
            for i, r in enumerate(results):
                fig.add_trace(
                    go.Scatter(
                        x=r.potential_corrected, y=lsv_d.current, mode="lines",
                        name=f"With iR comp {int(r.factor_percent)}%",
                        line=dict(color=_PALETTE[i % len(_PALETTE)]),
                    )
                )
            fig.update_layout(
                title="LSV — with vs without iR compensation",
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
                           line=dict(color="#1f77b4", width=3)),
                row=1, col=1,
            )
            for i, r in enumerate(results):
                fig.add_trace(
                    go.Scatter(
                        x=r.potential_corrected, y=lsv_d.current, mode="lines",
                        name=f"With iR comp {int(r.factor_percent)}%",
                        line=dict(color=_PALETTE[i % len(_PALETTE)]),
                    ),
                    row=1, col=2,
                )
            fig.update_xaxes(title_text="Potential / V", row=1, col=1)
            fig.update_xaxes(title_text="Potential / V", row=1, col=2)
            fig.update_yaxes(
                title_text=f"Current / {current_unit}", row=1, col=1
            )
            fig.update_layout(title="LSV — with vs without iR compensation")

        # Apply manual axis ranges (affects both the on-screen plot and TIFF).
        if x_range is not None:
            fig.update_xaxes(range=x_range)
        if y_range is not None:
            fig.update_yaxes(range=y_range)

        # Legend at the bottom so it never overlaps the title / subplot titles.
        # Journal style (Arial, box-border axes) applied globally, which
        # works for both the single-axes overlay and the two-panel
        # side-by-side view (update_xaxes/update_yaxes with no row/col
        # target every axis in the figure).
        fig.update_layout(
            template="plotly_white",
            height=470,
            font=dict(family="Arial", size=font_size),
            title=dict(y=0.97, yanchor="top"),
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="center", x=0.5,
                        font=dict(family="Arial", size=max(11, round(font_size * 0.55))),
                        bgcolor="rgba(255,255,255,0.7)", bordercolor="black",
                        borderwidth=1),
            margin=dict(l=10, r=10, t=60, b=90),
        )
        fig.update_xaxes(**_BOX_AXIS_STYLE)
        fig.update_yaxes(**_BOX_AXIS_STYLE)
        st.plotly_chart(fig, use_container_width=True)
        fac_png = "-".join(str(int(r.factor_percent)) for r in results)
        plotted = {
            "Potential raw (V)": lsv_d.potential,
            f"Current raw ({current_unit})": lsv_d.current,
        }
        for r in results:
            pct = int(r.factor_percent)
            plotted[f"Potential iR-corrected {pct}% (V)"] = r.potential_corrected
            plotted[f"Current iR-corrected {pct}% ({current_unit})"] = lsv_d.current
        figure_downloads(
            fig, f"lsv_iR_comparison_f{fac_png}pct", key="png_lsv",
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
        summary, use_container_width=True, hide_index=True,
        column_config={
            c: st.column_config.NumberColumn(format=fmt)
            for c, fmt in numeric_fmt.items()
        },
    )
    st.download_button(
        "⬇️ Download results table (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name=f"ir_results_summary_Ru{ru:.1f}.csv",
        mime="text/csv",
        key="dl_summary",
    )
    display = summary.copy()
    for c, fmt in numeric_fmt.items():
        display[c] = [fmt % v for v in summary[c]]
    _journal_table_figure(  # CSV of this table is the results button above
        display, font_size, f"ir_results_table_Ru{ru:.1f}", key="png_lsv_table",
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
_JOURNAL_FONT_SIZES = [28, 36]
_BOX_AXIS_STYLE = dict(
    showgrid=False, zeroline=False,
    showline=True, linewidth=1.5, linecolor="black", mirror=True,
    ticks="outside",
)


def _journal_axes_style(fig, xtitle: str, ytitle: str, font_size: int,
                        height: int = 460, yrange: list | None = None,
                        legend_position: str = "top-left") -> None:
    """Apply the shared journal look (Arial, box-border axes, a legend box)
    to ``fig`` in place: ``plotly_white`` template, closed axis border, and a
    bordered legend positioned inside the plot area."""
    axis_font = dict(family="Arial", size=font_size)
    small_font = dict(family="Arial", size=max(11, round(font_size * 0.55)))
    positions = {
        "top-left": dict(x=0.02, y=0.98, xanchor="left", yanchor="top"),
        "bottom-left": dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
    }
    fig.update_layout(
        template="plotly_white", height=height, font=axis_font,
        legend=dict(**positions[legend_position], font=small_font,
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="black",
                    borderwidth=1),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    fig.update_xaxes(
        title=dict(text=xtitle, font=axis_font), tickfont=axis_font,
        **_BOX_AXIS_STYLE,
    )
    fig.update_yaxes(
        title=dict(text=ytitle, font=axis_font), tickfont=axis_font,
        range=yrange, **_BOX_AXIS_STYLE,
    )


def _journal_table_figure(display_df: pd.DataFrame, font_size: int, stem: str,
                          key: str, what: str = "Results table") -> None:
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
    table_width = int(sum(widths) * char_px + 60)
    # Worst-case wrapped lines in any cell of a row -> uniform row height.
    max_wrap = max(
        (int(np.ceil(len(v) / w)) for col, w in zip(display_df.columns, widths)
         for v in str_cols[col]),
        default=1,
    )
    row_h = int(cell_font * 1.5 * max(1, max_wrap)) + 8
    table_fig = go.Figure(data=[go.Table(
        columnwidth=widths,
        header=dict(values=list(display_df.columns),
                    font=dict(family="Arial", size=header_font),
                    align="left", height=int(header_font * 1.6) + 10),
        cells=dict(values=[str_cols[c] for c in display_df.columns],
                   font=dict(family="Arial", size=cell_font), align="left",
                   height=row_h),
    )])
    table_height = int(header_font * 1.6) + 20 + row_h * len(display_df) + 20
    table_fig.update_layout(
        template="plotly_white",  # never export with Streamlit's placeholder colours
        margin=dict(l=10, r=10, t=10, b=10),
        width=table_width, height=table_height,
    )
    figure_downloads(
        table_fig, stem, key=key, what=what,
        width=table_width, height=table_height,
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
    trailing ``" (2)"``/``" (3)``/… — the exact suffix the data loader's own
    de-duplication adds when the same base file/column name is uploaded more
    than once (see ``_dedup`` in ``_tafel_data_loader``), which is exactly
    the common case of uploading several repeat scans of one sample."""
    stripped = _REPLICATE_SUFFIX.sub("", label).strip()
    return stripped or label


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


def _tafel_data_loader() -> list[tuple[str, "data_io.LSVData"]]:
    """File uploader local to the Tafel tab — independent of the main sidebar
    EIS/LSV loader. Multiple files (different samples/lots) may be uploaded
    at once; each becomes one or more labelled series so several samples can
    be combined into one journal-style overlay. Returns a flat list of
    ``(label, LSVData)``."""
    st.markdown(
        "**Data source** (independent of the EIS/LSV loader above) — upload "
        "one or more files, one per sample; they can be combined into a "
        "single overlay plot below."
    )
    source = st.radio(
        "Choose input",
        ["Upload Excel workbook(s)", "Upload CSV file(s)"],
        horizontal=True,
        key="tafel_source",
        help="Columns: Potential, Current (or current density). Several "
             "datasets may also sit side-by-side within one file as "
             "repeated column pairs.",
    )

    def _dedup(label: str, seen: dict[str, int]) -> str:
        seen[label] = seen.get(label, 0) + 1
        return label if seen[label] == 1 else f"{label} ({seen[label]})"

    series: list[tuple[str, "data_io.LSVData"]] = []
    seen_labels: dict[str, int] = {}
    try:
        if source == "Upload Excel workbook(s)":
            ups = st.file_uploader(
                "Excel (.xlsx)", type=["xlsx", "xls"], key="tafel_xlsx",
                accept_multiple_files=True,
            )
            if not ups:
                st.info("⬆️ Upload one or more workbooks to begin.")
                return []
            first_sheets = data_io.list_sheets(io.BytesIO(ups[0].getvalue()))
            sheet = st.selectbox(
                "Sheet (applied to every uploaded workbook)",
                first_sheets, index=0, key="tafel_sheet",
            )
            for up in ups:
                base = up.name.rsplit(".", 1)[0]
                try:
                    sheets_here = data_io.list_sheets(io.BytesIO(up.getvalue()))
                    use_sheet = sheet if sheet in sheets_here else sheets_here[0]
                    ds = data_io.load_lsv_datasets(
                        io.BytesIO(up.getvalue()), sheet=use_sheet
                    )
                except Exception as exc:
                    st.warning(f"{up.name}: {exc}")
                    continue
                for d in ds:
                    label = f"{base} · {d.label}" if len(ds) > 1 else base
                    series.append((_dedup(label, seen_labels), d))
            return series

        ups = st.file_uploader(
            "CSV (Potential, Current)", type=["csv", "txt"], key="tafel_csv",
            accept_multiple_files=True,
        )
        if not ups:
            st.info("⬆️ Upload one or more CSV files to begin.")
            return []
        for up in ups:
            base = up.name.rsplit(".", 1)[0]
            try:
                ds = data_io.load_lsv_datasets(io.BytesIO(up.getvalue()), sheet=None)
            except Exception as exc:
                st.warning(f"{up.name}: {exc}")
                continue
            for d in ds:
                label = f"{base} · {d.label}" if len(ds) > 1 else base
                series.append((_dedup(label, seen_labels), d))
        return series
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return []


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


def _render_rhe_conversion(key_prefix: str, default_ph: float = 13.0):
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
    mode = st.radio(
        "How should potentials be converted to the RHE scale?",
        ["Already vs RHE", "Reference electrode + electrolyte pH",
         "Direct calibration offset"],
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
                 "KOH — replace with your own electrode's measured value.",
        )
        return lambda pot: np.asarray(pot, dtype=float) + offset

    rc1, rc2 = st.columns([2, 1])
    ref_names = list(tafel.REFERENCE_ELECTRODES) + ["Custom"]
    ref_choice = rc1.selectbox(
        "Reference electrode used for the input data", ref_names,
        index=0, key=f"{key_prefix}_ref_electrode",
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
    default_electrolyte_idx = 0
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
        f"↪ E(RHE) = E(measured) + {e_ref:.3f} V + 0.0592 × {ph:g} = "
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
    default_reaction = top1.selectbox(
        "Default reaction (for newly added samples)", _TAFEL_REACTIONS, index=0,
        help="Each sample gets its own reaction below (samples of different "
             "reactions can share one overlay); this just sets the starting "
             "value for samples you haven't set yet. The legend and "
             "analysis are split by reaction type.",
    )
    font_size = top2.selectbox(
        "Figure/table export font size (pt)", _JOURNAL_FONT_SIZES, index=0,
        help="Font is fixed to Arial for publication-style export.",
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
            target_js.append(float(tok))
        except ValueError:
            st.warning(f"Ignoring unparseable target current density {tok!r}.")
    if not convert_density and not correction.is_density_unit(current_unit):
        st.caption(
            "↪ Current is not a density — these benchmarks compare across "
            "samples most meaningfully when normalized by electrode area "
            "('Convert ... to current density' above)."
        )

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
            pot = to_rhe_fn(d.potential[mask])
            cur_scaled = _rescale_current(d.current[mask], native_unit, current_unit)
            cur = cur_scaled / area_cm2 if convert_density else cur_scaled
            log_i = tafel.log_current(cur)
            n = len(pot)
            a0, a1 = tafel.auto_tafel_range(pot, log_i, current=cur)
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
                "Legend name", value=lbl, key=f"tafel_name_{i}",
                help="Shown in the plot legend; edit if the auto-detected "
                     "name isn't the one you want.",
            )
            replicate_group = cg.text_input(
                "Replicate group", value=_default_replicate_group(lbl),
                key=f"tafel_group_{i}",
                help="Give two or more samples the same group name to treat "
                     "them as repeat scans of the same underlying sample — "
                     "their fitted values (Tafel slope, onset, η@j, …) are "
                     "then also reported as a mean ± SD in the **Replicate "
                     "statistics** section below.",
            )
            sample_reaction = cr.selectbox(
                "Reaction", _TAFEL_REACTIONS,
                index=_TAFEL_REACTIONS.index(default_reaction),
                key=f"tafel_reaction_{i}",
                help="This sample's legend and analysis are grouped under "
                     "this reaction.",
            )
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
                "Color", value=_PALETTE[i % len(_PALETTE)], key=f"tafel_color_{i}"
            )
            fits.append(dict(label=display_name or lbl, orig_label=lbl,
                             reaction=sample_reaction,
                             replicate_group=replicate_group or (display_name or lbl),
                             potential=pot, current=cur, log_i=log_i,
                             start=start, stop=stop, color=color))

    if not fits:
        st.error("No usable samples (all-zero current).")
        return

    # Group samples by reaction (stable) so each reaction's traces sit
    # together in plot/legend order, enabling a split legend per reaction.
    fits.sort(key=lambda f: _TAFEL_REACTIONS.index(f["reaction"]))

    axis_font = dict(family="Arial", size=font_size)
    small_font = dict(family="Arial", size=max(12, round(font_size * 0.5)))

    # Original LSV (linear-scale polarization curve), before the log-current
    # Tafel transform — shown for context alongside the derived Tafel plot.
    st.markdown("**Original LSV (polarization curve)**")
    d_by_label = {lbl: d for lbl, d in chosen_series}
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
            line=dict(color=meta["color"], width=3),
        ))
    lsv_fig.update_layout(
        title=dict(text="Original LSV", font=dict(family="Arial", size=font_size)),
        template="plotly_white", height=460, font=axis_font,
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", font=small_font,
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1,
                    tracegroupgap=18),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    lsv_fig.update_xaxes(
        title=dict(text="Potential vs RHE / V", font=dict(family="Arial", size=font_size)),
        tickfont=dict(family="Arial", size=font_size), **_BOX_AXIS_STYLE,
    )
    lsv_fig.update_yaxes(
        title=dict(text=f"Current ({display_unit})", font=dict(family="Arial", size=font_size)),
        tickfont=dict(family="Arial", size=font_size), **_BOX_AXIS_STYLE,
    )
    st.plotly_chart(
        lsv_fig, use_container_width=True,
        config={"edits": {"legendPosition": True}, "displaylogo": False},
    )
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
    st.caption(
        f"Showing only the data within {vicinity_pct:g}% of each sample's "
        "fit-region width around its selected linear range (adjust above "
        "or widen the fit range in the expander if the plot looks too tight)."
    )
    distinct_reactions = []
    for f, _ in results:
        if f["reaction"] not in distinct_reactions:
            distinct_reactions.append(f["reaction"])
    multi_reaction = len(distinct_reactions) > 1

    fig = go.Figure()
    tafel_plotted: dict[str, list] = {}
    for f, r in results:
        color = f["color"]
        start, stop = f["start"], f["stop"]
        n_pts = len(f["log_i"])
        margin = int(round((stop - start) * vicinity_pct / 100.0))
        v0, v1 = max(0, start - margin), min(n_pts, stop + margin)
        grp = f["reaction"]
        # Same slices that are drawn, so the CSV reproduces the plot exactly:
        # the shown points, then the two endpoints of the fitted line.
        tafel_plotted[f"{f['label']} — log10|i| ({display_unit})"] = \
            list(f["log_i"][v0:v1])
        tafel_plotted[f"{f['label']} — E vs RHE (V)"] = \
            list(f["potential"][v0:v1])
        fig.add_trace(go.Scatter(
            x=f["log_i"][v0:v1], y=f["potential"][v0:v1], mode="lines+markers",
            name=f["label"],
            legendgroup=grp,
            marker=dict(size=10, color=color, opacity=0.55),
            line=dict(color=color, width=1),
            customdata=[f["orig_label"]] * (v1 - v0),
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
            line=dict(color=fit_color, width=3, dash="dot"),
        ))
        xmid, ymid = float(np.mean(xline)), float(np.mean(yline))
        label_text = (
            f"{slope_abs:.0f} mV/dec ({f['reaction']})" if multi_reaction
            else f"{slope_abs:.0f} mV/dec"
        )
        fig.add_annotation(
            x=xmid, y=ymid, text=label_text,
            showarrow=False, yshift=16,
            font=dict(family="Arial", color=fit_color, size=22),
        )

    title_text = (
        f"Tafel plot — {' & '.join(distinct_reactions)}" if multi_reaction
        else "Tafel plot"
    )
    fig.update_layout(
        title=dict(text=title_text, font=dict(family="Arial", size=font_size)),
        template="plotly_white",
        height=560,
        font=axis_font,  # baseline (inherited by legend/annotations unless overridden)
        legend=dict(x=0.02, y=0.98, xanchor="left", yanchor="top", font=small_font,
                    bgcolor="rgba(255,255,255,0.7)", bordercolor="black", borderwidth=1,
                    tracegroupgap=18),
        margin=dict(l=10, r=10, t=60, b=10),
        dragmode="select",
    )
    # Axis titles/ticks are the "axes" text: always the full 28/36 pt Arial.
    # A sparse tick count (~4-5) keeps a publication-style plot uncluttered.
    fig.update_xaxes(
        title=dict(text=f"log₁₀ |Current| ({display_unit})",
                   font=dict(family="Arial", size=font_size)),
        tickfont=dict(family="Arial", size=font_size), nticks=5, **_BOX_AXIS_STYLE,
    )
    fig.update_yaxes(
        title=dict(text="Potential vs RHE / V", font=dict(family="Arial", size=font_size)),
        tickfont=dict(family="Arial", size=font_size), nticks=5, **_BOX_AXIS_STYLE,
    )
    st.caption(
        "🖱️ Drag a box around a sample's linear region to set its fit range "
        "directly (mouse now defaults to box-select instead of zoom — use "
        "the toolbar's zoom icon or double-click to reset the view). Drag "
        "the legend or a slope label to reposition it before exporting."
    )
    st.plotly_chart(
        fig, use_container_width=True, key="tafel_plot_select",
        on_select="rerun", selection_mode=["box"],
        config={
            "edits": {"annotationPosition": True, "legendPosition": True},
            "displaylogo": False,
        },
    )
    figure_downloads(
        fig, "tafel_combined_plot", key="png_tafel", what="Tafel plot",
        data=_padded_frame(tafel_plotted),
    )

    rows = []
    for f, r in results:
        slope_abs = abs(r.slope_mv_per_dec)
        ref = tafel.nearest_reference(slope_abs, f["reaction"])
        row = {
            "Sample": f["label"],
            "Replicate group": f["replicate_group"],
            "Reaction": f["reaction"],
            "Tafel slope (mV/dec)": round(slope_abs, 1),
            "R2": round(r.r_squared, 4),
            f"Intercept current at E=0 ({display_unit})": r.exchange_current,
            "Fit points": f["stop"] - f["start"],
            "Nearest mechanistic benchmark": (
                f"~{ref[0]:.0f} mV/dec ({ref[1]})" if ref else "—"
            ),
        }
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
    st.markdown("**Results summary**")
    st.dataframe(summary, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download Tafel fit summary (CSV)",
        data=summary.to_csv(index=False).encode("utf-8"),
        file_name="tafel_fit_summary.csv",
        mime="text/csv",
        key="dl_tafel_summary",
    )

    # Render the numbers explicitly (the CSV above keeps the full-precision
    # floats); an intercept current is otherwise printed with a long tail of
    # zeros that blows the column width out.
    display = summary.copy()
    intercept_col = f"Intercept current at E=0 ({display_unit})"
    display[intercept_col] = [
        "—" if v is None or not np.isfinite(v) else f"{v:.3e}"
        for v in summary[intercept_col]
    ]
    display["R2"] = [f"{v:.4f}" for v in summary["R2"]]
    _journal_table_figure(  # CSV of this table is the summary button above
        display, font_size, "tafel_results_table", key="png_tafel_table",
    )
    st.caption(
        "ℹ️ On the RHE scale, 0 V is exactly the H⁺/H₂ equilibrium potential, "
        "so for **HER and HOR** the intercept current above is the physical "
        "*exchange current* i₀. Every other reaction has its equilibrium "
        "potential elsewhere — OER/ORR ≈ 1.23 V, NO₃RR (to NH₃) ≈ 0.69 V, "
        "N₂RR (to NH₃) ≈ 0.09 V, CO₂RR ≈ −0.1 V (product-dependent) vs RHE — "
        "so there the intercept is a fit-extrapolation value only. Re-express "
        "the potential as an overpotential (η = E_RHE − E_eq) before fitting "
        "if i₀ is what you need."
    )

    if summary["Replicate group"].duplicated().any():
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
        for group_name, gdf in summary.groupby(group_col, sort=False):
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
        st.dataframe(replicate_summary, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download replicate statistics (CSV)",
            data=replicate_summary.to_csv(index=False).encode("utf-8"),
            file_name="tafel_replicate_statistics.csv", mime="text/csv",
            key="dl_tafel_replicate_stats",
        )
        _journal_table_figure(
            replicate_summary, font_size, "tafel_replicate_stats_table",
            key="png_tafel_replicate_table", what="Replicate statistics table",
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

    samples = _orr_data_loader(
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

    font_size = st.selectbox(
        "Figure export font size (pt)", _JOURNAL_FONT_SIZES, index=0,
        key="kl_font_size",
        help="Font is fixed to Arial for publication-style export.",
    )

    def _style_axes(fig, xtitle, ytitle):
        _journal_axes_style(fig, xtitle, ytitle, font_size)

    st.markdown("**Current unit & electrode area**")
    cur1, cur2, cur3 = st.columns(3)
    convert_density = cur1.checkbox(
        "Convert to current density (÷ area)", value=True,
        key="kl_convert_density",
        help="RDE current is usually reported as an absolute current (A, "
             "mA, µA); enable to normalize by the electrode's geometric "
             "area for a comparable current density.",
    )
    current_unit = cur2.selectbox(
        "Current unit as uploaded", ["A"] + _ABS_CURRENT_UNITS, index=0,
        key="kl_current_unit",
    )
    area_cm2 = cur3.number_input(
        "Electrode area (cm²)", min_value=1e-4, value=0.196, step=0.001,
        format="%.4f", key="kl_area_cm2",
        help="0.196 cm² is the standard 5 mm-diameter RDE glassy-carbon disk.",
    ) if convert_density else None
    display_unit = f"{current_unit}/cm²" if convert_density else current_unit

    to_rhe_fn = _render_rhe_conversion("kl", default_ph=13.0)

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
    disk = df["disk_current"].to_numpy(dtype=float)
    disk = disk / area_cm2 if convert_density else disk
    rpm_arr = df["rpm"].to_numpy(dtype=float)
    rpm_values = sorted(set(rpm_arr.tolist()))
    if len(rpm_values) < 3:
        st.error(
            f"{active_label}: only {len(rpm_values)} rotation rate(s) loaded "
            "— Koutecky-Levich needs at least 3 for a meaningful fit."
        )
        return

    st.markdown(f"**RDE curves — {active_label}**")
    fig_rde = go.Figure()
    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for i, rv in enumerate(rpm_values):
        m = np.isclose(rpm_arr, rv)
        order = np.argsort(pot_rhe[m])
        p, j = pot_rhe[m][order], disk[m][order]
        curves[rv] = (p, j)
        fig_rde.add_trace(go.Scatter(
            x=p, y=j, mode="lines", name=f"{rv:g} rpm",
            line=dict(color=_PALETTE[i % len(_PALETTE)], width=2.5),
        ))
    _style_axes(fig_rde, "Potential vs RHE / V", f"Disk current ({display_unit})")
    st.plotly_chart(fig_rde, use_container_width=True)
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
    n_points = st.number_input(
        "Number of analysis potentials (evenly spaced across the range all "
        "rotation rates share)",
        min_value=1, max_value=15, value=5, step=1, key="kl_n_points",
    )
    analysis_pots = np.linspace(lo, hi, int(n_points))

    kl_rows = []
    fig_kl = go.Figure()
    kl_data: dict[str, list] = {}
    for idx, ap in enumerate(analysis_pots):
        omegas_rpm, invj = [], []
        for rv in rpm_values:
            p, j = curves[rv]
            jval = float(np.interp(ap, p, j))
            if jval == 0 or not np.isfinite(jval):
                continue
            omegas_rpm.append(rv)
            invj.append(1.0 / jval)
        if len(omegas_rpm) < 3:
            continue
        try:
            fit = orr.fit_koutecky_levich(omegas_rpm, 1.0 / np.array(invj))
        except ValueError:
            continue
        n_val = orr.levich_slope_to_n(fit.slope, diff_coeff, viscosity, bulk_c)
        color = _PALETTE[idx % len(_PALETTE)]
        x = 1.0 / np.sqrt(orr.angular_velocity(omegas_rpm))
        y = np.array(invj)
        fig_kl.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=f"{ap:.3f} V",
            marker=dict(color=color, size=9),
        ))
        xline = np.array([float(np.min(x)), float(np.max(x))])
        yline = fit.slope * xline + fit.intercept
        fig_kl.add_trace(go.Scatter(
            x=xline, y=yline, mode="lines", showlegend=False,
            line=dict(color=_darken(color), width=2.5, dash="dot"),
        ))
        kl_data[f"{ap:.3f}V — 1/sqrt(omega)"] = list(x)
        kl_data[f"{ap:.3f}V — 1/j"] = list(y)
        kl_rows.append({
            "Potential (V vs RHE)": round(float(ap), 3),
            "KL slope": float(fit.slope),
            "R²": round(fit.r_squared, 4),
            # Reported as its positive magnitude (0-4), matching literature
            # convention -- the sign otherwise just tracks the disk current's
            # own recorded sign (negative for a cathodic/reduction sweep).
            "n (Levich)": round(abs(n_val), 2) if n_val is not None else None,
            f"j_k ({display_unit})": (
                round(fit.kinetic_current_density, 4)
                if fit.kinetic_current_density is not None else None
            ),
            "Rotation rates used": fit.n_rotation_rates,
        })

    if not kl_rows:
        st.warning(
            "Not enough overlapping rotation-rate data to fit any analysis "
            "potential — try fewer/different points, or check the uploaded "
            "files share a common potential range."
        )
        return

    _style_axes(fig_kl, "1 / √ω (s¹ᐟ²/rad¹ᐟ²)", f"1 / j ({display_unit}⁻¹)")
    st.plotly_chart(fig_kl, use_container_width=True)
    figure_downloads(
        fig_kl, f"kl_plot_{active_label}", key="png_kl_plot",
        what="Koutecky–Levich plot", data=_padded_frame(kl_data),
    )

    kl_summary = pd.DataFrame(kl_rows)
    st.markdown("**Results summary**")
    st.dataframe(kl_summary, use_container_width=True, hide_index=True)
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
        key="png_kl_table",
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


def _orr_read_file(up, key_prefix: str = "orr") -> pd.DataFrame | None:
    """Read one uploaded file's raw table (no column interpretation yet)."""
    try:
        if up.name.lower().endswith((".xlsx", ".xls")):
            sheets = data_io.list_sheets(io.BytesIO(up.getvalue()))
            sheet = sheets[0]
            if len(sheets) > 1:
                sheet = st.selectbox(
                    f"Sheet for {up.name}", sheets,
                    key=f"{key_prefix}_sheet_{up.name}",
                )
            return data_io.read_table(io.BytesIO(up.getvalue()), sheet=sheet)
        return data_io.read_table(io.BytesIO(up.getvalue()), sheet=None)
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
                order = np.argsort(pot_r)
                frame["ring_current"] = np.interp(pot_d, pot_r[order], cur_r[order])
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
            order = np.argsort(pot_r)
            ring_aligned = np.interp(pot, pot_r[order], ring[order])
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
) -> list[tuple[str, pd.DataFrame]]:
    """Batch-load RRDE files from a ZIP of a data folder (or several sample
    folders zipped together) — for pasting in a whole export at once instead
    of picking files one by one. Files are grouped into one sample per
    top-level folder inside the zip (files at the zip's own root all become
    one sample, named after the zip); role (disk/ring) and rotation rate are
    guessed entirely from filenames, the same way real RDE/RRDE software
    names its exports (e.g. ``Disk Current vs Disk Potential (1600
    RPM).csv``) — there is no per-file tagging UI here, since a batch upload
    may contain many files; use the per-sample uploaders above instead if a
    file needs correcting by hand.
    """
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

    default_sample = upload.name.rsplit(".", 1)[0]
    grouped: dict[str, list] = {}
    skipped = 0
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
    return samples


def _orr_data_loader(
    key_prefix: str = "orr",
    file_help: str = "Disk/ring current file(s) for this sample",
) -> list[tuple[str, pd.DataFrame]]:
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
    ``rpm``.
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
            zip_samples = _orr_extract_zip_samples(zip_up, key_prefix=key_prefix)
            if zip_samples:
                st.success(
                    f"Loaded {len(zip_samples)} sample(s) from {zip_up.name}: "
                    + ", ".join(lbl for lbl, _ in zip_samples)
                )
            samples.extend(zip_samples)

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
                raw = _orr_read_file(up, key_prefix=key_prefix)
                if raw is None:
                    continue
                coerced, cols = _orr_numeric_columns(raw)
                if len(cols) < 2:
                    st.warning(f"{up.name}: fewer than 2 numeric columns, skipped.")
                    continue
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
                samples.append((sample_name, df))
    return samples


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

    samples = _orr_data_loader()
    if not samples:
        return

    labels = [lbl for lbl, _ in samples]
    chosen = st.multiselect("Samples to compare", labels, default=labels, key="orr_chosen")
    if not chosen:
        st.info("Select at least one sample to plot.")
        return
    chosen_samples = {lbl: df for lbl, df in samples if lbl in chosen}

    font_size = st.selectbox(
        "Figure export font size (pt)", _JOURNAL_FONT_SIZES, index=0,
        key="orr_font_size",
        help="Font is fixed to Arial for publication-style export.",
    )

    st.markdown("**Current unit, electrode area & collection efficiency**")
    cur1, cur2, cur3, cur4 = st.columns(4)
    convert_density = cur1.checkbox(
        "Convert to current density (÷ area)", value=True,
        key="orr_convert_density",
        help="RRDE current is usually reported as an absolute current (A, "
             "mA, µA); enable to normalize by the electrode's geometric "
             "area for a comparable current density.",
    )
    current_unit = cur2.selectbox(
        "Current unit as uploaded", ["A"] + _ABS_CURRENT_UNITS, index=0,
        key="orr_current_unit",
    )
    area_cm2 = cur3.number_input(
        "Electrode area (cm²)", min_value=1e-4, value=0.196, step=0.001,
        format="%.4f", key="orr_area_cm2",
        help="0.196 cm² is the standard 5 mm-diameter RRDE glassy-carbon disk.",
    ) if convert_density else None
    display_unit = f"{current_unit}/cm²" if convert_density else current_unit
    collection_efficiency = cur4.number_input(
        "Ring collection efficiency N", min_value=0.01, max_value=1.0,
        value=0.37, step=0.01, format="%.2f", key="orr_collection_efficiency",
        help="From the RRDE electrode's own calibration (e.g. a "
             "ferri/ferrocyanide test); 0.37 is the common Pine 5 mm "
             "Pt-ring/glassy-carbon-disk default. Only used where ring "
             "current is available.",
    )

    to_rhe_fn = _render_rhe_conversion("orr", default_ph=13.0)

    # Prepare each sample's full (all-rotation-rate) data once: RHE
    # potential, disk/ring current density, and its own set of available
    # rpm values -- reused by every plot below.
    prepared = {}
    all_rpms: set[float] = set()
    for lbl, df in chosen_samples.items():
        pot_rhe = to_rhe_fn(df["potential"].to_numpy(dtype=float))
        disk = df["disk_current"].to_numpy(dtype=float)
        disk = disk / area_cm2 if convert_density else disk
        has_ring = "ring_current" in df.columns
        ring = None
        if has_ring:
            ring = df["ring_current"].to_numpy(dtype=float)
            ring = ring / area_cm2 if convert_density else ring
        rpm_arr = df["rpm"].to_numpy(dtype=float)
        rpm_values = sorted(set(rpm_arr.tolist()))
        all_rpms.update(rpm_values)
        prepared[lbl] = dict(
            potential=pot_rhe, disk=disk, ring=ring, has_ring=has_ring,
            rpm=rpm_arr, rpm_values=rpm_values,
        )

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
        slices[lbl] = dict(
            rpm=sample_rpm, potential=p["potential"][idx], disk=p["disk"][idx],
            ring=(p["ring"][idx] if p["has_ring"] else None), has_ring=p["has_ring"],
        )
    if not slices:
        return

    palette_for = {lbl: _PALETTE[i % len(_PALETTE)] for i, lbl in enumerate(chosen)}

    def _style_axes(fig, xtitle, ytitle, yrange=None):
        _journal_axes_style(fig, xtitle, ytitle, font_size, yrange=yrange)

    # ---- Disk polarization curve, all chosen samples overlaid ------------
    st.markdown(f"**Disk polarization curve @ ~{primary_rpm:g} rpm**")
    fig_disk = go.Figure()
    for lbl, s in slices.items():
        fig_disk.add_trace(go.Scatter(
            x=s["potential"], y=s["disk"], mode="lines", name=lbl,
            line=dict(color=palette_for[lbl], width=3),
        ))
    _style_axes(fig_disk, "Potential vs RHE / V", f"Disk current ({display_unit})")
    st.plotly_chart(fig_disk, use_container_width=True)
    disk_data = _padded_frame({
        **{f"{lbl} — Potential vs RHE (V)": list(s["potential"]) for lbl, s in slices.items()},
        **{f"{lbl} — Disk current ({display_unit})": list(s["disk"]) for lbl, s in slices.items()},
    })
    figure_downloads(
        fig_disk, f"orr_disk_curve_{int(primary_rpm)}rpm", key="png_orr_disk",
        what="Disk curve", data=disk_data,
    )

    # ---- Per-sample onset / E1/2 / Tafel ----------------------------------
    st.markdown("**Onset, half-wave potential & Tafel slope**")
    st.caption(
        "Each sample gets its own Tafel fit-range slider — auto-started "
        "near the kinetic (low-overpotential) region of its mass-transport-"
        "corrected current."
    )
    results_rows = []
    for lbl, s in slices.items():
        try:
            onset_res = orr.onset_and_half_wave(s["potential"], s["disk"])
        except ValueError as exc:
            st.warning(f"{lbl}: could not locate onset/E½ ({exc}).")
            continue
        row = {
            "Sample": lbl, "Rotation rate (rpm)": s["rpm"],
            "E_onset (V vs RHE)": round(onset_res.onset_potential, 3),
            "E_half-wave (V vs RHE)": round(onset_res.half_wave_potential, 3),
            f"j_limiting ({display_unit})": round(onset_res.limiting_current, 4),
        }

        jk = orr.mass_transport_corrected_current(s["disk"], onset_res.limiting_current)
        valid = np.isfinite(jk) & (jk != 0)
        tafel_result = None
        if valid.sum() >= 5:
            pot_tafel = s["potential"][valid]
            log_jk = tafel.log_current(jk[valid])
            a0, a1 = tafel.auto_tafel_range(pot_tafel, log_jk, current=jk[valid])
            range_key = f"orr_tafel_range_{lbl}"
            slider_kwargs = ({} if range_key in st.session_state
                             else {"value": (int(a0), int(a1))})
            start, stop = st.slider(
                f"{lbl} — Tafel fit range (index)", 0, len(pot_tafel),
                key=range_key, **slider_kwargs,
            )
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
            order = np.argsort(s["potential"])
            row["n @ E½"] = round(float(np.interp(
                onset_res.half_wave_potential, s["potential"][order], n_arr[order]
            )), 2)
            row["%H₂O₂ @ E½"] = round(float(np.interp(
                onset_res.half_wave_potential, s["potential"][order], pct_arr[order]
            )), 1)

        results_rows.append(row)
        s["onset"] = onset_res
        s["tafel"] = tafel_result
        s["jk"] = jk
        s["jk_valid"] = valid

    if results_rows:
        summary_df = pd.DataFrame(results_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download ORR summary (CSV)",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name=f"orr_summary_{int(primary_rpm)}rpm.csv", mime="text/csv",
            key="dl_orr_summary",
        )
        _journal_table_figure(  # CSV of this table is the summary button above
            summary_df, font_size, f"orr_results_table_{int(primary_rpm)}rpm",
            key="png_orr_table",
        )

    # ---- Tafel plot, all chosen samples overlaid --------------------------
    tafel_samples = {lbl: s for lbl, s in slices.items() if s.get("tafel") is not None}
    if tafel_samples:
        st.markdown("**Tafel plot (mass-transport corrected)**")
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
                marker=dict(size=8, color=color, opacity=0.5),
            ))
            xs = log_jk[start:stop]
            xline = np.array([float(np.min(xs)), float(np.max(xs))])
            yline = r.slope_v_per_dec * xline + r.intercept_v
            slope_abs = abs(r.slope_mv_per_dec)
            fit_color = _darken(color)
            fig_tafel.add_trace(go.Scatter(
                x=xline, y=yline, mode="lines", showlegend=False,
                name=f"{lbl} fit, {slope_abs:.0f} mV/dec",
                line=dict(color=fit_color, width=3, dash="dot"),
            ))
            xmid, ymid = float(np.mean(xline)), float(np.mean(yline))
            fig_tafel.add_annotation(
                x=xmid, y=ymid, text=f"{slope_abs:.0f} mV/dec", showarrow=False,
                yshift=14,
                font=dict(family="Arial", color=fit_color, size=round(font_size * 0.6)),
            )
        _style_axes(fig_tafel, f"log₁₀ |j_k| ({display_unit})", "Potential vs RHE / V")
        st.plotly_chart(fig_tafel, use_container_width=True)
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
        fig_ho2, fig_n = go.Figure(), go.Figure()
        ho2_data, n_data = {}, {}
        for lbl, s in ring_samples.items():
            color = palette_for[lbl]
            n_arr = orr.electron_number(s["disk"], s["ring"], collection_efficiency)
            pct_arr = orr.peroxide_percent(s["disk"], s["ring"], collection_efficiency)
            fig_ho2.add_trace(go.Scatter(
                x=s["potential"], y=pct_arr, mode="lines", name=lbl,
                line=dict(color=color, width=3),
            ))
            fig_n.add_trace(go.Scatter(
                x=s["potential"], y=n_arr, mode="lines", name=lbl,
                line=dict(color=color, width=3),
            ))
            ho2_data[f"{lbl} — Potential vs RHE (V)"] = list(s["potential"])
            ho2_data[f"{lbl} — %H2O2"] = list(pct_arr)
            n_data[f"{lbl} — Potential vs RHE (V)"] = list(s["potential"])
            n_data[f"{lbl} — n"] = list(n_arr)

        _style_axes(fig_ho2, "Potential vs RHE / V", "%H₂O₂", yrange=[0, 100])
        _style_axes(fig_n, "Potential vs RHE / V", "n", yrange=[0, 4])

        pc1, pc2 = st.columns(2)
        with pc1:
            st.plotly_chart(fig_ho2, use_container_width=True)
            figure_downloads(
                fig_ho2, f"orr_ho2_{int(primary_rpm)}rpm", key="png_orr_ho2",
                what="%H₂O₂ plot", data=_padded_frame(ho2_data),
            )
        with pc2:
            st.plotly_chart(fig_n, use_container_width=True)
            figure_downloads(
                fig_n, f"orr_n_{int(primary_rpm)}rpm", key="png_orr_n",
                what="n plot", data=_padded_frame(n_data),
            )

    # ---- Rotation-rate comparison, one sample, ring/disk merged axes -----
    multi_rpm_labels = [lbl for lbl in chosen if len(prepared[lbl]["rpm_values"]) > 1]
    if multi_rpm_labels:
        st.markdown("**Rotation-rate comparison (single sample)**")
        rrde_label = st.selectbox(
            "Sample", multi_rpm_labels, key="orr_rrde_sample",
            help="Every rotation rate this sample has, ring and disk current "
                 "sharing one potential axis — ring reads above zero, disk "
                 "below, so the pair reads as one merged figure (as in a "
                 "typical published RRDE overlay).",
        )
        p = prepared[rrde_label]
        fig_rrde = go.Figure()
        for i, rv in enumerate(p["rpm_values"]):
            m = np.isclose(p["rpm"], rv)
            color = _PALETTE[i % len(_PALETTE)]
            if p["has_ring"]:
                fig_rrde.add_trace(go.Scatter(
                    x=p["potential"][m], y=p["ring"][m], mode="lines",
                    name=f"{rv:g} rpm", legendgroup=f"{rv:g}",
                    line=dict(color=color, width=2.5),
                ))
            fig_rrde.add_trace(go.Scatter(
                x=p["potential"][m], y=p["disk"][m], mode="lines",
                name=f"{rv:g} rpm", legendgroup=f"{rv:g}",
                showlegend=not p["has_ring"],
                line=dict(color=color, width=2.5),
            ))
        if p["has_ring"]:
            fig_rrde.add_annotation(
                xref="paper", yref="paper", x=0.98, y=0.95, showarrow=False,
                text="ring", font=dict(family="Arial", size=round(font_size * 0.7)),
            )
            fig_rrde.add_annotation(
                xref="paper", yref="paper", x=0.98, y=0.05, showarrow=False,
                text="disk", font=dict(family="Arial", size=round(font_size * 0.7)),
            )
            fig_rrde.add_hline(y=0, line_color="black", line_width=1)
        _style_axes(fig_rrde, "Potential vs RHE / V", f"Current ({display_unit})")
        fig_rrde.update_layout(height=560)
        st.plotly_chart(fig_rrde, use_container_width=True)
        rrde_cols: dict[str, list] = {}
        for rv in p["rpm_values"]:
            m = np.isclose(p["rpm"], rv)
            rrde_cols[f"{rv:g}rpm — Potential vs RHE (V)"] = list(p["potential"][m])
            rrde_cols[f"{rv:g}rpm — Disk current ({display_unit})"] = list(p["disk"][m])
            if p["has_ring"]:
                rrde_cols[f"{rv:g}rpm — Ring current ({display_unit})"] = list(p["ring"][m])
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

    eis_list, lsv_list, label = sidebar_data_loader()

    tab_eis, tab_lsv, tab_tafel, tab_kl, tab_orr = st.tabs(
        ["📈 EIS / Ru Analysis", "🔬 LSV iR Correction", "📐 LSV Analysis",
         "📉 K-L Analysis", "⚛️ ORR / RRDE Analysis"]
    )

    if eis_list and lsv_list:
        # Link EIS dataset i with LSV dataset i (paired samples).
        n_pairs = min(len(eis_list), len(lsv_list))
        st.sidebar.header("2 · Sample")
        if len(eis_list) != len(lsv_list):
            st.sidebar.warning(
                f"EIS has {len(eis_list)} dataset(s) but LSV has "
                f"{len(lsv_list)}. Pairing the first {n_pairs} by position."
            )
        options = list(range(n_pairs))
        sel = st.sidebar.selectbox(
            "Active sample (EIS ↔ LSV pair)",
            options,
            format_func=lambda i: f"Sample {i + 1}",
            help="EIS pair i is linked to LSV pair i.",
        )
        eis_d, lsv_d = eis_list[sel], lsv_list[sel]
        st.sidebar.success(
            f"Loaded: {label}\n\n{n_pairs} sample(s) · "
            f"Sample {sel + 1}: EIS {len(eis_d)} pts · LSV {len(lsv_d)} pts\n\n"
            f"EIS cols: {eis_d.label}\n\nLSV cols: {lsv_d.label}"
        )

        cur_default, ru_default = _detect_units(eis_d.label, lsv_d.label)
        current_unit, ru_unit, area_cm2 = sidebar_units(cur_default, ru_default)

        with tab_eis:
            # Returns the raw Ru in the EIS file's unit; the LSV tab reconciles it
            # with the electrode area to report both Ru and Ru effective.
            ru = render_eis_tab(eis_d, eis_list, sel, ru_unit, current_unit, area_cm2)
        with tab_lsv:
            render_lsv_tab(lsv_d, ru, current_unit, ru_unit, area_cm2)
    else:
        with tab_eis:
            st.info(
                "⬅️ Load an EIS/LSV workbook or CSV pair in the sidebar "
                "(section 1) to use this tab."
            )
        with tab_lsv:
            st.info(
                "⬅️ Load an EIS/LSV workbook or CSV pair in the sidebar "
                "(section 1) to use this tab."
            )

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
