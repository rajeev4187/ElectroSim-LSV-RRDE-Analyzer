"""Streamlit GUI for LSV analysis: iR compensation and Tafel-slope analysis.

Run with:
    streamlit run app.py

Workflow
--------
1. Upload an Excel workbook (Sheet 1 = EIS, Sheet 2 = LSV) or two CSV files.
2. **EIS / Ru Analysis** tab: fit the Nyquist arc to extract Ru (and Rct).
3. **LSV iR Correction** tab: apply the ohmic-drop correction with a
   compensation factor selectable from 5 % to 100 %; download the result.
4. **Tafel Slope Analysis** tab: independent of the above — upload its own
   polarization-curve file and fit the linear (kinetic) region to extract the
   Tafel slope.
"""

from __future__ import annotations

import hashlib
import hmac
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ir_compensation import correction, data_io, eis, tafel

st.set_page_config(
    page_title="LSV analysis-iR compensation, Tafel slope anlaysis",
    page_icon="⚡", layout="wide",
)

SAMPLE_PATH = "sample-data/Book1-original data.xlsx"
REPO_URL = "https://github.com/rajeev4187/LSV-Analysis-iR-compensation-Tafel-slope"
CITATION_TEXT = (
    "Kumar, R. (2026). LSV Analysis: iR Compensation and Tafel Slope "
    "(v1.1.0) [Computer software]. North Carolina Central "
    f"University. {REPO_URL}"
)
CITATION_BIBTEX = (
    "@software{kumar_lsv_analysis_2026,\n"
    "  author  = {Kumar, Rajeev},\n"
    "  title   = {LSV Analysis: iR Compensation and Tafel Slope},\n"
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

    st.title("⚡ LSV analysis-iR compensation, Tafel slope anlaysis")
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

    Used to detect that a cached PNG no longer matches the figure on screen
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
# server-side PNG, or a standalone HTML file — takes them literally, which is
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


def figure_downloads(fig, stem: str, key: str, what: str = "figure",
                     width: int | None = None, height: int | None = None,
                     data: "pd.DataFrame | None" = None) -> None:
    """Render the download controls for one figure: PNG, interactive HTML and
    (optionally) the plotted data as CSV.

    PNG rendering is server-side (kaleido), which launches a headless browser
    per call — too slow/fragile to run on *every* script rerun (it would fire
    on every unrelated widget interaction, e.g. dragging a slider). It only
    happens when the user clicks "Prepare"; the bytes are cached in
    session_state together with a signature of the figure, so the download
    button persists across reruns but is withdrawn as soon as the figure
    itself changes. The HTML and CSV exports need no external renderer and are
    therefore always available — they are the fallback when kaleido/Chrome is
    unavailable on the host (e.g. a slim cloud container).
    """
    state_key = f"_export_{key}"
    export_fig = _export_figure(fig)
    sig = _figure_signature(export_fig)
    cols = st.columns(3 if data is not None else 2)

    with cols[0]:
        if st.button(f"🖼️ Prepare {what} (PNG)", key=f"_png_prep_{key}",
                     use_container_width=True):
            w, h = _export_size(export_fig, width, height)
            try:
                st.session_state[state_key] = {
                    "sig": sig,
                    "bytes": export_fig.to_image(format="png", width=w,
                                                 height=h, scale=2),
                    "error": None,
                }
            except Exception as exc:  # kaleido missing / no Chrome / render error
                st.session_state[state_key] = {
                    "sig": sig, "bytes": None, "error": str(exc),
                }
        cached = st.session_state.get(state_key) or {}
        if cached.get("bytes") and cached.get("sig") == sig:
            st.download_button(
                f"⬇️ {what} (PNG)", data=cached["bytes"],
                file_name=f"{stem}.png", mime="image/png", key=f"_png_dl_{key}",
                use_container_width=True,
            )
        elif cached.get("bytes"):
            st.caption("↻ Figure changed — press Prepare again.")
        elif cached.get("error"):
            st.caption(
                f"PNG export unavailable ({cached['error']}). Use the HTML "
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
                     "when the server-side PNG renderer is not.",
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
        fig.update_layout(
            xaxis_title=f"Z′ / {disp_unit}",
            yaxis_title=f"−Z″ / {disp_unit}",
            template="plotly_white",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
        )
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
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, hide_index=True,
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
        with st.expander("🔧 Axis range (applies to plot & PNG export)"):
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

        # Apply manual axis ranges (affects both the on-screen plot and PNG).
        if x_range is not None:
            fig.update_xaxes(range=x_range)
        if y_range is not None:
            fig.update_yaxes(range=y_range)

        # Legend at the bottom so it never overlaps the title / subplot titles.
        fig.update_layout(
            template="plotly_white",
            height=470,
            title=dict(y=0.97, yanchor="top"),
            legend=dict(orientation="h", yanchor="top", y=-0.18,
                        xanchor="center", x=0.5),
            margin=dict(l=10, r=10, t=60, b=90),
        )
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
_TAFEL_FONT_SIZES = [28, 36]
# Journal style: a closed box border (mirrored axis lines) around the plot,
# with no interior gridlines.
_BOX_AXIS_STYLE = dict(
    showgrid=False, zeroline=False,
    showline=True, linewidth=1.5, linecolor="black", mirror=True,
    ticks="outside",
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


def render_tafel_tab() -> None:
    st.subheader("Tafel slope analysis")
    st.caption(
        "Fits E = a + b·log₁₀|i| over the linear (activation-controlled) "
        "region of a polarization curve; **b** is the Tafel slope, reported "
        "as its positive magnitude (mV/dec) per literature convention. This "
        "tab has its own file upload and does not use the EIS/LSV data "
        "loaded in the sidebar."
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
        "Figure/table export font size (pt)", _TAFEL_FONT_SIZES, index=0,
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

    labels = [lbl for lbl, _ in series]
    chosen = st.multiselect(
        "Samples to combine (journal-style overlay)",
        labels, default=labels[: min(len(labels), 8)], key="tafel_chosen",
    )
    if not chosen:
        st.info("Select at least one sample to plot.")
        return
    chosen_series = [(lbl, d) for lbl, d in series if lbl in chosen]

    st.markdown("**Reference electrode → RHE conversion**")
    st.caption(
        "E(RHE) = E(measured) + E°(reference vs NHE) + 0.0592 V·pH⁻¹ × pH. "
        "All fitting, plotting, and the exported data below use the "
        "RHE-converted potential."
    )
    already_rhe = st.checkbox(
        "Input data is already reported vs RHE (skip conversion)",
        value=False, key="tafel_already_rhe",
    )
    e_ref, ph = 0.0, 0.0
    if not already_rhe:
        rc1, rc2, rc3 = st.columns([2, 1, 1])
        ref_names = list(tafel.REFERENCE_ELECTRODES) + ["Custom"]
        ref_choice = rc1.selectbox(
            "Reference electrode used for the input data", ref_names,
            index=0, key="tafel_ref_electrode",
        )
        if ref_choice == "Custom":
            e_ref = rc2.number_input(
                "E° vs NHE (V)", value=0.000, step=0.001, format="%.3f",
                key="tafel_ref_custom",
            )
        else:
            e_ref = tafel.REFERENCE_ELECTRODES[ref_choice]
            rc2.metric("E° vs NHE (V)", f"{e_ref:.3f}")
        ph = rc3.number_input(
            "Electrolyte pH", min_value=0.0, max_value=14.0, value=7.0,
            step=0.1, key="tafel_ph",
        )
        st.caption(
            f"↪ E(RHE) = E(measured) + {e_ref:.3f} V + 0.0592 × {ph:g} = "
            f"E(measured) + {(e_ref + tafel.NERNST_SLOPE_V_PER_PH * ph):.3f} V"
        )

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
            pot = d.potential[mask] if already_rhe else tafel.to_rhe(
                d.potential[mask], e_ref, ph
            )
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
            c0, cr, c1, c2 = st.columns([1.1, 0.9, 2.0, 0.5])
            display_name = c0.text_input(
                "Legend name", value=lbl, key=f"tafel_name_{i}",
                help="Shown in the plot legend; edit if the auto-detected "
                     "name isn't the one you want.",
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
                             potential=pot, log_i=log_i,
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
        pot_full = d.potential if already_rhe else tafel.to_rhe(d.potential, e_ref, ph)
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
        rows.append({
            "Sample": f["label"],
            "Reaction": f["reaction"],
            "Tafel slope (mV/dec)": round(slope_abs, 1),
            "R2": round(r.r_squared, 4),
            f"Intercept current at E=0 ({display_unit})": r.exchange_current,
            "Fit points": f["stop"] - f["start"],
            "Nearest mechanistic benchmark": (
                f"~{ref[0]:.0f} mV/dec ({ref[1]})" if ref else "—"
            ),
        })
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

    # Results table as a figure (Arial, publication-style). The exported
    # canvas is sized from the table's own content: columns are widened in
    # proportion to the longest string they hold (the mechanistic-benchmark
    # column is far wider than "R2"), and the row height leaves room for the
    # lines Plotly wraps text onto — otherwise long entries overlap and the
    # last rows fall outside a fixed-size export.
    cell_font = max(9.0, font_size * 0.4)
    header_font = max(10.0, font_size * 0.45)
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
    str_cols = {c: display[c].astype(str) for c in display.columns}
    widths = []
    for c in display.columns:
        longest = str_cols[c].str.len().max()
        longest = int(longest) if pd.notna(longest) else 0
        # +2 characters of breathing room so short entries aren't wrapped, and
        # a cap so one long sentence can't squeeze every other column.
        widths.append(min(max(len(str(c)), longest, 6) + 2, 46))
    char_px = cell_font * 0.66
    table_width = int(sum(widths) * char_px + 60)
    # Worst-case wrapped lines in any cell of a row -> uniform row height.
    max_wrap = max(
        (int(np.ceil(len(v) / w)) for col, w in zip(display.columns, widths)
         for v in str_cols[col]),
        default=1,
    )
    row_h = int(cell_font * 1.5 * max(1, max_wrap)) + 8
    table_fig = go.Figure(data=[go.Table(
        columnwidth=widths,
        header=dict(values=list(display.columns),
                    font=dict(family="Arial", size=header_font),
                    align="left", height=int(header_font * 1.6) + 10),
        cells=dict(values=[str_cols[c] for c in display.columns],
                   font=dict(family="Arial", size=cell_font), align="left",
                   height=row_h),
    )])
    table_height = int(header_font * 1.6) + 20 + row_h * len(display) + 20
    table_fig.update_layout(
        template="plotly_white",  # never export with Streamlit's placeholder colours
        margin=dict(l=10, r=10, t=10, b=10),
        width=table_width, height=table_height,
    )
    figure_downloads(  # CSV of this table is the summary button above
        table_fig, "tafel_results_table", key="png_tafel_table",
        what="Results table", width=table_width, height=table_height,
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
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    if not require_access():
        st.stop()

    st.title("⚡ LSV analysis-iR compensation, Tafel slope anlaysis")
    st.caption(
        "EIS fitting → Ru extraction → LSV ohmic-drop correction, plus "
        "independent Tafel-slope analysis"
    )

    eis_list, lsv_list, label = sidebar_data_loader()

    tab_eis, tab_lsv, tab_tafel = st.tabs(
        ["📈 EIS / Ru Analysis", "🔬 LSV iR Correction", "📐 Tafel Slope Analysis"]
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

    render_citation()
    st.divider()
    st.caption(
        f"LSV analysis-iR compensation, Tafel slope anlaysis v1.1.0 · please "
        f"cite if used ([repository]({REPO_URL})). See **Cite this app** in "
        "the sidebar."
    )


if __name__ == "__main__":
    main()
