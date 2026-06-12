"""Streamlit GUI for automated iR compensation from EIS + LSV data.

Run with:
    streamlit run app.py

Workflow
--------
1. Upload an Excel workbook (Sheet 1 = EIS, Sheet 2 = LSV) or two CSV files.
2. **EIS / Ru Analysis** tab: fit the Nyquist arc to extract Ru (and Rct).
3. **LSV iR Correction** tab: apply the ohmic-drop correction with a
   compensation factor selectable from 5 % to 85 %; download the result.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ir_compensation import correction, data_io, eis

st.set_page_config(page_title="Automated iR Compensation", page_icon="⚡", layout="wide")

SAMPLE_PATH = "sample-data/Book1.xlsx"
REPO_URL = "https://github.com/rajeev4187/LSV-iR-compensation-calculation"
CITATION_TEXT = (
    "Kumar, R. (2026). Automated iR Compensation: EIS fitting and LSV "
    "correction (v1.0.0) [Computer software]. North Carolina Central "
    f"University. {REPO_URL}"
)
CITATION_BIBTEX = (
    "@software{kumar_ir_compensation_2026,\n"
    "  author  = {Kumar, Rajeev},\n"
    "  title   = {Automated iR Compensation: EIS fitting and LSV correction},\n"
    "  version = {1.0.0},\n"
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


def png_download(fig, filename: str, key: str,
                 label: str = "⬇️ Download figure (PNG)") -> None:
    """Render a button that exports a Plotly figure as a high-res PNG.

    Falls back to a hint about the chart's built-in camera icon if server-side
    rendering (kaleido) is unavailable.
    """
    try:
        png = fig.to_image(format="png", width=1100, height=520, scale=2)
    except Exception as exc:  # kaleido missing / render error
        st.caption(
            f"PNG export unavailable ({exc}). Use the 📷 icon on the chart "
            "to save a PNG instead."
        )
        return
    st.download_button(
        label, data=png, file_name=filename, mime="image/png", key=key
    )


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def sidebar_data_loader():
    """Render the data-source controls.

    Returns ``(eis_list, lsv_list, label)`` where each list holds one or more
    datasets parsed from repeated column pairs in the sheet/file.
    """
    st.sidebar.header("1 · Data source")
    source = st.sidebar.radio(
        "Choose input",
        ["Sample workbook", "Upload Excel workbook", "Upload two CSV files"],
        help="Excel: EIS sheet = Z', Z'' pairs; LSV sheet = Potential, Current "
             "pairs. Several datasets may sit side-by-side as repeated pairs.",
    )

    try:
        if source in ("Sample workbook", "Upload Excel workbook"):
            if source == "Sample workbook":
                src = SAMPLE_PATH
                name = "sample-data/Book1.xlsx"
            else:
                up = st.sidebar.file_uploader("Excel (.xlsx)", type=["xlsx", "xls"])
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
        eis_up = st.sidebar.file_uploader("EIS CSV (Z', Z'')", type=["csv", "txt"])
        lsv_up = st.sidebar.file_uploader(
            "LSV CSV (Potential, Current)", type=["csv", "txt"]
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
# EIS / Ru analysis tab                                                       #
# --------------------------------------------------------------------------- #
def render_eis_tab(eis_d, eis_list, sel) -> float | None:
    """Render the EIS analysis tab and return the chosen Ru (ohm)."""
    st.subheader("EIS — uncompensated resistance (Ru) from the Nyquist arc")
    st.caption(
        f"Sample {sel + 1} · columns: {eis_d.label or 'Z′, Z″'}. The "
        "high-frequency real-axis intercept of the kinetic semicircle is Ru; "
        "adjust the arc range to exclude the low-frequency diffusion tail."
    )

    n = len(eis_d)
    auto_start, auto_stop = eis.auto_arc_range(eis_d.z_real, eis_d.z_imag)

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
                ru_result = eis.fit_ru_circle(
                    eis_d.z_real, eis_d.z_imag, start=start, stop=stop
                )
            except Exception as exc:
                st.error(f"Circle fit failed: {exc}")
        else:
            quick = eis.fit_ru_circle(eis_d.z_real, eis_d.z_imag, start=start, stop=stop)
            manual_ru_val = st.number_input(
                "Ru (Ω)", value=float(round(quick.ru, 3)), step=0.1, format="%.3f"
            )

    with right:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=eis_d.z_real,
                y=np.abs(eis_d.z_imag),
                mode="markers",
                name="EIS data",
                marker=dict(size=7, color="#1f77b4"),
            )
        )
        # Highlight the points selected for the fit.
        fig.add_trace(
            go.Scatter(
                x=eis_d.z_real[start:stop],
                y=np.abs(eis_d.z_imag)[start:stop],
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
                    name="Ru", text=[f"Ru={ru_for_marker:.2f} Ω"],
                    textposition="top center",
                    marker=dict(size=13, color="red", symbol="x"),
                )
            )
        fig.update_layout(
            xaxis_title="Z′ / Ω",
            yaxis_title="−Z″ / Ω",
            template="plotly_white",
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        fig.update_yaxes(scaleanchor="x", scaleratio=1)  # equal aspect -> true circle
        st.plotly_chart(fig, use_container_width=True)
        png_download(
            fig, f"nyquist_sample{sel + 1}.png", key="png_eis",
            label="⬇️ Download Nyquist plot (PNG)",
        )

    # Metrics row
    chosen_ru = None
    if ru_result is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ru (Ω)", f"{ru_result.ru:.3f}")
        c2.metric("Rct (Ω)", f"{ru_result.rct:.3f}" if ru_result.rct else "—")
        c3.metric(
            "Ru + Rct (Ω)",
            f"{ru_result.r_low:.3f}" if ru_result.r_low else "—",
        )
        c4.metric(
            "Fit RMSE (Ω)",
            f"{ru_result.rmse:.3f}" if ru_result.rmse else "—",
        )
        chosen_ru = ru_result.ru
    elif manual_ru_val is not None:
        st.metric("Ru (Ω)", f"{manual_ru_val:.3f}")
        chosen_ru = float(manual_ru_val)

    # Batch view: circle-fit Ru for every loaded EIS sample.
    if len(eis_list) > 1:
        with st.expander(f"Ru for all {len(eis_list)} samples (batch fit)"):
            rows = []
            for i, d in enumerate(eis_list):
                try:
                    rr = eis.fit_ru_circle(d.z_real, d.z_imag)
                    rows.append({
                        "Sample": f"Sample {i + 1}",
                        "Columns": d.label,
                        "Ru (Ω)": round(rr.ru, 3),
                        "Rct (Ω)": round(rr.rct, 3) if rr.rct else None,
                        "RMSE (Ω)": round(rr.rmse, 3) if rr.rmse else None,
                    })
                except Exception as exc:
                    rows.append({
                        "Sample": f"Sample {i + 1}",
                        "Columns": d.label, "Ru (Ω)": None,
                        "Rct (Ω)": None, "RMSE (Ω)": f"fit failed: {exc}",
                    })
            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True, hide_index=True,
            )

    return chosen_ru


# --------------------------------------------------------------------------- #
# LSV iR-correction tab                                                        #
# --------------------------------------------------------------------------- #
_FACTOR_CHOICES = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 85]
_PALETTE = ["#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b",
            "#e377c2", "#17becf", "#bcbd22"]


def _build_export_csv(lsv_d, results, ru, current_unit, source_label) -> str:
    """Assemble a CSV with metadata comment lines + one column block per factor."""
    meta = [
        "# Automated iR compensation — corrected LSV data",
        f"# Source: {source_label}",
        f"# Ru (uncompensated resistance) = {ru:.4f} ohm",
        f"# Current unit in source = {current_unit}",
        f"# Compensation factors (%) = {', '.join(str(r.factor_percent) for r in results)}",
        "# Formula: E_corrected = E_measured - (factor/100) * I * Ru",
    ]
    cols = {
        "Potential_raw_V": lsv_d.potential,
        f"Current_{current_unit}": lsv_d.current,
    }
    for r in results:
        p = int(r.factor_percent)
        cols[f"iRdrop_{p}pct_V"] = r.ir_drop
        cols[f"Ecorr_{p}pct_V"] = r.potential_corrected
    body = pd.DataFrame(cols).to_csv(index=False)
    return "\n".join(meta) + "\n" + body


def render_lsv_tab(lsv_d, ru: float | None, source_label: str):
    st.subheader("LSV — ohmic-drop (iR) correction")
    if ru is None:
        st.warning("Determine Ru on the **EIS / Ru Analysis** tab first.")
        return

    left, right = st.columns([1, 2])
    with left:
        st.metric("Using Ru (Ω)", f"{ru:.3f}")
        current_unit = st.selectbox(
            "Current unit (in the data file)",
            ["mA", "A", "µA", "nA"],
            index=0,
            help="Used to convert I·Ru to volts.",
        )
        factors = st.multiselect(
            "Compensation factors (%) — compare several",
            options=_FACTOR_CHOICES,
            default=[85],
            help="Each selected factor is corrected, plotted, and exported "
                 "(project range: 5–85 %).",
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
            "E_corrected = E_measured − (factor) · I · Ru. "
            "Partial compensation (≤ 85 %) avoids over-correction/oscillation."
        )

    factors = sorted(set(factors))
    results = [
        correction.apply_ir_correction(
            lsv_d.potential, lsv_d.current, ru,
            factor_percent=f, current_unit=current_unit,
        )
        for f in factors
    ]

    with right:
        view = st.radio(
            "Comparison view",
            ["Overlay (same axes)", "Side-by-side"],
            horizontal=True,
            help="Both show the LSV with vs without iR compensation.",
        )
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

        fig.update_layout(
            template="plotly_white",
            height=440,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=10, r=10, t=70, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        fac_png = "-".join(str(int(r.factor_percent)) for r in results)
        png_download(
            fig, f"lsv_iR_comparison_f{fac_png}pct.png", key="png_lsv",
            label="⬇️ Download comparison plot (PNG)",
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

    # Per-factor summary table.
    summary = pd.DataFrame(
        {
            "Compensation %": [int(r.factor_percent) for r in results],
            "Ru (Ω)": [round(ru, 3)] * len(results),
            "Max |iR drop| (mV)": [
                round(float(np.max(np.abs(r.ir_drop))) * 1000, 2)
                for r in results
            ],
            "Fold-back %": [
                round(a.reverted_fraction * 100, 1) for a in assessments
            ],
            "Status": [_status(a) for a in assessments],
        }
    )
    st.markdown("**Results summary**")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Warn if any selected factor over-compensates, and recommend a safe one.
    if any(a.over_compensated for a in assessments):
        bad = [int(r.factor_percent)
               for r, a in zip(results, assessments) if a.over_compensated]
        rec = correction.recommend_factor(
            lsv_d.potential, lsv_d.current, ru, current_unit
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
            "fold-back appears; this tool caps it at **85 %** for safety. If "
            "fold-back shows up below 85 %, your `Ru` may be over-estimated — "
            "re-check the EIS arc fit."
        )

    csv_text = _build_export_csv(lsv_d, results, ru, current_unit, source_label)
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
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    st.title("⚡ Automated iR Compensation")
    st.caption("EIS fitting → Ru extraction → LSV ohmic-drop correction")

    eis_list, lsv_list, label = sidebar_data_loader()
    if not eis_list or not lsv_list:
        st.stop()

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

    sample_tag = f"Sample {sel + 1}"
    tab_eis, tab_lsv = st.tabs(
        ["📈 EIS / Ru Analysis", "🔬 LSV iR Correction"]
    )
    with tab_eis:
        ru = render_eis_tab(eis_d, eis_list, sel)
    with tab_lsv:
        render_lsv_tab(
            lsv_d, ru, f"{label or 'uploaded data'} [{sample_tag}]"
        )

    render_citation()
    st.divider()
    st.caption(
        f"Automated iR Compensation v1.0.0 · please cite if used "
        f"([repository]({REPO_URL})). See **Cite this app** in the sidebar."
    )


if __name__ == "__main__":
    main()
