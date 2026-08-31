"""EIS Nyquist/Bode fit figures + tabulated report, styled like the app.

Two datasets, each already exported by the instrument software as ready-made
Nyquist (Z', -Z'') and Bode (|Z|, phase vs frequency) CSVs per EIS-POT
experiment:

1. **1 M KOH, HER/OER catalysts** (``sample-data/Obed data/20260730/``) --
   5 samples (LaFeO3, 5/10/20 wt% ReS2-LaFeO3, ReS2), each with 3 EIS-POT
   runs at OCP, the HER hold potential (-1.5 V vs Hg/HgO) and the OER hold
   potential (+0.9 V vs Hg/HgO) -- the exact same potentials as the
   `make_reviewer_figures.py` chronoamperometry. RHE conversion uses the
   user-supplied 1 M KOH direct-calibration offset (0.945 V).
2. **0.1 M KOH, ORR catalysts** (``sample-data/Obed data/RRDE LFO-ReS2/``) --
   as many of the same 5 samples as have a clean, unambiguous EIS-POT
   folder (see ``ORR_EIS_SAMPLES`` below); RHE conversion uses the
   0.841 V direct-calibration offset instead.

For each dataset this produces one PNG per condition, each a 5-row x 2-col
grid (Nyquist | Bode) across all samples, plus one CSV + one table PNG of
the fitted Ru/Rct/RMSE/arc-coverage parameters for every sample x condition.

Run from the repository root::

    python scripts/make_eis_figures.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app  # noqa: E402 -- styling helpers (_nice_axis, fonts, palettes)
from scripts.modules import eis  # noqa: E402
from scripts.make_reviewer_figures import (  # noqa: E402
    FAMILY, FONT_SIZE, AXIS_FONT, TITLE_FONT, PALETTE,
    PANEL_W, RHE_1M_KOH, RHE_01M_KOH, OUT, ROOT, render_png,
)

DATA = os.path.join(ROOT, "sample-data", "Obed data")

# --- 1 M KOH: HER/OER catalyst EIS ------------------------------------------
# (sample label, folder, {condition: EIS-POT experiment-number suffix})
EIS_1M_KOH_DIR = os.path.join(DATA, "20260730")
EIS_1M_KOH_SAMPLES = [
    ("LaFeO3", "LFO 1 M KOH N2/LFO 1 M KOH [2026-JUL-30 1221 #1]",
     {"OCP": "2", "HER": "8", "OER": "6"}),
    ("5% ReS2-LaFeO3", "5-ReS2-lfo 1 M KOH [2026-JUL-30 1521 #1]",
     {"OCP": "20", "HER": "25", "OER": "26"}),
    ("10% ReS2-LaFeO3", "10-ReS2-lfo 1 M KOH [2026-JUL-30 1521 #1]",
     {"OCP": "28", "HER": "32", "OER": "33"}),
    ("20% ReS2-LaFeO3", "20-ReS2-lfo 1 M KOH [2026-JUL-30 1621 #1]",
     {"OCP": "39", "HER": "43", "OER": "44"}),
    ("ReS2", "ReS2 1 M KOH [2026-JUL-30 1421 #1]",
     {"OCP": "11", "HER": "17", "OER": "18"}),
]
# V vs Hg/HgO, matching the CA holds in make_reviewer_figures.py.
EIS_1M_KOH_POTENTIALS = {"OCP": None, "HER": -1.500, "OER": +0.900}

# --- 0.1 M KOH: ORR catalyst EIS --------------------------------------------
ORR_EIS_DIR = os.path.join(DATA, "RRDE LFO-ReS2")
# Folder discovery from the reconnaissance pass: each sample's own ORR run
# folder holds 3 EIS-POT experiments at ~-0.10/-0.28/-0.40 V vs Hg/HgO. The
# exact experiment-number suffix differs per sample (unlike the 1 M KOH set,
# these were not run from one shared sequence), so each is spelled out by
# its full folder name rather than a shared numbering scheme.
ORR_EIS_SAMPLES = [
    ("LaFeO3", "20240807 ORR - LFO/2024 ORR 0-1 M KOH LFO",
     {"-0.10 V": "EIS-POT Experiment [2024-AUG-07 1709 #22]",
      "-0.28 V": "EIS-POT Experiment [2024-AUG-07 1720 #24]"}),
    ("ReS2", "20240807 ORR RES2/20240807 ORR 0-1 M KOH ReS2",
     {"-0.40 V": "EIS-POT Experiment [2024-AUG-07 1520 #3]",
      "-0.28 V": "EIS-POT Experiment [2024-AUG-07 1523 #4]",
      "-0.10 V": "EIS-POT Experiment [2024-AUG-07 1528 #5]"}),
    ("5% ReS2-LaFeO3", "5% RES2-LFO - ORR/5% RES2-LFO  ORR 0-1 M KOH LFO",
     {"-0.10 V": "EIS-POT Experiment [2024-AUG-09 1446 #34]",
      "-0.28 V": "EIS-POT Experiment [2024-AUG-09 1449 #35]",
      "-0.40 V": "EIS-POT Experiment [2024-AUG-09 1453 #36]"}),
    # Only one clean EIS-POT for 10%: the "- 2" duplicate folder's own 3
    # experiments sit behind a path over Windows' 260-char MAX_PATH (the
    # folder name is nested inside itself), so os.path.exists silently
    # returns False for them -- this shorter, singly-nested folder is used
    # instead, and it only has this one condition.
    ("10% ReS2-LaFeO3", "20240809 ORR RES2-LFO/10% RES2-LFO  ORR 0-1 M KOH LFO",
     {"-0.10 V": "EIS-POT Experiment [2024-AUG-09 1247 #8]"}),
]
# 20% ReS2-LaFeO3 is NOT included: the only entry under this tree is
# "20% RES2-LFO ORR.opju", an Origin project file with no raw CSV export --
# see figures/reviewer-response/PROGRESS.md. Reported as a gap below rather
# than silently omitted.


def load_eis(expt_dir: str):
    """``(freq /Hz, z_real /ohm, z_imag /ohm)`` from one EIS-POT experiment,
    high frequency first (the order the instrument itself records)."""
    nyq = np.genfromtxt(
        os.path.join(expt_dir, "Nyquist Plot", "Nyquist (-Zi vs Zr).csv"),
        delimiter=",", skip_header=1)
    mag = np.genfromtxt(
        os.path.join(expt_dir, "Bode Plot", "Magnitude !Z! vs Frequency.csv"),
        delimiter=",", skip_header=1)
    phase = np.genfromtxt(
        os.path.join(expt_dir, "Bode Plot", "Phase (Φ) vs Frequency.csv"),
        delimiter=",", skip_header=1)
    freq, zr, zi = mag[:, 0], nyq[:, 0], nyq[:, 1]
    # Sanity check the file's own internal consistency (|Z| = sqrt(Zr^2+Zi^2))
    # before trusting the row alignment between the two separately-exported
    # CSVs -- see the ORR unit-consistency lesson in PROGRESS.md: two arrays
    # that are supposed to correspond row-for-row but come from different
    # files are exactly the kind of thing worth checking once, in code,
    # rather than assuming.
    calc = np.sqrt(zr**2 + zi**2)
    if not np.allclose(calc, mag[:, 1], rtol=1e-3):
        raise ValueError(f"Nyquist/Bode magnitude mismatch in {expt_dir}")
    return freq, zr, zi, phase[:, 1]


def dc_bias_v(expt_dir: str) -> float:
    """The DC bias (V vs Hg/HgO) this EIS-POT experiment was recorded at."""
    f = os.path.join(expt_dir, "experiment", "results",
                     "Applied EIS Offset (1 Hz).csv")
    return float(open(f).readlines()[1].split(",")[1])


def _subplot_axis(row: int, col: int, cols: int) -> tuple[str, str]:
    """Plotly axis id strings ('x3','y3', etc.) for a make_subplots cell."""
    idx = (row - 1) * cols + col
    suffix = "" if idx == 1 else str(idx)
    return f"x{suffix}", f"y{suffix}"


def make_table_png(df: pd.DataFrame, path: str, *, dpi: int = 300) -> None:
    """A journal-style table PNG, styling copied from app._journal_table_figure
    (that function also calls Streamlit's download widgets, so its table
    logic is reproduced here rather than imported)."""
    family = FAMILY
    cell_font = max(9.0, FONT_SIZE * 0.4)
    header_font = max(10.0, FONT_SIZE * 0.45)
    str_cols = {c: df[c].apply(lambda v: "—" if pd.isna(v) else str(v))
               for c in df.columns}
    widths = [min(max(len(str(c)), int(str_cols[c].str.len().max()), 6) + 2, 46)
             for c in df.columns]
    char_px = cell_font * 0.66
    table_width = int(min(max(sum(widths) * char_px + 60, 200), 4000))
    max_wrap = max((int(np.ceil(len(v) / w))
                   for col, w in zip(df.columns, widths) for v in str_cols[col]),
                  default=1)
    row_h = int(cell_font * 1.5 * max(1, max_wrap)) + 8
    n_rows = len(df)
    row_fill = ["#ffffff" if i % 2 == 0 else "#f4f4f4" for i in range(n_rows)]

    def _is_numeric(col):
        vals = [v for v in str_cols[col] if v != "—"]
        if not vals:
            return False
        try:
            [float(v) for v in vals]
            return True
        except ValueError:
            return False

    aligns = ["right" if _is_numeric(c) else "left" for c in df.columns]
    fig = go.Figure(data=[go.Table(
        columnwidth=widths,
        header={"values": [f"<b>{c}</b>" for c in df.columns],
                "font": {"family": family, "size": header_font, "color": "#000000"},
                "fill": {"color": "#e2e2e2"},
                "line": {"color": "#000000", "width": 1},
                "align": aligns, "height": int(header_font * 1.6) + 10},
        cells={"values": [str_cols[c] for c in df.columns],
               "font": {"family": family, "size": cell_font}, "align": aligns,
               "fill": {"color": [row_fill] * len(df.columns)},
               "line": {"color": "rgba(0,0,0,0)", "width": 0}, "height": row_h},
    )])
    table_height = int(header_font * 1.6) + 20 + row_h * n_rows + 20
    fig.update_layout(template="plotly_white",
                      margin={"l": 10, "r": 10, "t": 10, "b": 10},
                      width=table_width, height=table_height)
    render_png(fig, path, dpi=dpi)
    print(f"  wrote {os.path.relpath(path, ROOT)}")


def _run_1m_koh() -> None:
    print("EIS: 1 M KOH HER/OER catalysts")
    report_rows = []
    for condition in ("OCP", "HER", "OER"):
        n = len(EIS_1M_KOH_SAMPLES)
        # Subplot titles carry the sample name *and* its potential -- filled
        # in per-row once the potential is known below -- rather than a
        # separate in-panel annotation, which used to sit on top of the
        # Nyquist data itself in several rows.
        fig = make_subplots(
            rows=n, cols=2, horizontal_spacing=0.14, vertical_spacing=0.12,
            specs=[[{}, {"secondary_y": True}] for _ in range(n)],
            subplot_titles=sum(([f"{label} — Nyquist", f"{label} — Bode"]
                               for label, *_ in EIS_1M_KOH_SAMPLES), []),
        )
        for row, (label, folder, tags) in enumerate(EIS_1M_KOH_SAMPLES, start=1):
            suffix = tags[condition]
            expt_dir = None
            for name in os.listdir(os.path.join(EIS_1M_KOH_DIR, folder)):
                if name.startswith("EIS-POT") and name.rstrip("]").endswith(
                        "#" + suffix):
                    expt_dir = os.path.join(EIS_1M_KOH_DIR, folder, name)
                    break
            if expt_dir is None:
                print(f"  [MISSING] {label} / {condition}")
                continue
            freq, zr, zi, phase = load_eis(expt_dir)
            e_hg = dc_bias_v(expt_dir)
            e_rhe = e_hg + rhe_offset_for(condition)
            # At OCP there is no net Faradaic current driving a resolvable
            # kinetic semicircle -- these spectra are a shallow dip followed
            # almost immediately by a steep capacitive/diffusion rise, and
            # forcing a circle fit onto that shape gives nonsense (negative
            # Ru, Rct in the tens of thousands of ohms, <30 deg of "arc").
            # Fall back to the quick Z'-at-min-|Z''| estimate there, and
            # more generally whenever the circle fit produces an
            # unphysical (negative) Ru regardless of condition -- a
            # nonphysical number is worse than a caveated simpler one.
            a0, a1 = eis.auto_arc_range(zr, zi)
            fit = eis.fit_ru_circle(zr, zi, start=a0, stop=a1)
            fit_note = None
            if condition == "OCP" or fit.ru < 0:
                # The auto-detected window is still used for the zoomed
                # view below; only the fit method changes.
                fit = eis.fit_ru_min_imag(zr[a0:a1], zi[a0:a1])
                fit_note = ("no resolvable kinetic arc at OCP" if condition == "OCP"
                           else "circle fit unphysical; quick Z' @ min|Z''| used")
            report_rows.append({
                "Sample": label, "Condition": condition,
                "E (V vs Hg/HgO)": round(e_hg, 3),
                "E (V vs RHE)": round(e_rhe, 3),
                "Ru (ohm)": round(fit.ru, 2),
                "Rct (ohm)": round(fit.rct, 2) if fit.rct else None,
                "Ru+Rct (ohm)": round(fit.r_low, 2) if fit.r_low else None,
                "Fit RMSE (ohm)": round(fit.rmse, 3) if fit.rmse else None,
                "Arc coverage (deg)": round(fit.arc_coverage_deg, 0)
                                      if fit.arc_coverage_deg else None,
                "Note": fit_note,
            })
            xr, yr = _subplot_axis(row, 1, 2)
            color = PALETTE[(row - 1) % len(PALETTE)]
            zi_abs = np.abs(zi)
            fig.add_trace(go.Scatter(
                x=zr, y=zi_abs, mode="markers", marker={"size": 5, "color": color},
                showlegend=False,
            ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=zr[a0:a1], y=zi_abs[a0:a1], mode="markers",
                marker={"size": 7, "color": "#ff7f0e", "symbol": "circle-open",
                        "line": {"width": 1.5}}, showlegend=False,
            ), row=row, col=1)
            if fit.center is not None:
                cx, cy = eis.circle_path(fit.center, fit.radius)
                fig.add_trace(go.Scatter(
                    x=cx, y=cy, mode="lines",
                    line={"color": "#2ca02c", "dash": "dash", "width": 1.5},
                    showlegend=False,
                ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=[fit.ru], y=[0], mode="markers",
                marker={"size": 10, "color": "red", "symbol": "x"},
                showlegend=False,
            ), row=row, col=1)
            lo, hi, step = app._nice_axis(
                max(zr[a0:a1].max(), zi_abs[a0:a1].max(), fit.ru),
                min(0.0, zr[a0:a1].min()), target=5)
            fig.update_xaxes(range=[lo, hi], dtick=step, row=row, col=1,
                             constrain="domain")
            fig.update_yaxes(range=[lo, hi], dtick=step, row=row, col=1,
                             scaleanchor=xr, scaleratio=1, constrain="domain")

            fig.add_trace(go.Scatter(
                x=freq, y=np.abs(zr + 1j * zi), mode="lines+markers",
                marker={"size": 4}, line={"color": color, "width": 2},
                showlegend=False,
            ), row=row, col=2, secondary_y=False)
            fig.add_trace(go.Scatter(
                x=freq, y=np.degrees(phase), mode="lines+markers",
                marker={"size": 4, "symbol": "diamond"},
                line={"color": "#7f7f7f", "width": 1.5, "dash": "dot"},
                showlegend=False,
            ), row=row, col=2, secondary_y=True)
            # exponentformat="power" renders decade ticks as 10^n (matching
            # standard Bode-plot convention) instead of Plotly's default
            # "1, 100, 10k, 1M" mixed notation.
            fig.update_xaxes(type="log", exponentformat="power",
                             row=row, col=2)
            fig.update_yaxes(type="log", exponentformat="power",
                             row=row, col=2, secondary_y=False)
            fig.update_yaxes(row=row, col=2, secondary_y=True, showgrid=False)
            # Fold the potential into this row's two subplot titles (set by
            # make_subplots above) instead of a floating in-panel
            # annotation -- title space is dedicated and never overlaps
            # data, unlike a fixed (x, y) position that has to guess where
            # a differently-shaped arc leaves room free.
            fig.layout.annotations[2 * (row - 1)].text = (
                f"{label} — Nyquist<br>({e_hg:+.3f} V vs Hg/HgO, "
                f"{e_rhe:+.3f} V vs RHE)")
            fig.layout.annotations[2 * (row - 1) + 1].text = f"{label} — Bode"
            arc_str = (f"{fit.arc_coverage_deg:.0f} deg"
                      if fit.arc_coverage_deg is not None else "n/a")
            note_str = f"  [{fit_note}]" if fit_note else ""
            print(f"  {label:16s} {condition:4s} E={e_hg:+.3f} V vs Hg/HgO "
                 f"({e_rhe:+.3f} V vs RHE)  Ru={fit.ru:.2f} ohm  "
                 f"Rct={fit.rct if fit.rct else float('nan'):.2f} ohm  "
                 f"arc={arc_str}{note_str}")

        for row in range(1, n + 1):
            # Slash notation ("Z' / Ohm"), matching the app's own EIS tab
            # exactly (app.py's render_eis_tab titles its Nyquist axes
            # "Z' / {unit}" / "-Z'' / {unit}") rather than the "(unit)"
            # parenthetical style used for current/potential elsewhere.
            fig.update_xaxes(title={"text": "Z′ / Ω", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=1,
                             **{k: v for k, v in app._BOX_AXIS_STYLE.items()
                                if k not in ("nticks",)})
            fig.update_yaxes(title={"text": "−Z″ / Ω", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=1,
                             **{k: v for k, v in app._BOX_AXIS_STYLE.items()
                                if k not in ("nticks",)})
            fig.update_xaxes(title={"text": "Frequency / Hz", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=2,
                             **app._BOX_AXIS_STYLE)
            fig.update_yaxes(title={"text": "|Z| / Ω", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=2, secondary_y=False,
                             **app._BOX_AXIS_STYLE)
            fig.update_yaxes(title={"text": "Phase / °", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=2, secondary_y=True,
                             showline=True, linewidth=2, linecolor="#7f7f7f",
                             ticks="outside", tickcolor="#7f7f7f")

        for note in fig.layout.annotations:
            if "—" in (note.text or ""):
                note.font = {"family": FAMILY, "size": FONT_SIZE - 10, "color": "black"}
                note.yshift = 10

        cond_label = ("open-circuit potential" if condition == "OCP" else
                     f"{EIS_1M_KOH_POTENTIALS[condition]:+.3f} V vs Hg/HgO")
        row_h = 500
        fig.update_layout(
            template="plotly_white", height=row_h * n + 140, width=PANEL_W * 2.1,
            font=AXIS_FONT, paper_bgcolor="white", plot_bgcolor="white",
            # Extra top margin: the two-line "sample — Nyquist / (potential)"
            # subplot titles need more headroom above row 1 than Plotly's
            # fixed single-line gap gives them, and without it they collide
            # with the figure-level title above.
            margin={"l": 140, "r": 90, "t": 190, "b": 90}, showlegend=False,
            title={"text": f"EIS — 1 M KOH, {condition} ({cond_label})",
                  "font": {"family": FAMILY, "size": FONT_SIZE + 6},
                  "y": 0.995, "yanchor": "top", "x": 0.5, "xanchor": "center"},
        )
        out_path = os.path.join(OUT, f"fig5_EIS_1M-KOH_{condition}.png")
        render_png(fig, out_path, dpi=300)
        print(f"  wrote {os.path.relpath(out_path, ROOT)}")

    df = pd.DataFrame(report_rows)
    csv_path = os.path.join(OUT, "eis_1M-KOH_fit_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"  wrote {os.path.relpath(csv_path, ROOT)}")
    make_table_png(df, os.path.join(OUT, "fig5b_EIS_1M-KOH_fit_table.png"))


def rhe_offset_for(condition: str) -> float:
    return RHE_1M_KOH


def _run_orr_eis() -> None:
    print("EIS: 0.1 M KOH ORR catalysts (RRDE LFO-ReS2)")
    print("  [MISSING] 20% ReS2-LaFeO3: only an .opju Origin file exists, no raw CSVs")

    conditions = sorted({c for _, _, tags in ORR_EIS_SAMPLES for c in tags},
                        key=lambda c: float(c.split()[0]))
    report_rows = []
    for condition in conditions:
        rows_here = [(label, folder, tags[condition]) for label, folder, tags
                    in ORR_EIS_SAMPLES if condition in tags]
        n = len(rows_here)
        fig = make_subplots(
            rows=n, cols=2, horizontal_spacing=0.14, vertical_spacing=0.12,
            specs=[[{}, {"secondary_y": True}] for _ in range(n)],
            subplot_titles=sum(([f"{label} — Nyquist", f"{label} — Bode"]
                               for label, *_ in rows_here), []),
        )
        for row, (label, folder, suffix) in enumerate(rows_here, start=1):
            expt_dir = os.path.join(ORR_EIS_DIR, folder, suffix)
            freq, zr, zi, phase = load_eis(expt_dir)
            e_hg = dc_bias_v(expt_dir)
            e_rhe = e_hg + RHE_01M_KOH
            a0, a1 = eis.auto_arc_range(zr, zi)
            fit = eis.fit_ru_circle(zr, zi, start=a0, stop=a1)
            fit_note = None
            if fit.ru < 0 or (fit.arc_coverage_deg or 0) < 30:
                fit = eis.fit_ru_min_imag(zr[a0:a1], zi[a0:a1])
                fit_note = "circle fit unreliable (<30 deg arc); quick Z' @ min|Z''| used"
            report_rows.append({
                "Sample": label, "Condition (V vs Hg/HgO)": condition,
                "E (V vs RHE)": round(e_rhe, 3),
                "Ru (ohm)": round(fit.ru, 2),
                "Rct (ohm)": round(fit.rct, 2) if fit.rct else None,
                "Ru+Rct (ohm)": round(fit.r_low, 2) if fit.r_low else None,
                "Fit RMSE (ohm)": round(fit.rmse, 3) if fit.rmse else None,
                "Arc coverage (deg)": round(fit.arc_coverage_deg, 0)
                                      if fit.arc_coverage_deg else None,
                "Note": fit_note,
            })
            xr, yr = _subplot_axis(row, 1, 2)
            color = PALETTE[(row - 1) % len(PALETTE)]
            zi_abs = np.abs(zi)
            fig.add_trace(go.Scatter(x=zr, y=zi_abs, mode="markers",
                                     marker={"size": 5, "color": color},
                                     showlegend=False), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=zr[a0:a1], y=zi_abs[a0:a1], mode="markers",
                marker={"size": 7, "color": "#ff7f0e", "symbol": "circle-open",
                        "line": {"width": 1.5}}, showlegend=False,
            ), row=row, col=1)
            if fit.center is not None:
                cx, cy = eis.circle_path(fit.center, fit.radius)
                fig.add_trace(go.Scatter(
                    x=cx, y=cy, mode="lines",
                    line={"color": "#2ca02c", "dash": "dash", "width": 1.5},
                    showlegend=False,
                ), row=row, col=1)
            fig.add_trace(go.Scatter(
                x=[fit.ru], y=[0], mode="markers",
                marker={"size": 10, "color": "red", "symbol": "x"},
                showlegend=False,
            ), row=row, col=1)
            lo, hi, step = app._nice_axis(
                max(zr[a0:a1].max(), zi_abs[a0:a1].max(), fit.ru),
                min(0.0, zr[a0:a1].min()), target=5)
            fig.update_xaxes(range=[lo, hi], dtick=step, row=row, col=1,
                             constrain="domain")
            fig.update_yaxes(range=[lo, hi], dtick=step, row=row, col=1,
                             scaleanchor=xr, scaleratio=1, constrain="domain")

            fig.add_trace(go.Scatter(
                x=freq, y=np.abs(zr + 1j * zi), mode="lines+markers",
                marker={"size": 4}, line={"color": color, "width": 2},
                showlegend=False,
            ), row=row, col=2, secondary_y=False)
            fig.add_trace(go.Scatter(
                x=freq, y=np.degrees(phase), mode="lines+markers",
                marker={"size": 4, "symbol": "diamond"},
                line={"color": "#7f7f7f", "width": 1.5, "dash": "dot"},
                showlegend=False,
            ), row=row, col=2, secondary_y=True)
            # exponentformat="power" renders decade ticks as 10^n (matching
            # standard Bode-plot convention) instead of Plotly's default
            # "1, 100, 10k, 1M" mixed notation.
            fig.update_xaxes(type="log", exponentformat="power",
                             row=row, col=2)
            fig.update_yaxes(type="log", exponentformat="power",
                             row=row, col=2, secondary_y=False)
            fig.update_yaxes(row=row, col=2, secondary_y=True, showgrid=False)

            fig.layout.annotations[2 * (row - 1)].text = (
                f"{label} — Nyquist<br>({e_hg:+.3f} V vs Hg/HgO, "
                f"{e_rhe:+.3f} V vs RHE)")
            fig.layout.annotations[2 * (row - 1) + 1].text = f"{label} — Bode"
            arc_str = (f"{fit.arc_coverage_deg:.0f} deg"
                      if fit.arc_coverage_deg is not None else "n/a")
            note_str = f"  [{fit_note}]" if fit_note else ""
            print(f"  {label:16s} {condition:8s} E={e_hg:+.3f} V vs Hg/HgO "
                 f"({e_rhe:+.3f} V vs RHE)  Ru={fit.ru:.2f} ohm  "
                 f"Rct={fit.rct if fit.rct else float('nan'):.2f} ohm  "
                 f"arc={arc_str}{note_str}")

        for row in range(1, n + 1):
            fig.update_xaxes(title={"text": "Z′ / Ω", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=1,
                             **{k: v for k, v in app._BOX_AXIS_STYLE.items()
                                if k not in ("nticks",)})
            fig.update_yaxes(title={"text": "−Z″ / Ω", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=1,
                             **{k: v for k, v in app._BOX_AXIS_STYLE.items()
                                if k not in ("nticks",)})
            fig.update_xaxes(title={"text": "Frequency / Hz", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=2,
                             **app._BOX_AXIS_STYLE)
            fig.update_yaxes(title={"text": "|Z| / Ω", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=2, secondary_y=False,
                             **app._BOX_AXIS_STYLE)
            fig.update_yaxes(title={"text": "Phase / °", "font": TITLE_FONT},
                             tickfont=AXIS_FONT, row=row, col=2, secondary_y=True,
                             showline=True, linewidth=2, linecolor="#7f7f7f",
                             ticks="outside", tickcolor="#7f7f7f")

        for note_ in fig.layout.annotations:
            if "—" in (note_.text or ""):
                note_.font = {"family": FAMILY, "size": FONT_SIZE - 10, "color": "black"}
                note_.yshift = 10

        row_h = 500
        fig.update_layout(
            template="plotly_white", height=row_h * n + 140, width=PANEL_W * 2.1,
            font=AXIS_FONT, paper_bgcolor="white", plot_bgcolor="white",
            margin={"l": 140, "r": 90, "t": 190, "b": 90}, showlegend=False,
            title={"text": f"EIS — 0.1 M KOH ORR, {condition} vs Hg/HgO "
                          f"({float(condition.split()[0]) + RHE_01M_KOH:+.3f} V vs RHE)",
                  "font": {"family": FAMILY, "size": FONT_SIZE + 6},
                  "y": 0.995, "yanchor": "top", "x": 0.5, "xanchor": "center"},
        )
        tag = condition.replace(" ", "").replace(".", "p")
        out_path = os.path.join(OUT, f"fig6_EIS_ORR_0p1M-KOH_{tag}.png")
        render_png(fig, out_path, dpi=300)
        print(f"  wrote {os.path.relpath(out_path, ROOT)}")

    df = pd.DataFrame(report_rows)
    csv_path = os.path.join(OUT, "eis_ORR_0.1M-KOH_fit_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"  wrote {os.path.relpath(csv_path, ROOT)}")
    make_table_png(df, os.path.join(OUT, "fig6b_EIS_ORR_fit_table.png"))


if __name__ == "__main__":
    _run_1m_koh()
    _run_orr_eis()
